import json
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from data_folder import ImageFolder
from sklearn.metrics.pairwise import euclidean_distances

from tqdm import tqdm

import umap


def get_images():
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )

    val_path = Path("../../imagenet/val")
    val_dataset = ImageFolder(
        val_path, transform=transform, used_classes = ['n02093256','n02093647','n02093859', 'n02094114','n02094433', 'n02095570','n02096177', 'n02096585',
'n02093428', 'n02093754', 'n02093991', 'n02094258', 'n02095314','n02095889','n02096294']
    )

    for i, (img, label) in tqdm(enumerate(val_dataset)):
        img.save("../../imagenet/eval_images_dog_new/"+str(i)+".jpeg")


def main():

    result_dir = 'results_layer4_1_relu_avgpool'

    get_images()

    sample_img_dist_list = []

    val_path = Path("../../imagenet/val")
    val_dataset = ImageFolder(val_path, loader=lambda path: path, used_classes = ['n02093256','n02093647','n02093859', 'n02094114','n02094433', 'n02095570','n02096177', 'n02096585',
'n02093428', 'n02093754', 'n02093991', 'n02094258', 'n02095314','n02095889','n02096294'])

    predictions_list = []
    for file_path in Path(result_dir).glob("predictions_*.pth"):
        predictions = torch.load(file_path, map_location="cpu") #(   ,batch_size)
        print('predictions shape : ',predictions.shape) #(N, num_classes)
        predictions_list.append(predictions)
    assert len(predictions_list) == 10
    # N, num_samples, num_classes
    predictions = np.stack(predictions_list, axis=1) #(N, num_samples, num_classes)

    activations_list = []
    for file_path in Path(result_dir).glob("activations_*.json"):
        activations = json.load(open(file_path,'r')) #(N,num_channels)
        activations = np.array(activations)
        activations_list.append(activations)
    assert len(activations_list) == 10

    # N, num_samples, num_channels
    activations = np.stack(activations_list, axis=1)

    print('activations shape : ',activations.shape) #(N, num_samples, num_neurons)

    n_samples = activations.shape[1]
    n_images = activations.shape[0]
    n_neurons = activations.shape[2]

    saliency_maps_list = []
    for file_path in Path(result_dir).glob("saliency_maps_*.pth"):
        saliency_maps = torch.load(file_path, map_location="cpu") 
        saliency_maps = np.concatenate(saliency_maps, axis=0)
        saliency_maps_list.append(saliency_maps)

    # N, num_samples, 1, H, W
    saliency_maps = np.stack(saliency_maps_list, axis=1)
    print(saliency_maps.shape) #(50000, 30, 1, 7, 7) (N,num_samples,1,7,7)

    for i, (path, label) in enumerate(tqdm(val_dataset)):
        # num_samples, num_classes
        prediction = predictions[i]
        # num_samples, H, W
        saliency_map = saliency_maps[i]
        # num_samples, num_neurons
        activation = activations[i]

        with open(result_dir+"/image_"+str(i)+".json",'w') as f:
            json.dump({
                'image_id':i,
                "path": path,
                "eval_image_path": "eval_images_dog_new/"+str(i)+".jpeg",
                "pred": prediction.argmax(1).tolist(),
                "label": label,
                "softmax": prediction.tolist(),
                "activation": activation.tolist(),
                "saliency_map": saliency_map.tolist(),
            }, f)


    umap_model = umap.UMAP(metric='precomputed')
    
    for sample_id in range(n_samples): # predictions.shape[1] = n_samples

        sample_umap_list = []

        sample_activations = np.transpose(activations,(1,0,2))[sample_id] #(N,num_channels)

        sample_dist_matrix = euclidean_distances(sample_activations, sample_activations) #(N,N)

        sample_img_dist_list.append(sample_dist_matrix)

        umap_embeddings = umap_model.fit_transform(sample_dist_matrix)

        print('umap embedding shape : ',umap_embeddings.shape) # (N,2)
        for image_id in range(sample_dist_matrix.shape[0]):

            sample_umap_list.append({

                'x': float(umap_embeddings[image_id][0]),
                'y': float(umap_embeddings[image_id][1]),
                'sample_id': sample_id,
                'image_id': image_id,
                "eval_image_path": "eval_images_dog_new/"+str(image_id)+".jpeg",
                "pred": int(np.argmax(predictions[image_id][sample_id])), # prediction from this sample
                "pred_var":float(np.var(predictions[image_id][sample_id])),
                "label": val_dataset[image_id][1],

                })

        json.dump(sample_umap_list,open('./'+result_dir+'/umap_sample_'+str(sample_id)+'_scatterplot.json','w'))

    umap_avg_list = []
    # umap for marginalized activations
    img_dist_avg = np.mean(sample_img_dist_list,axis=0)

    sample_img_dist_list = [item.tolist() for item in sample_img_dist_list]

    json.dump(sample_img_dist_list,open('./'+result_dir+'/umap_distance_sample_matrix.json','w'))

    umap_model = umap.UMAP(metric='precomputed')
    umap_embeddings_avg = umap_model.fit_transform(img_dist_avg)

    for image_id in range(umap_embeddings_avg.shape[0]):

        umap_avg_list.append({

            'x': float(umap_embeddings_avg[image_id][0]),
            'y': float(umap_embeddings_avg[image_id][1]),
            'image_id': image_id,
            'eval_image_path': 'eval_images_dog_new/'+str(image_id)+'.jpeg',
            'pred': int(np.mean(predictions[image_id],axis=0).argmax(0)),
            "pred_var":float(np.var(np.mean(predictions[image_id],axis=0))),
            "softmax": np.mean(predictions[image_id],axis=0).tolist(),
            'label': val_dataset[image_id][1],
            })

    json.dump(umap_avg_list,open('./'+result_dir+'/umap_sample_avg_scatterplot.json','w'))





if __name__ == "__main__":
    main()




