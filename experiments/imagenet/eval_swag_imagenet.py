import argparse
import os
import random
import sys
import time
import tabulate
from pathlib import Path

import numpy as np
import json

import torch
import torch.nn.functional as F
import torchvision.models

from data_finetune import *
from dataset import *

#from swag import utils_orig as utils
from swag import utils,losses
from swag.posteriors import SWAG

parser = argparse.ArgumentParser(description="SGD/SWA training")

parser.add_argument(
    "--data_path",
    type=str,
    default='../../food-101/',
    # default='../../imagenet',
    required=False,
    metavar="PATH",
    help="path to datasets location (default: None)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    metavar="N",
    help="input batch size (default: 256)",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    metavar="N",
    help="number of workers (default: 4)",
)
parser.add_argument(
    "--model",
    type=str,
    default='resnet50',
    required=False,
    metavar="MODEL",
    help="model name (default: None)",
)

parser.add_argument(
    "--ckpt",
    type=str,
    required=True,
    default=None,
    metavar="CKPT",
    help="checkpoint to load (default: None)",
)

parser.add_argument(
    "--num_samples",
    type=int,
    default=100,
    metavar="N",
    help="number of samples for SWAG (default: 30)",
)
parser.add_argument(
    "--saliency",
    action="store_true",
)

parser.add_argument("--scale", type=float, default=1.0, help="SWAG scale")
parser.add_argument("--cov_mat", action="store_true", help="save sample covariance")
parser.add_argument(
    "--use_diag_bma", action="store_true", help="sample only diag variacne for BMA"
)

parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)

parser.add_argument(
    "--save_path_swa",
    type=str,
    default='save_swa',
    required=False,
    help="path to SWA npz results file",
)
parser.add_argument(
    "--save_path_swag",
    type=str,
    default='save_swag',
    required=False,
    help="path to SWAG npz results file",
)

args = parser.parse_args()

eps = 1e-12

args.device = None
if torch.cuda.is_available():
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print("Using model %s" % args.model)
model_class = getattr(torchvision.models, args.model)

print("Loading ImageNet from %s" % (args.data_path))
# loaders, num_classes = data.loaders(args.data_path, args.batch_size, args.num_workers)

train_df_path = '../../food-101/finetune_food101_train.csv'
test_df_path = '../../food-101/finetune_food101_test.csv'
CHOOSED_CLASSES = ['french_toast', 'greek_salad', 'caprese_salad', 'chocolate_cake', 'pizza', 'cup_cakes', 'carrot_cake','cheesecake','pancakes', 'strawberry_shortcake']

loaders = data_loaders(train_df_path, test_df_path, train_batch_size=args.batch_size, test_batch_size=1)

num_classes=10

# x_train,y_train,x_test,y_test=load_data_from_folder()
# train_ld=Data_Loader(x_train,y_train,is_train=True)
# test_ld=Data_Loader(x_test,y_test,is_train=False)
# loaders={"train": train_ld,"test": test_ld}


print("Preparing model")
swag_model = SWAG(
    model_class,
    no_cov_mat=not args.cov_mat,
    # loading=True,
    max_num_models=20,
    num_classes=num_classes,
)

swag_model.to(args.device)

criterion = losses.cross_entropy

print("Loading checkpoint %s" % args.ckpt)
checkpoint = torch.load(args.ckpt)
swag_model.load_state_dict(checkpoint["state_dict"])

# mean, var, cov_mat_list = swag_model.export_numpy_params(True)
print(type(swag_model.named_parameters()))
# w=swag_model.export_numpy_params(swag_model,'base.fc.weight')#layer1.0.bn1.weight

# np.save('layer_var.npy',w[1])
# np.save('layer_cov_mat.npy',w[2][0])


print("SWA")
swag_model.sample(0.0)
print("SWA BN update")
utils.bn_update(loaders["train"], swag_model, verbose=True, subset=0.1)
print("SWA EVAL")
swa_res = utils.predict(loaders["test"], swag_model, None, verbose=True)

targets = swa_res["targets"]
swa_predictions = swa_res["predictions"]

swa_accuracy = np.mean(np.argmax(swa_predictions, axis=1) == targets)
swa_nll = -np.mean(
    np.log(swa_predictions[np.arange(swa_predictions.shape[0]), targets] + eps)
)
print("SWA. Accuracy: %.2f%% NLL: %.4f" % (swa_accuracy * 100, swa_nll))
swa_entropies = -np.sum(np.log(swa_predictions + eps) * swa_predictions, axis=1)

# np.savez(
#     args.save_path_swa,
#     accuracy=swa_accuracy,
#     nll=swa_nll,
#     entropies=swa_entropies,
#     predictions=swa_predictions,
#     targets=targets,
# )

print("SWAG")

save_dir = Path("../../food-101/json_results")
save_dir2 = Path("../../food-101/json_results2")
save_dir.mkdir()
save_dir2.mkdir()

swag_predictions = np.zeros((len(loaders["test"].dataset), num_classes))


for i in range(args.num_samples):


    print('sample i : ',i)
    swag_model.sample(args.scale, cov=args.cov_mat and (not args.use_diag_bma))

    print("SWAG Sample %d/%d. BN update" % (i + 1, args.num_samples))
    utils.bn_update(loaders["train"], swag_model, verbose=True, subset=0.1)
    print("SWAG Sample %d/%d. EVAL" % (i + 1, args.num_samples))

    # activations_all = []
    # def store_activations(module, input, output):

    #     global activations_all

    #     # print('output shape : ',output.shape) # (8,2048,7,7) (batch size, num_channels，7，7)

    #     activations = F.avg_pool2d(output, kernel_size=7) # (8,2048,1,1)

    #     # print('activations shape : ',activations.shape)

    #     activations = output.detach().cpu()[:,:,0,0].tolist()

    #     # print('output size: ',output.shape) #(batch size, 2048, 1,1); 2048 is the dimension of avgpool output
    #     # print('activations shape : ',len(activations),len(activations[0]),activations[0]) (8, 2048) (batch_size, num of channels)
    #     activations_all.extend(activations)

    # hook = swag_model.base.layer4[1].register_forward_hook(store_activations)

    res = utils.predict(loaders["test"], swag_model, sample_id=i, verbose=True)
    predictions = res["predictions"]
    prediction_result_dict = res["result_dict"]

    json.dump(prediction_result_dict,open("../../food-101/json_results2/result_dict_"+str(i)+'.json','w'))



    # json.dump(activations_all,open(save_dir / f'activations_{i:03d}.json','w'))

    # hook.remove()
    # print('predictions : ',predictions)
    # print('predictions shape : ',predictions.shape) #(N,num_classes) for food101: (1000,10)
    # print('activations all shape : ',len(activations_all),len(activations_all[0])) (750,2048) (N,num of channels)

    # print('saliency map')
    # saliency_maps = utils.calc_saliency_maps(loaders["test"], swag_model)
    # torch.save(saliency_maps, save_dir / f"saliency_maps_{i:03d}.pth")

    # print('accuracy')
    # accuracy = np.mean(np.argmax(predictions, axis=1) == targets)
    # nll = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), targets] + eps))
    # print(
    #     "SWAG Sample %d/%d. Accuracy: %.2f%% NLL: %.4f"
    #     % (i + 1, args.num_samples, accuracy * 100, nll)
    # )

    # swag_predictions += predictions

    torch.save(predictions, save_dir / f"predictions_{i:03d}.pth")


    # ens_accuracy = np.mean(np.argmax(swag_predictions, axis=1) == targets)
    # ens_nll = -np.mean(
    #     np.log(
    #         swag_predictions[np.arange(swag_predictions.shape[0]), targets] / (i + 1)
    #         + eps
    #     )
    # )
    # print(
    #     "Ensemble %d/%d. Accuracy: %.2f%% NLL: %.4f"
    #     % (i + 1, args.num_samples, ens_accuracy * 100, ens_nll)
    # )

# swag_predictions /= args.num_samples

# swag_accuracy = np.mean(np.argmax(swag_predictions, axis=1) == targets)
# swag_nll = -np.mean(
#     np.log(swag_predictions[np.arange(swag_predictions.shape[0]), targets] + eps)
# )
# swag_entropies = -np.sum(np.log(swag_predictions + eps) * swag_predictions, axis=1)

# np.savez(
#     args.save_path_swag,
#     accuracy=swag_accuracy,
#     nll=swag_nll,
#     entropies=swag_entropies,
#     predictions=swag_predictions,
#     targets=targets,
# )