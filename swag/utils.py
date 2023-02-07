import itertools
import torch
import os
import copy
from datetime import datetime
import math
import numpy as np
import tqdm

import torch.nn.functional as F

from .interpretability_methods import gradcam, vanilla_gradients, integrated_gradients


def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList


def LogSumExp(x, dim=0):
    m, _ = torch.max(x, dim=dim, keepdim=True)
    return m + torch.log((x - m).exp().sum(dim=dim, keepdim=True))


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(dir, epoch, name="checkpoint", **kwargs):
    state = {"epoch": epoch}
    state.update(kwargs)
    filepath = os.path.join(dir, "%s-%d.pt" % (name, epoch))
    torch.save(state, filepath)


def save_model_sample_checkpoint(dir, sample_id, name="swag_sample", **kwargs):
    state = {"sample_id": sample_id}
    state.update(kwargs)
    filepath = os.path.join(dir, "%s_%d.pt" % (name, sample_id))
    torch.save(state, filepath)


def train_epoch(
    loader,
    model,
    criterion,
    optimizer,
    cuda=True,
    regression=False,
    verbose=False,
    subset=None,
):
    loss_sum = 0.0
    correct = 0.0
    verb_stage = 0

    num_objects_current = 0
    num_batches = len(loader)

    model.train()

    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches)

    # for i, (input, target) in enumerate(loader):

    for i, (input, target, category, image_path,image_id) in enumerate(loader):
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        loss, output = criterion(model, input, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.data.item() * input.size(0)

        if not regression:
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        num_objects_current += input.size(0)

        if verbose and 10 * (i + 1) / num_batches >= verb_stage + 1:
            print(
                "Stage %d/10. Loss: %12.4f. Acc: %6.2f"
                % (
                    verb_stage + 1,
                    loss_sum / num_objects_current,
                    correct / num_objects_current * 100.0,
                )
            )
            verb_stage += 1

    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": None if regression else correct / num_objects_current * 100.0,
    }


def eval(loader, model, criterion, cuda=True, regression=False, verbose=False):
    loss_sum = 0.0
    correct = 0.0
    num_objects_total = len(loader.dataset)

    model.eval()

    with torch.no_grad():
        if verbose:
            loader = tqdm.tqdm(loader)
        # for i, (input, target) in enumerate(loader):
        for i, (input, target, category, image_path,image_id) in enumerate(loader):

            if cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            loss, output = criterion(model, input, target)

            loss_sum += loss.item() * input.size(0)

            if not regression:
                pred = output.data.argmax(1, keepdim=True)
                correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        "loss": loss_sum / num_objects_total,
        "accuracy": None if regression else correct / num_objects_total * 100.0,
    }


def calc_saliency_maps(loader, model, subset=None):

    model.eval()

    num_batches = len(loader)

    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    #print(model.base.layer4[-1])
    saliency_method = gradcam.GradCAM(model, model.base.layer4)

    gradcam_list = []
    for input, target in tqdm.tqdm(loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        upsampled_gradcam, gradcam_map = saliency_method.get_saliency(input)

        # upsampled_shape = upsampled_gradcam.shape
        # upsampled_gradcam = upsampled_gradcam.reshape((upsampled_shape[0], -1))
        # upsampled_gradcam_min = upsampled_gradcam.min(1)
        # upsampled_gradcam_max = upsampled_gradcam.max(1)
        # upsampled_gradcam_norm = (upsampled_gradcam_map - upsampled_gradcam_min[:, None]) / (
        #     upsampled_gradcam_max[:, None] - upsampled_gradcam_min[:, None] + 1e-8
        # )
        # upsampled_gradcam_norm = upsampled_gradcam_norm.reshape(upsampled_shape)

        # gradcam_list.append(upsampled_gradcam_norm)

        gradcam_list.append(gradcam_map)

    return gradcam_list


# predict for food101
def predict(loader, model, sample_id, verbose=False):
    predictions = list()
    targets = list()

    model.eval()

    print(model)

    # if verbose:
    #     loader = tqdm.tqdm(loader)

    swag_prediction_list = []


    offset = 0
    with torch.no_grad():

        # for loader_idx, (input, target) in enumerate(loader): # start from category 0

        for i, (input, target, category, image_path,image_id) in enumerate(loader):

            input = input.cuda(non_blocking=True)

            output = model(input)

            batch_size = input.size(0)
            #print('batch size : ',batch_size) 1
            
            prediction = F.softmax(output, dim=1).cpu().numpy()
            # print('prediction : ',prediction) [[...]]

            swag_prediction_list.append({'image_path':image_path[0],'label_id':int(target.item()),'label':category[0],'image_id':int(image_id.item()),'predict_softmax':prediction[0].tolist(),'sample_id':sample_id})

            predictions.append(prediction)

            targets.append(target.numpy())

            offset += batch_size
        
    # return {"predictions": np.vstack(predictions), "targets": np.concatenate(targets)}

    return {"result_dict": swag_prediction_list, "predictions": np.vstack(predictions),"targets": np.concatenate(targets)}


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:

            loader = tqdm.tqdm(loader, total=num_batches)

        print('loader : ',loader)
        for input, _, _, _, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def inv_softmax(x, eps=1e-10):
    return torch.log(x / (1.0 - x + eps))


def predictions(test_loader, model, seed=None, cuda=True, regression=False, **kwargs):
    # will assume that model is already in eval mode
    # model.eval()
    preds = []
    targets = []
    for input, target ,_ ,_ in test_loader:
        if seed is not None:
            torch.manual_seed(seed)
        if cuda:
            input = input.cuda(non_blocking=True)
        output = model(input, **kwargs)
        if regression:
            preds.append(output.cpu().data.numpy())
        else:
            probs = F.softmax(output, dim=1)
            preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def schedule(epoch, lr_init, epochs, swa, swa_start=None, swa_lr=None):
    t = (epoch) / (swa_start if swa else epochs)
    lr_ratio = swa_lr / lr_init if swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_init * factor