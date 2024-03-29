import torch
import torch.nn as nn
import torch.nn.functional as F
from .prunable_conv import (
    PrunableWeights,
    PrunableConv3d,
    PrunableConv2d,
    PrunableDeconv3d,
    PrunableDeconv2d,
    PrunableLinear,
)

import os, copy, types, time, json
import numpy as np
from monai_ex.utils.misc import Print


def snip_forward_conv2d(self, x):
    return F.conv2d(
        x,
        self.weight * self.weight_mask,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )


def snip_forward_conv3d(self, x):
    # Print('W shape:', self.weight.shape, 'WM shape:', self.weight_mask.shape, color='y')
    return F.conv3d(
        x,
        self.weight * self.weight_mask,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )


def snip_forward_deconv2d(self, x):
    return F.conv_transpose2d(
        x,
        self.weight * self.weight_mask,
        self.bias,
        self.stride,
        self.padding,
        self.output_padding,
    )


def snip_forward_deconv3d(self, x):
    return F.conv_transpose3d(
        x,
        self.weight * self.weight_mask,
        self.bias,
        self.stride,
        self.padding,
        self.output_padding,
    )


def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)


def apply_prune_mask(net, keep_masks, device, verbose=False):
    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, PrunableWeights), net.modules()
    )

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        Print(
            "Prune layer:",
            layer,
            "\tPruned unit size:",
            np.count_nonzero((keep_mask == 0).cpu().numpy()),
            verbose=verbose,
            color="y",
        )
        layer.set_pruning_mask(keep_mask.to(device))

    return net


def _count_diff(w0, w1):
    assert w0.shape == w1.shape, "Diff parms shape!"
    diff_num = torch.sum(torch.abs(w0 - w1) > 0)
    return diff_num


def SNIP(
    input_net,
    prepare_batch_fn,
    loss_fn,
    keep_ratio,
    train_dataloader,
    device="cpu",
    output_dir=None,
):
    # TODO: shuffle?
    # Grab a single batch from the training dataset
    batchdata = next(iter(train_dataloader))
    batch = prepare_batch_fn(batchdata, device, False)
    if len(batch) == 2:
        inputs, targets = batch
    else:
        raise NotImplementedError

    if isinstance(inputs, (tuple, list)):  # multiple inputs
        spatial_ndim = inputs[0].ndim - 2
    else:
        spatial_ndim = inputs.ndim - 2  # assume inputs dim [BCHWD]or[BCHW]
    if spatial_ndim not in [2, 3]:
        raise ValueError(
            f"Currently only support 2&3D data, but got dim={spatial_ndim}"
        )

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(input_net).to(device)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, PrunableWeights):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        if spatial_ndim == 3:
            snip_conv_forward = snip_forward_conv3d
            snip_deconv_forward = snip_forward_deconv3d
        elif spatial_ndim == 2:
            snip_conv_forward = snip_forward_conv2d
            snip_deconv_forward = snip_forward_deconv2d
        else:
            raise NotImplementedError

        # Override the forward methods:
        if isinstance(layer, (PrunableConv3d, PrunableConv2d)):
            layer.forward = types.MethodType(snip_conv_forward, layer)

        if isinstance(layer, (PrunableDeconv3d, PrunableDeconv2d)):
            layer.forward = types.MethodType(snip_deconv_forward, layer)

        if isinstance(layer, PrunableLinear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    if outputs.shape != targets.shape and 1 in outputs.shape:
        outputs.squeeze_()
    loss = loss_fn(outputs, targets)
    loss.backward()

    grads_abs = []
    for layer in net.modules():
        if isinstance(layer, PrunableWeights):
            # Print('Layer:', layer, 'weight shape:', layer.weight_mask.shape, color='r')
            grads_abs.append(torch.abs(layer.weight_mask.grad))
        # if isinstance(layer, nn.BatchNorm3d):
        #    Print('BN:', layer, 'bn shape:', layer.weight.shape, color='y')
    assert len(grads_abs) != 0, "No prunable layer defined in the network"

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    if keep_ratio > 0:
        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
    else:
        acceptable_score = np.mean(all_scores)

    keep_masks = []
    for g in grads_abs:
        msk = (g / norm_factor) >= acceptable_score
        if msk.any():
            keep_masks.append(msk.float())
        else:
            onehot = torch.zeros(len(msk))
            keep_masks.append(onehot.scatter_(0, torch.argmax(g), 1).float())

    # print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))
    Print(
        "Scores min:",
        torch.min(all_scores),
        "Scores max:",
        torch.max(all_scores),
        "Scores mean:",
        torch.mean(all_scores),
        color="y",
    )
    Print(
        "Keep_masks ratio:",
        [
            f"{np.count_nonzero(m.cpu().numpy())/m.cpu().numpy().size:0.2f}"
            for m in keep_masks
        ],
        color="y",
    )
    if output_dir and os.path.isdir(output_dir):
        np.save(
            os.path.join(
                output_dir, "snip_w_scores_{}.npy".format(time.strftime("%H%M"))
            ),
            all_scores.cpu().numpy(),
        )

    return keep_masks


def cSNIP(
    input_net,
    loss_fn,
    keep_ratio,
    train_dataloader,
    min_chs=3,
    use_cuda=True,
    output_dir=None,
):
    # TODO: shuffle?

    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(input_net)

    if use_cuda:
        inputs = inputs.cuda()
        targets = targets.cuda()
        net = net.cuda()

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, (PrunableConv3d, PrunableConv2d)):
            # Print('Layer w dim:', layer.weight.shape, color='y')
            layer.weight_mask = (
                nn.Parameter(torch.ones([layer.weight.shape[0], 1, 1, 1, 1]).cuda())
                if use_cuda
                else nn.Parameter(torch.ones([layer.weight.shape[0], 1, 1, 1, 1]))
            )
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False
        elif isinstance(layer, (PrunableDeconv3d, PrunableDeconv2d)):
            layer.weight_mask = (
                nn.Parameter(torch.ones([1, layer.weight.shape[1], 1, 1, 1]).cuda())
                if use_cuda
                else nn.Parameter(torch.ones([1, layer.weight.shape[1], 1, 1, 1]))
            )
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        if spatial_ndim == 3:
            snip_conv_forward = snip_forward_conv3d
            snip_deconv_forward = snip_forward_deconv3d
        elif spatial_ndim == 2:
            snip_conv_forward = snip_forward_conv2d
            snip_deconv_forward = snip_forward_deconv2d

        # Override the forward methods:
        if isinstance(layer, (PrunableConv3d, PrunableConv2d)):
            layer.forward = types.MethodType(snip_forward_conv3d, layer)

        if isinstance(layer, (PrunableDeconv3d, PrunableDeconv2d)):
            layer.forward = types.MethodType(snip_forward_deconv3d, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    grads_abs, idx = [], []
    for i, layer in enumerate(net.modules()):
        if isinstance(layer, PrunableWeights):
            grads_abs.append(torch.abs(torch.squeeze(layer.weight_mask.grad)))
            idx.append(i)
            # Print('Layer:', layer, 'weight shape:', layer.weight.shape, color='r')
        # if isinstance(layer, nn.BatchNorm3d):
        # Print('BN:', layer, 'bn shape:', layer.weight.shape, color='g')

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)
    Print(
        "Scores min:",
        torch.min(all_scores),
        "Scores max:",
        torch.max(all_scores),
        "Scores mean:",
        torch.mean(all_scores),
        color="y",
    )
    if output_dir and os.path.isdir(output_dir):
        with open(os.path.join(output_dir, "snip_chs_scores.json"), "w") as f:
            json.dump(all_scores.cpu().numpy().tolist(), f)

    if keep_ratio > 0:
        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
    else:
        acceptable_score = np.mean(all_scores)

    keep_masks = []
    for i, g in enumerate(grads_abs):
        if 1 < i < len(grads_abs) - 1:
            msk = (g / norm_factor) >= acceptable_score
            if torch.sum(msk) >= min_chs:
                keep_masks.append(msk.cpu().float())
            else:
                ids = torch.topk(g, k=min_chs)[1]
                keep_masks.append(torch.zeros(len(g)).scatter_(0, ids.cpu(), 1))
        else:  # keep last conv channel num
            msk = torch.ones(len(g))
            keep_masks.append(msk)

    if output_dir and os.path.isdir(output_dir):
        out_mask = [m.numpy().tolist() for m in keep_masks]
        with open(
            os.path.join(output_dir, "snip_ch_mask_{}.json".format(keep_ratio)), "w"
        ) as f:
            json.dump(out_mask, f)

    remains = torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks]))
    Print("Remain #{} channels".format(remains), color="g")

    return keep_masks
