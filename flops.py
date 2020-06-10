from __future__ import print_function
import os
import argparse
import torch
import json
import csv
import shutil
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
from utils.model_parse import  mask_decorater
from utils.filter_pruner import filter_pruner
from utils.graph_from_trace import VisualGraph


parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
# parser.add_argument('--sparsity_file', required=True, help='config file for the pruner')
args = parser.parse_args()

cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def forward_hook(module, inputs, output):
    if hasattr(module, 'flops'):
        return
    input=inputs[0]
    g = module.groups
    n, c, _, _ = input.size()
    _, _, h,w= output.size()
    k1, k2 = module.kernel_size
    flops = (1*c*k1*k2 ) * h * w * module.out_channels / g
    module.flops = flops


def get_flops(net, data):
    Flops = {}
    total_flops = 0
    for name, layer in net.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            layer.register_forward_hook(forward_hook)
    net(data)
    for name, layer in net.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            total_flops += layer.flops
            Flops[name] = layer.flops
    print('TOTAL FLOPS', total_flops)
    return Flops


def load_sparsity(filepath):
    jf = open(filepath, 'r')
    tmp_cfg = json.load(jf)
    jf.close()
    for c in json.loads(tmp_cfg["config_list"]):
        print(c['op_names'][0])
        print(c['sparsity'])
        prune_cfg[c['op_names'][0]] = 1.0 - c['sparsity']


def flops_traverse(node, ratio):
    #print(node)
    if node in gf.c2py and gf.c2py[node].isOp and node.kind()=='aten::_convolution':
        # print(node)
        name = gf.c2py[node].name
        if name not in in_prune_ratio:
            in_prune_ratio[name] = ratio
        else:
            in_prune_ratio[name] = max(ratio, in_prune_ratio[name])
        ratio = 1.0
        if name in prune_cfg:
            ratio = prune_cfg[name]
    if node in gf.forward_edge:
        for _next in gf.forward_edge[node]:
            flops_traverse(_next, ratio)


if __name__ == '__main__':
    performances = {}
    flops = {}

    sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]
    pruners = ['NetAdaptPruner005', 'SimulatedAnnealingPruner']
    for pruner in pruners:
        performances[pruner] = []
        flops[pruner] = []
        for sparsity in sparsities:
            with open(os.path.join('experiment_data/archive640/', pruner, str(sparsity).replace('.', ''), 'performance.json'), 'r') as jsonfile:
                performance = json.load(jsonfile)
                performances[pruner].append(performance['finetuned'])

            sparsity_file = os.path.join('experiment_data/archive640/', pruner, str(sparsity).replace('.', ''), 'search_result.json')
            
            # FLOPS calc
            net = RetinaFace(cfg=cfg)
            net = load_model(net, args.trained_model, args.cpu)

            print('Finished loading model!')
            #print(net)
            cudnn.benchmark = False
            device = torch.device("cpu" if args.cpu else "cuda")
            net = net.to(device)
            # data = torch.rand(1, 3, 560, 1024)
            data = torch.rand(1,3, 640, 640)
            data = data.to(device)
            gf = VisualGraph(net,data)
            FLOPS = get_flops(net, data)
            pruned_flops = 0.0
            visited = set()
            in_prune_ratio = {}
            prune_cfg = {}

            load_sparsity(sparsity_file)
            for input in gf.graph.inputs():
                if input.type().kind() == 'ClassType':
                    continue
                flops_traverse(input, 1.0)
            for name in FLOPS:
                #print(prune_cfg[name])
                #print(in_prune_ratio.keys())
                remained = 1.0
                if name in prune_cfg:
                    remained = prune_cfg[name]
                # print(name)
                if name not in in_prune_ratio:
                    in_prune_ratio[name] = 1
                pruned_flops += FLOPS[name] * remained * in_prune_ratio[name]

            flops[pruner].append(int(pruned_flops))


    for pruner in pruners:
        with open(os.path.join('experiment_data/flops_{}.csv'.format(pruner)), 'w+') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['accuracy', 'flops', 'sparsity'])
            for idx, performance in enumerate(performances[pruner]):
                writer.writerow([performance, flops[pruner][idx], sparsities[idx]])
            