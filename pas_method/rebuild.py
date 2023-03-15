import copy
import torch.nn as nn
import numpy as np
import torch.fx as fx
import torch_pruning as tp
import torch
from pas_method import find_node_name, get_model_op, ShapeProp


def find_bn_name(node):
    if 'bn' in node.next.name:
        return node.next.name
    return ''


def get_rm_names(dbc_model, dbc_weights, bn_names):
    remove_name = {}
    fx_model = fx.symbolic_trace(dbc_model)
    for node in fx_model.graph.nodes:
        if node.name in bn_names:
            name = find_node_name(model=dbc_model,node=node, name='batch_norm', prev=False)
            if name == '':
                remove_name[node.name] = {'name': find_node_name(dbc_model,node, 'bn', prev=False)}
            else:
                remove_name[node.name] = {'name': name}
    idxes = []
    dbc_model.load_state_dict(dbc_weights)
    for name, module in dbc_model.named_modules():
        if '.scale' in name:
            w = module.weight.detach()
            binary_w = (w > 0.5).float()
            residual = w - binary_w
            weight = module.weight.detach().cpu().numpy() - residual.detach().cpu().numpy()
            idxes.append(np.where(weight < 0.5)[0].tolist())
            print(f"{weight.shape[0]} - > {weight.shape[0] - len(np.where(weight < 0.5)[0].tolist())}")
    for i, name in enumerate(remove_name.keys()):
        remove_name[name]['idx'] = idxes[i]
    return remove_name


def remove_batch_idx(dbc_model, name):
    fx_model = fx.symbolic_trace(dbc_model)
    batch_list = []
    for node in fx_model.graph.nodes:
        if name == node.name:
            batch_list = [node.name for node in node.all_input_nodes]
            return batch_list
    return batch_list


def check_batch_norma(ori_model, remove_name):
    last_idx = []
    for node in fx.symbolic_trace(ori_model).graph.nodes:
        if 'batch_norm' in node.name and node.op == 'call_function':
            batch_list = [node.name for node in node.all_input_nodes]
            idx = []
            out_channel = -1
            for name in batch_list:
                op = get_model_op(ori_model, name)
                if isinstance(op, torch.Tensor):
                    if out_channel != -1 and op.data.size(0) > out_channel:
                        if len(idx) == 0:
                            idx = last_idx
                        if len(last_idx) == 0:
                            remove_length = op.data.size(0) - out_channel
                            for key in remove_name.keys():
                                if len(remove_name[key]['idx']) == remove_length:
                                    last_idx = remove_name[key]['idx']
                                    idx = last_idx
                                    break
                        op.data[idx] = 0
                        # import pdb; pdb.set_trace()
                        remain_idx = [x for x in list(range(op.size()[0])) if x not in idx]
                        new_data = op.data[remain_idx]
                        op.data = new_data.to(op.data.device)
                else:
                    out_channel = op.out_channels
                    if name in remove_name.keys():
                        idx = remove_name[name]['idx']
                        last_idx = idx
    return ori_model


def tp_rebuild(ori_model, remove_name, input_shape=None):
    DG = tp.DependencyGraph()
    if input_shape is None:
        input_shape = [1, 3, 224, 224]
    example_input = torch.randn(tuple(input_shape)).to('cuda')
    ori_model = ori_model.to('cuda')
    DG.build_dependency(ori_model.eval(), example_inputs=example_input)
    for i, name in enumerate(remove_name.keys()):
        prune_op = get_model_op(ori_model, name)
        # bn_op = get_model_op(ori_model, remove_name[name]['name'])
        if prune_op.weight.shape[0] == len(remove_name[name]['idx']):
            # prune_op.wxeight.data = torch.zeros_like(prune_op.weight.data).to(prune_op.weight.device)
            # prune_op.bias = torch.nn.Parameter(torch.zeros_like(prune_op.bias.data))
        #     if bn_op:
        #         bn_op.weight.data = torch.zeros_like(bn_op.weight.data).to(bn_op.weight.device)
        #         bn_op.running_mean.data = torch.zeros_like(bn_op.running_mean.data).to(bn_op.running_mean.data.device)
        #         bn_op.running_var.data = torch.zeros_like(bn_op.running_var.data).to(bn_op.running_var.data.device)
        #         bn_op.bias.data = torch.zeros_like(bn_op.bias.data).to(bn_op.bias.data.device)
            continue
        pruning_group = DG.get_pruning_group(prune_op,
                                             tp.prune_conv_out_channels, idxs=remove_name[name]['idx'])
        # 3. prune all grouped layer that is coupled with model.conv1
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
    # ori_model = check_batch_norma(ori_model, remove_name)
    return ori_model


def transform_model(ori_model):
    ori_model = fx.symbolic_trace(ori_model)
    ori_model.graph.lint()
    ori_model.recompile()
    return ori_model


def rebuild_model(ori_model, dbc_model, dbc_weights, bn_names, input_shape=None, is_load=True):
    remove_name = get_rm_names(dbc_model, dbc_weights, bn_names)
    ori_model = transform_model(ori_model)
    state_dict = copy.deepcopy({k: v for k, v in dbc_weights.items() if k in ori_model.state_dict().keys()})
    if is_load:
        ori_model.load_state_dict(state_dict)
    ori_model = tp_rebuild(ori_model, remove_name, input_shape=input_shape)
    return ori_model
