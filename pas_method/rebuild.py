import copy

import numpy as np
import torch.fx as fx
import torch_pruning as tp
import torch


def find_bn_name(node):
    if 'bn' in node.next.name:
        return node.next.name
    return ''


def get_rm_names(dbc_model, dbc_weights, bn_names):
    remove_name = {}
    fx_model = fx.symbolic_trace(dbc_model)
    for node in fx_model.graph.nodes:
        if node.name in bn_names:
            # remove_name.append(node.prev.name)
            remove_name[node.name] = find_bn_name(node)
    idxes = []
    dbc_model.load_state_dict(dbc_weights)
    for name, module in dbc_model.named_modules():
        if '.scale' in name:
            w = module.weight.detach()
            binary_w = (w > 0.5).float()
            residual = w - binary_w
            # import pdb; pdb.set_trace()
            weight = module.weight.detach().cpu().numpy() - residual.detach().cpu().numpy()
            idxes.append(np.where(weight < 0.5)[0].tolist())
            print(f"{weight.shape[0]} - > {weight.shape[0] - len(np.where(weight < 0.5)[0].tolist())}")
    return remove_name, idxes


def tp_rebuild(ori_model, remove_name, idxes, input_shape=None):
    DG = tp.DependencyGraph()
    if input_shape is None:
        input_shape = [1, 3, 224, 224]
    example_input = torch.randn(tuple(input_shape)).to('cuda')
    DG.build_dependency(ori_model.eval(), example_inputs=example_input)
    for i, name in enumerate(remove_name.keys()):
        prune_op = get_model_op(ori_model, name)
        bn_op = get_model_op(ori_model, remove_name[name])
        if prune_op.weight.shape[0] == len(idxes[i]):
            prune_op.weight.data = torch.zeros_like(prune_op.weight.data).to(prune_op.weight.device)
            if bn_op:
                bn_op.weight.data = torch.zeros_like(bn_op.weight.data).to(bn_op.weight.device)
                bn_op.running_mean.data = torch.zeros_like(bn_op.running_mean.data).to(bn_op.running_mean.data.device)
                bn_op.running_var.data = torch.zeros_like(bn_op.running_var.data).to(bn_op.running_var.data.device)
                bn_op.bias.data = torch.zeros_like(bn_op.bias.data).to(bn_op.bias.data.device)
            continue
        pruning_group = DG.get_pruning_group(prune_op,
                                             tp.prune_conv_out_channels, idxs=idxes[i])
        # 3. prune all grouped layer that is coupled with model.conv1
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
    return ori_model


def get_model_op(model, name):
    if name == '':
        return None
    names = name.split('_')
    # if len(names) == 1:
    #     op = getattr(model, names[0])
    # elif len(names) == 2:
    #     op = get_sub_op(getattr(model, names[0]), names[1])
    # elif len(names) == 3:
    #     op = get_sub_op(get_sub_op(getattr(model, names[0]), names[1]), names[2])
    # else:
    #     op = get_sub_op(get_sub_op(get_sub_op(getattr(model, names[0]), names[1]), names[2]), names[3])
    op = get_sub_op(model, names[0])
    for name in names[1:]:
        op = get_sub_op(op, name)
    return op


def get_sub_op(sub_model, sub_name):
    if sub_name.isnumeric():
        op = sub_model[int(sub_name)]
    else:
        op = getattr(sub_model, sub_name)
    return op


def rebuild_model(ori_model, dbc_model, dbc_weights, bn_names, input_shape=None):
    remove_name, idxes = get_rm_names(dbc_model, dbc_weights, bn_names)
    # remove_name = [name.replace('_', '.') for name in remove_name]
    state_dict = copy.deepcopy({k: v for k, v in dbc_weights.items() if k in ori_model.state_dict().keys()})
    # import pdb;pdb.set_trace()
    ori_model.load_state_dict(state_dict)
    ori_model = tp_rebuild(ori_model, remove_name, idxes, input_shape=input_shape)
    return ori_model
