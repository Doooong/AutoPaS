import copy

import numpy as np
import torch.fx as fx
import torch_pruning as tp
import torch


def get_rm_names(dbc_model, dbc_weights, bn_names):
    remove_name = {}
    fx_model = fx.symbolic_trace(dbc_model)
    for node in fx_model.graph.nodes:
        if node.name in bn_names:
            # remove_name.append(node.prev.name)
            remove_name[node.prev.name] = node.name
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
        # if len(name.split('_')) == 3:
        #     layer_name, index, conv_name = name.split('_')
        #     prune_op = getattr(getattr(ori_model, layer_name)[int(index)], conv_name)
        # else:
        #     prune_op = getattr(ori_model, name)
        prune_op = get_model_op(ori_model, name)
        bn_op = get_model_op(ori_model, remove_name[name])
        # tp.prune_conv_out_channels(getattr(net, name), idxs=idxes[i])
        # import pdb; pdb.set_trace()
        if prune_op.weight.shape[0] == len(idxes[i]):
            # print(prune_op)
            import pdb;pdb.set_trace()
            prune_op.weight.data = torch.zeros_like(prune_op.weight.data).to(prune_op.weight.device)
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
    if len(name.split('_')) == 3:
        layer_name, index, conv_name = name.split('_')
        op = getattr(getattr(model, layer_name)[int(index)], conv_name)
    else:
        op = getattr(model, name)
    return op


def rebuild_model(ori_model, dbc_model, dbc_weights, bn_names, input_shape=None):
    remove_name, idxes = get_rm_names(dbc_model, dbc_weights, bn_names)
    # remove_name = [name.replace('_', '.') for name in remove_name]
    state_dict = copy.deepcopy({k: v for k, v in dbc_weights.items() if k in ori_model.state_dict().keys()})
    # import pdb;pdb.set_trace()
    ori_model.load_state_dict(state_dict)
    ori_model = tp_rebuild(ori_model, remove_name, idxes, input_shape=input_shape)
    return ori_model
