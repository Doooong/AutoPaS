import copy
import torch.nn as nn
import numpy as np
import torch.fx as fx
import torch_pruning as tp
import torch


# def find_bn_name(node):
#     if 'bn' in node.next.name:
#         return node.next.name
#     return ''
def find_bn_name(node, name):
    if name in node.next.name:
        return node.next.name
    return find_bn_name(node.next, name)


def get_rm_names(dbc_model, dbc_weights, bn_names):
    remove_name = {}
    fx_model = fx.symbolic_trace(dbc_model)
    for node in fx_model.graph.nodes:
        if node.name in bn_names:
            # remove_name.append(node.prev.name)
            name = find_bn_name(node, 'batch')
            if name == '':
                remove_name[node.name] = find_bn_name(node, 'bn')
            else:
                remove_name[node.name] = name
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


def remove_batch_idx(dbc_model, name):
    fx_model = fx.symbolic_trace(dbc_model)
    batch_list = []
    for node in fx_model.graph.nodes:
        if name == node.name:
            batch_list = [node.name for node in node.all_input_nodes]
            return batch_list
    return batch_list


def tp_rebuild(ori_model, remove_name, idxes, input_shape=None):
    DG = tp.DependencyGraph()
    if input_shape is None:
        input_shape = [1, 3, 224, 224]
    example_input = torch.randn(tuple(input_shape)).to('cuda')
    ori_model = ori_model.to('cuda')
    DG.build_dependency(ori_model.eval(), example_inputs=example_input)
    for i, name in enumerate(remove_name.keys()):
        prune_op = get_model_op(ori_model, name)
        bn_op = get_model_op(ori_model, remove_name[name])
        if bn_op == '':
            remove_batch_list = remove_batch_idx(ori_model, remove_name[name])
            if len(remove_batch_list):
                for name in remove_batch_list:
                    op = get_model_op(ori_model, name)
                    if isinstance(op, torch.Tensor):
                        op.data[idxes[i]] = 0
                        remain_idx = np.where(op.data.detach().cpu().numpy() != 0)
                        new_data = op.data[remain_idx]
                        op.data = new_data.to(op.data.device)
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
    try:
        op = getattr(model, name)
    except Exception as e:
        try:
            all_names = name.split('_')
            length = len(all_names)
            op = model
            for i in range(length):
                if i < length - 1:
                    try:
                        if all_names[i + 1].isnumeric():
                            op_name = all_names[i]
                            op = get_sub_op(op, op_name)
                        else:
                            op_name = all_names[i] + '_' + all_names[i + 1]
                            try:
                                op = get_sub_op(op, op_name)
                                i += 2
                                if i >= length:
                                    break
                            except:
                                op_name = all_names[i]
                                op = get_sub_op(op, op_name)
                    except:
                        op = ''
                elif i > length - 1:
                    break
                else:
                    op = get_sub_op(op, all_names[i])
        except Exception as e:
            op = ''
    return op


def get_sub_op(sub_model, sub_name):
    if sub_name.isnumeric():
        if isinstance(sub_model, nn.Module):
            op = list(sub_model.children())[int(sub_name)]
        else:
            op = sub_model[int(sub_name)]
    else:
        op = getattr(sub_model, sub_name)
    return op


def transform_model(ori_model):
    ori_model = fx.symbolic_trace(ori_model)
    ori_model.graph.lint()
    ori_model.recompile()
    return ori_model


def rebuild_model(ori_model, dbc_model, dbc_weights, bn_names, input_shape=None):
    remove_name, idxes = get_rm_names(dbc_model, dbc_weights, bn_names)
    ori_model = transform_model(ori_model)
    state_dict = copy.deepcopy({k: v for k, v in dbc_weights.items() if k in ori_model.state_dict().keys()})
    ori_model.load_state_dict(state_dict)
    ori_model = tp_rebuild(ori_model, remove_name, idxes, input_shape=input_shape)
    return ori_model
