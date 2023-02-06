import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from typing import Any, Callable, Dict, Optional, Tuple
from torch.fx.node import Node


# from torchvision.models.feature_extraction import get_graph_node_names
# from torchvision.models.feature_extraction import create_feature_extractor

def _parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name


def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    # assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)

    print("replaced {}".format(node.target))


class ModulePathTracer(torch.fx.Tracer):
    """
    ModulePathTracer is an FX tracer that--for each operation--also records
    the qualified name of the Module from which the operation originated.
    """

    # The current qualified name of the Module being traced. The top-level
    # module is signified by empty string. This is updated when entering
    # call_module and restored when exiting call_module
    current_module_qualified_name: str = ''
    # A map from FX Node to the qualname of the Module from which it
    # originated. This is recorded by `create_proxy` when recording an
    # operation
    node_to_originating_module: Dict[torch.fx.Node, str] = {}

    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any],
                    args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Override of Tracer.call_module (see
        https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.call_module).
        This override:
        1) Stores away the qualified name of the caller for restoration later
        2) Installs the qualified name of the caller in `current_module_qualified_name`
           for retrieval by `create_proxy`
        3) Delegates into the normal Tracer.call_module method
        4) Restores the caller's qualified name into current_module_qualified_name
        """
        old_qualname = self.current_module_qualified_name
        try:
            self.current_module_qualified_name = self.path_of_module(m)
            return super().call_module(m, forward, args, kwargs)
        finally:
            self.current_module_qualified_name = old_qualname

    def create_proxy(self, kind: str, target: torch.fx.node.Target, args: Tuple[Any, ...],
                     kwargs: Dict[str, Any], name: Optional[str] = None, type_expr: Optional[Any] = None):
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording
        of every operation and stores away the current traced module's qualified
        name in `node_to_originating_module`
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_originating_module[proxy.node] = self.current_module_qualified_name
        return proxy


class BinaryConv2d(nn.Conv2d):
    """docstring for QuanConv"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                           groups, bias)
        nn.init.constant_(self.weight, 1.0)

    # @weak_script_method
    def forward(self, x):
        # weight = self.weight
        w = self.weight.detach()
        binary_w = (w > 0.5).float()
        residual = w - binary_w
        weight = self.weight - residual
        weight = weight.to(x.device)
        output = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


class ScaledConv2D(nn.Module):
    def __init__(self, inplanes):
        super(ScaledConv2D, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.scale0 = BinaryConv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, groups=inplanes, bias=False)

    def forward(self, x):
        x = self.relu(x)
        # x = F.relu(self.scale0(x))
        x = self.scale0(x)

        return x


class ShapeProp:
    """
    Shape propagation. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. As each operation
    executes, the ShapeProp class stores away the shape and
    element type for the output values of each operation on
    the `shape` and `dtype` attributes of the operation's
    `Node`.
    """

    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args):
        args_iter = iter(args)
        env: Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target: str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                # import pdb; pdb.set_trace()
                print(node.name)
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.dtype = result.dtype

            env[node.name] = result

        return load_arg(self.graph)


def add_binary_model(model, bn_names, input_shape=None):
    if input_shape is None:
        input_shape = [1, 3, 224, 224]
    fx_model = fx.symbolic_trace(model)
    i = 0
    visited = set()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    ShapeProp(fx_model).propagate(torch.rand(*input_shape, dtype=dtype).to(device))
    relu_scaled_list = nn.ModuleList()
    node_list_add = []
    node_list_relu = []
    activate_names = ['relu', 'sigmoid']
    node_list_activate = []
    node_list_names = []
    # for node in fx_model.graph.nodes:
    #     # if ('relu' in node.name and 'relu' in node.target) or ('relu' in node.name and node.op == 'call_function'):
    #     if 'relu' in node.name and node.op == 'call_function':
    #         with fx_model.graph.inserting_after(node):
    #             new_node = fx_model.graph.call_method(node.name, node.args, node.kwargs, torch.nn.ReLU)
    #             node.replace_all_uses_with(new_node)
    #         fx_model.graph.erase_node(node)
    # fx_model.graph.lint()
    # fx_model.recompile()
    def is_activate(node, activate_names):
        for name in activate_names:
            if name in node.name:
                return True
        return False

    for node in fx_model.graph.nodes:
        if 'add' in node.name:
            node_list_add.append(node)
        # if ('relu' in node.name and 'relu' in node.target) or ('relu' in node.name and node.op == 'call_function'):
        # if ('relu' in node.name) and (
        #         node.op == 'call_function' or node.op == 'call_module') and 'pool' not in node.next.name:
        # import pdb; pdb.set_trace()
        if ('relu' in node.name) and 'pool' not in node.next.name:
            node_list_relu.append(node)
            # node_list_activate.append(node)
            # node_list_names.append(node.name)
    print("node_list_relu", node_list_relu)

    # if 'relu' in node.name and 'relu' in node.target and 'pool' not in node.next.name:
    #     node_list_relu.append(node)

    def find_group_conv(model, node_input):
        node_name = node_input.prev.name
        if 'conv' not in node_name:
            return False
        if len(node_name.split('_')) == 3:
            layer_name, index, conv_name = node_name.split('_')
            conv_op = getattr(getattr(model, layer_name)[int(index)], conv_name)
        else:
            conv_op = getattr(model, node_name)
        if conv_op.groups > 1:
            return True
        else:
            return False

    for node in node_list_add:
        # 判断要替换的node节点在不在node.add的历史记录里
        for node_input in node.all_input_nodes:
            if node_input in node_list_relu:
                node_list_relu.remove(node_input)
    # 判读激活函数前的节点是否是group_conv，如果是则剔除
    for node in node_list_relu:
        for node_input in node.all_input_nodes:
            if 'bn' not in node_input.name:
                continue
            if find_group_conv(model, node_input):
                node_list_relu.remove(node)
    for node in node_list_relu:
        # import pdb;pdb.set_trace()
        try:
            inplanes = node.shape[1]
        except Exception as e:
            inplanes = node.prev.shape[1]
        if 'bn' not in node.prev.name:
            continue
        else:
            #     bn_names.append(node.prev.prev.name)
            bn_names.append(node.prev.name)
        with fx_model.graph.inserting_after(node):

            relu_scaled_list.append(ScaledConv2D(inplanes))
            fx_model.add_submodule(f'relu_scaled_{i}', relu_scaled_list[i])
            new_node = fx_model.graph.call_module(f'relu_scaled_{i}', node.args, {})
            node.replace_all_uses_with(new_node)
            visited.add(f'relu_scaled_{i}')
            i += 1
        fx_model.graph.erase_node(node)
    fx_model.graph.lint()
    fx_model.recompile()
    return fx_model, bn_names
    # for node in fx_model.graph.nodes:
    #     if node.op == "call_module" and 'relu' in node.target and node.target not in visited:
    #         inplanes = node.shape[1]
    #         # node_list_relu.append(node)
    #         with fx_model.graph.inserting_after(node):
    #             relu_scaled_list.append(ScaledConv2D(inplanes))
    #             fx_model.add_submodule(f'relu_scaled_{i}', relu_scaled_list[i])
    #             new_node = fx_model.graph.call_module(f'relu_scaled_{i}', node.args, node.kwargs)
    #             node.replace_all_uses_with(new_node)
    #             visited.add(f'relu_scaled_{i}')
    #             i += 1
    #         fx_model.graph.erase_node(node)
    # fx_model.graph.lint()
    # fx_model.recompile()
    # return fx_model


if __name__ == "__main__":
    from torchvision.models import resnet50
    # from co_lib.co_lib.pruning.ptflops import get_flops_model

    model = resnet50(pretrained=False)
    # model.eval()
    # new_model = add_binary_model(model)
    # x = torch.randn([1, 3, 224, 224])  # 生成张量
    channel_list, mac_list, param_list = [], [], []
    flop_model = get_flops_model(model, (3, 224, 224))
    # macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    gm = fx.symbolic_trace(model)
    ShapeProp(gm).propagate(torch.rand([1, 3, 224, 224]))
    for name, module in flop_model.named_modules():
        if 'bn' in name:
            channel_list.append(module.weight.data.detach().shape[0])
        if 'conv' in name:
            mac_list.append(module.__flops__)
            # param_list.append(module.__params__)
            for node in gm.graph.nodes:
                if name == node.name.replace("_", '.'):
                    kernel_size = module.kernel_size
                    param_list.append(kernel_size[0] * kernel_size[0] * node.shape[2] * node.shape[3])
                    break
            # a.append(module(name, inputs))
    print(channel_list)
    print('\n')
    print(mac_list)
    print(param_list)
    # x = x.to("cuda")
    # export_onnx_file = "test.onnx"  # 目的ONNX文件名
    # torch.onnx.export(new_model, x, export_onnx_file, opset_version=10, do_constant_folding=True, input_names=["input"],
    #                   output_names=["output"], dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
    # gm = fx.symbolic_trace(new_model)
    # gm.graph.print_tabular()
    # print(new_model)
