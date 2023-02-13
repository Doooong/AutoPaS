import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from typing import Any, Callable, Dict, Optional, Tuple
from torch.fx.node import Node
from .rebuild import get_model_op


# from torchvision.models.feature_extraction import get_graph_node_names
# from torchvision.models.feature_extraction import create_feature_extractor


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
    def __init__(self, inplanes, node):
        super(ScaledConv2D, self).__init__()
        # if node.op == 'call_function':
        #     self.act = getattr(F, node)
        # elif node.op == "call_module":
        #     self.act = getattr(F, node.name.split("_")[-1])
        # else:
        #     self.act = getattr(F, node.target)
        self.act = getattr(F, node)
        self.scale0 = BinaryConv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, groups=inplanes, bias=False)

    def forward(self, x):
        # x = self.relu(x)
        x = self.act(x)
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
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.dtype = result.dtype

            env[node.name] = result

        return load_arg(self.graph)


def find_activate(nodes, activate_names):
    op_name = ''
    for node in nodes:
        confirm = 0
        if node.op == 'call_method':
            if node.target in activate_names:
                confirm += 1
                op_name = node.target
        else:
            if node.name.split('_')[0] in activate_names:
                confirm += 1
                op_name = node.name.split('_')[0]
    if op_name == 'hardtanh':
        op_name = 'relu6'
    return op_name


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

    node_list_activate = {}

    def is_suitable(node):
        external_names = ['conv', 'bn', 'pool', 'classifier', 'drop']
        for name in external_names:
            if name in node.name and 'act' not in node.name:
                return False
        return True

    def is_activate(model, node, node_list_activate):
        activate_names = ['relu', 'sigmoid', 'silu', 'hardtanh']

        if node.op == 'call_method':
            if node.target in activate_names:
                node_list_activate[node] = node.target
        elif node.op == "call_module":
            if not is_suitable(node):
                return False
            sub_module = get_model_op(model, node.name)
            try:
                nodes = fx.symbolic_trace(sub_module).graph.nodes
                op_name = find_activate(nodes, activate_names)
                if op_name != '':
                    node_list_activate[node] = op_name
            except Exception as e:
                pass
        else:
            if node.name.split('_')[0] in activate_names:
                node_list_activate[node] = node.name.split("_")[0]

    for node in fx_model.graph.nodes:
        if 'add' in node.name:
            node_list_add.append(node)
        if 'pool' not in node.next.name:
            is_activate(model, node, node_list_activate)

    def find_group_conv(model, node_input):
        node_name = find_node_name(node_input, 'conv')
        conv_op = get_model_op(model, node_name)
        if conv_op.groups > 1:
            return True
        else:
            return False

    def find_node_name(node, name):
        if name in node.prev.name:
            return node.prev.name
        return find_node_name(node.prev, name)

    for node in node_list_add:
        # 判断要替换的node节点在不在node.add的历史记录里
        for node_input in node.all_input_nodes:
            if node_input in node_list_activate:
                node_list_activate.pop(node_input)
        if node.next in node_list_activate:
            node_list_activate.pop(node.next)
    # 判读激活函数前的节点是否是group_conv，如果是则剔除(考虑conv-bn-act, conv-act两种)
    for node in list(node_list_activate.keys()):
        for node_input in node.all_input_nodes:
            if find_group_conv(model, node_input):
                del node_list_activate[node]
    if len(node_list_activate) == 0:
        print("Not found suitable node in model, which PaS not support!!")
    for node in node_list_activate:
        try:
            inplanes = node.shape[1]
        except Exception as e:
            inplanes = node.prev.shape[1]
        rm_name = find_node_name(node, 'conv')
        bn_names.append(rm_name)
        active_name = node.name.split('_')[0]
        with fx_model.graph.inserting_after(node):
            relu_scaled_list.append(ScaledConv2D(inplanes, node_list_activate[node]))
            fx_model.add_submodule(f'{active_name}_scaled_{i}', relu_scaled_list[i])
            new_node = fx_model.graph.call_module(f'{active_name}_scaled_{i}', node.args, {})
            node.replace_all_uses_with(new_node)
            visited.add(f'{active_name}_scaled_{i}')
            i += 1
        fx_model.graph.erase_node(node)
    fx_model.graph.lint()
    fx_model.recompile()
    return fx_model, bn_names


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
