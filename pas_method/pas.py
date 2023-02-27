import random
import torch.nn as nn
import torch
from .utils_pas import add_binary_model, ShapeProp
from .ptflops import get_flops_model
import torch.fx as fx


class PaS:

    def __init__(self, model, input_shape, prune_ratio):
        super().__init__()
        self.channel_list = []
        self.flop_list = []
        self.mac_list = []
        self.bn_names = []
        self.model = model
        self.prune_ratio = prune_ratio
        self.input_shape = input_shape
        self.data_parallel = False

    def get_model_channel_flop(self):
        if len(self.channel_list) != 0:
            self.channel_list = []
            self.flop_list = []
            self.mac_list = []
        flop_model = get_flops_model(self.model, self.input_shape[1:])
        gm = fx.symbolic_trace(self.model)
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        ShapeProp(gm.to(device)).propagate(torch.rand(*self.input_shape, dtype=dtype).to(device))
        for name, module in flop_model.named_modules():
            if 'bn' in name:
                self.channel_list.append(module.weight.data.detach().shape[0])
            if 'conv' in name:
                self.flop_list.append(module.__flops__)
                # param_list.append(module.__params__)
                for node in gm.graph.nodes:
                    if name.replace("_", '.') == node.name.replace("_", '.'):
                        kernel_size = module.kernel_size
                        self.mac_list.append(kernel_size[0] * kernel_size[0] * node.shape[2] * node.shape[3])
                        break

    def init(self):
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
            self.data_parallel = True
        # self.get_model_channel_flop()
        self.model, self.bn_names = add_binary_model(self.model, self.bn_names, self.input_shape)
        # self.model = add_binary_model(self.model, self.bn_names, self.input_shape)
        self.init_dbc_weights()
        return self.model, self.bn_names

    def init_dbc_weights(self):
        for name, module in self.model.named_modules():
            if '_scaled' in name and ".scale" in name:
                channel_num = module.weight.shape[0]
                zero_num = int(channel_num * self.prune_ratio)
                device = module.weight.device
                init_weight = torch.ones(channel_num).to(device)
                zero_idx = random.sample(list(range(channel_num)), zero_num)
                init_weight[zero_idx] = 0
                module.weight.data = module.weight.data.to(device)
                if isinstance(module, nn.Conv2d):
                    module.weight.data *= init_weight.reshape(init_weight.size(0), 1, 1, 1)
                elif isinstance(module, nn.Conv3d):
                    module.weight.data *= init_weight.reshape(init_weight.size(0), 1, 1, 1, 1)
                # count_layer += 1
