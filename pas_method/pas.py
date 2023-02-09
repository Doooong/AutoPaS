import random

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
        self.generate_bn_weights()
        return self.model, self.bn_names

    def generate_bn_weights(self):
        # BNs = torch.tensor([]).cuda()
        # weights = torch.tensor([]).cuda()
        # length = []
        # # names = []
        # n = 0
        # bn_names = [name.replace('_', '.') for name in self.bn_names]
        # channel_list = []
        # for name, module in self.model.named_modules():
        #     if 'bn' in name and name in bn_names:
        #         data = module.weight.data.detach().to(BNs.device)
        #         BNs = torch.cat((BNs, data), dim=0)
        #         weights = torch.cat((weights, torch.ones(data.size()).cuda() * n), dim=0)
        #         length.append(data.size(0))
        #         channel_list.append(data.size(0))
        #         # names.append('module.' + name + '.weight')
        #         n += 1
        #
        # srt, idx = torch.sort(torch.clone(BNs))
        # sort_weights = weights[idx]
        # lis = channel_list
        # # accumulation = 0.
        # j = 0
        # for i in range(sort_weights.size(0)):
        #     # accumulation += sort_weights[i]
        #     # meter.update(sort_weights[i])
        #     lis[int(sort_weights[i])] -= 1
        #     j += 1
        #     # print(sum(meter.macs_list)/1e9, sum(repvgg_b1_list)/1e9)
        #     if sum(lis) < sum(length) * (1 - self.prune_ratio):
        #         break
        # _, final_idx = torch.topk(BNs, int(j), largest=False)
        # BNs[final_idx] = 0
        #
        # BNs = (BNs > 0).float()
        #
        # sep = torch.split(BNs, length)
        # count_layer = 0
        for name, module in self.model.named_modules():
            if '_scaled' in name and ".scale" in name:
                channel_num = module.weight.shape[0]
                zero_num = int(channel_num * self.prune_ratio)
                device = module.weight.device
                init_weight = torch.ones(channel_num).to(device)
                zero_idx = random.sample(list(range(channel_num)), zero_num)
                init_weight[zero_idx] = 0
                module.weight.data = module.weight.data.to(device)
                module.weight.data *= init_weight.reshape(init_weight.size(0), 1, 1, 1)
                # count_layer += 1
