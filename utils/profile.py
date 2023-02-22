import numpy as np
from thop import clever_format


def profile(model):
    sp_layer = []
    num_dense_layer = 0
    total_num_g = 0
    total_num_nz_g = 0

    for (name, W) in model.named_parameters():
        non_zeros = W.detach().cpu().numpy().astype(np.float32) != 0
        num_nonzeros = np.count_nonzero(non_zeros)
        total_num = non_zeros.size
        total_num_g += total_num
        total_num_nz_g += num_nonzeros

        sparsity = 1 - (num_nonzeros * 1.0) / total_num
        if sparsity > 0:
            sp_layer.append(sparsity)
        else:
            num_dense_layer += 1

    gp = 1 - (total_num_nz_g * 1.0) / total_num_g
    if len(sp_layer) > 0:
        alp = sum(sp_layer) / len(sp_layer)
    else:
        alp = 0
    total_num_g, total_num_nz_g = clever_format([total_num_g, total_num_nz_g], "%.3f")
    # print(total_num_nz_g, total_num_g)
    print(f'model params is : {total_num_g}')