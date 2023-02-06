import torch


def export_onnx(model, input_sample, save_path, is_cuda=True):
    if isinstance(input_sample, list):
        input_sample = torch.randn(input_sample)
    if is_cuda:
        model = model.to('cuda')
        input_sample = input_sample.to("cuda")
    torch.onnx.export(model, input_sample, save_path, opset_version=11, do_constant_folding=True, input_names=["input"],
                      output_names=["output"], dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
