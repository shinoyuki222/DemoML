import torch
from TorchCRF import CRF1
from utils import create_dir
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 2
sequence_size = 3
num_labels = 5
mask = torch.ByteTensor([[1, 1, 1], [1, 1, 0]]).to(device) # (batch_size. sequence_size)
labels = torch.LongTensor([[0, 2, 3], [1, 4, 1]]).to(device)  # (batch_size, sequence_size)
hidden = torch.randn((batch_size, sequence_size, num_labels), requires_grad=True).to(device)
crf = CRF(num_labels)
a = crf.forward(hidden, labels, mask)
b = crf.viterbi_decode(hidden, mask)
print(a)
print(b)

import argparse
parser = argparse.ArgumentParser(description='CRF')
parser.add_argument('--save-dir', type=str, default='./data_char/',
                    help='path to save processed data')
parser.add_argument('--onnx-dir', type=str, default='./model_onnx/',
                    help='path to save processed data')
args = parser.parse_args()

create_dir(args.onnx_dir)

import onnx
model = crf
x = (hidden, labels, mask)
# get results from model
model_out = model(hidden, labels, mask)
print(a)
# logits_tgt, logits_clsf = model(enc,enc_self_attn_mask)
torch.onnx.export(model,               # model being run
              x,                         # model input (or a tuple for multiple inputs)
              args.onnx_dir+"CRF.onnx",   # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'],
              )

import onnx

onnx_model = onnx.load(args.onnx_dir+"CRF.onnx")
onnx.checker.check_model(onnx_model)


import onnxruntime

ort_session = onnxruntime.InferenceSession(args.onnx_dir+"CRF.onnx")


def to_numpy(tensor):
    # return tensor.cpu().numpy()
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()



# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[i].name: to_numpy(x[i]) for i in range(len(x))}
ort_outs= ort_session.run(None, ort_inputs)




# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(model_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")