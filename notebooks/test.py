import torch

from nn.san_block import SAN
from nn.res_block import ResNet
from nn.model import CNN_LSTM, Model

input_tensor = torch.rand(10, 1, 80, 80).to('cuda:0')
san = SAN(sa_type=0, layers=(3, 4, 6, 8), kernels=[3, 7, 7, 7]).to('cuda:0')
san_out = san(input_tensor)
print(san_out.shape, torch.flatten(san_out, 1).shape)

res = ResNet().to('cuda:0')
res_out = res(input_tensor)
print(res_out.shape, torch.flatten(res_out, 1).shape)

input_trace = torch.rand(10, 1, 500).to('cuda:0')
cnn_lstm = CNN_LSTM(500, hidden_dim=100, output_dim=500).to('cuda:0')
lstm_out = cnn_lstm(input_trace)
print(lstm_out.shape)

model = Model().to('cuda:0')
out = model((input_trace, input_tensor))
print(out.shape)
