# coding : utf-8
# Author : yuxiang Zeng

# coding: utf-8
# Author: Yuxiang Zeng

import torch
from modules.pred_layer import Predictor
import pickle
import tracemalloc

# Enable tracing of memory allocations
tracemalloc.start()

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, args):
        super(LSTM, self).__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 定义LSTM
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.predictor = Predictor(self.hidden_size, self.hidden_size, 1)

    def forward(self, x):
        x = x.to(torch.float32)  # 确保输入是float32类型
        print(f"Input shape: {x.shape}, dtype: {x.dtype}")

        # 检查输入是否包含NaN或无穷大值
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Input contains NaN or Inf values!")
            return None

        try:
            output, (hn, cn) = self.lstm(x)
        except RuntimeError as e:
            print(f"Runtime error during LSTM forward pass: {e}")
            return None

        print(f"LSTM output shape: {output.shape}")
        print(f"Hidden state shape: {hn.shape}")
        print(f"Cell state shape: {cn.shape}")

        # 取出每个批次的最后一个时间步的输出
        last_output = output[:, -1, :]
        print(f"Last time step output shape: {last_output.shape}")

        try:
            y = self.predictor(last_output)
        except RuntimeError as e:
            print(f"Runtime error during prediction: {e}")
            return None

        print(f"Prediction shape: {y.shape}\n\n\n")
        return y

if __name__ == "__main__":
    # Sample code to test the LSTM model
    input_size = 10
    hidden_size = 20
    num_layers = 2
    args = {}

    model = LSTM(input_size, hidden_size, num_layers, args)
    sample_input = torch.randn(5, 15, input_size)  # (batch_size, seq_length, input_size)

    try:
        output = model(sample_input)
    except Exception as e:
        print(f"Error during model forward pass: {e}")

    # Print memory usage statistics
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6} MB; Peak was {peak / 10**6} MB")
    tracemalloc.stop()