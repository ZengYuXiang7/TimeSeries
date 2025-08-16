import torch


class SeqLayer(torch.nn.Module):
    def __init__(self, d_model, seq_method, bidirectional=True):
        super(SeqLayer, self).__init__()
        self.bidirectional = bidirectional
        self.seq_method = seq_method
        self.d_model = d_model

        if seq_method == 'rnn':  # 新增RNN
            self.rnn = torch.nn.RNN(self.d_model, self.d_model, num_layers=2, bias=True, batch_first=True, dropout=0.10, bidirectional=bidirectional)
        elif seq_method == 'lstm':
            self.lstm = torch.nn.LSTM(self.d_model, self.d_model, num_layers=2, bias=True, batch_first=True, dropout=0.10, bidirectional=bidirectional)
        elif seq_method == 'gru':
            self.gru = torch.nn.GRU(self.d_model, self.d_model, num_layers=2, bias=True, batch_first=True, dropout=0.10, bidirectional=bidirectional)
        elif seq_method == 'cnn':
            self.cnn = torch.nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=3, padding=1)
            self.cnn_activation = torch.nn.GELU()

        if bidirectional:
            self.aggregator = torch.nn.Linear(self.d_model * 2, self.d_model)

        self.MLP1 = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, self.d_model),
            torch.nn.GELU(),
            torch.nn.Linear(self.d_model, self.d_model)
        )
        self.dropout = torch.nn.Dropout(0.1)
        self.seq_output_norm1 = torch.nn.LayerNorm(self.d_model)
        self.seq_output_norm2 = torch.nn.LayerNorm(self.d_model)

    def forward(self, x):
        # 选择性处理不同类型的序列
        if self.seq_method == 'rnn':
            out, _ = self.rnn(x)
        elif self.seq_method == 'gru':
            if self.bidirectional:
                out, _ = self.gru(x)
            else:
                out, (_, _) = self.gru(x)
        elif self.seq_method == 'lstm':
            out, (_, _) = self.lstm(x)

        elif self.seq_method == 'cnn':
            out = x.permute(0, 2, 1)        # 转换为 (batch_size, d_model, L)
            out = self.cnn(out)             # 卷积操作
            out = self.cnn_activation(out)  # 激活函数
            out = out.permute(0, 2, 1)      # 转回 (batch_size, L, d_model)

        if self.bidirectional:
            out = self.aggregator(out)

        # resnet + Dropout
        out = x + self.dropout(out)
        out = self.seq_output_norm1(out)
        out = out + self.dropout(self.MLP1(out))
        out = self.seq_output_norm2(out)
        return out


class SeqEncoder(torch.nn.Module):
    def __init__(self, input_size, d_model, seq_len, num_layers, seq_method, bidirectional=True):
        super(SeqEncoder, self).__init__()
        self.num_layers = num_layers
        self.seq_transfer = torch.nn.Linear(input_size, d_model)
        self.seq_encoder = torch.nn.ModuleList(
            [SeqLayer(d_model, seq_method, bidirectional) for _ in range(num_layers)]
        )
        self.aggregator = torch.nn.Linear(seq_len * d_model, d_model)

    def forward(self, x):
        out = self.seq_transfer(x)
        for i in range(self.num_layers):
            out = out + self.seq_encoder[i](out)
        # out = out.reshape(out.shape[0], -1)
        # out = self.aggregator(out)
        return out


if __name__ == '__main__':
    batch_size = 4
    seq_len = 10
    d_model = 16
    input_size = 8
    num_layers = 2

    x = torch.randn(batch_size, seq_len, input_size)
    print(x.shape)
    seq_encoder = SeqEncoder(input_size, d_model, seq_len, num_layers, 'rnn', True)
    output = seq_encoder(x)
    print(output.shape)

    seq_encoder = SeqEncoder(input_size, d_model, seq_len, num_layers, 'lstm', True)
    output = seq_encoder(x)
    print(output.shape)  # 输出形状

    seq_encoder = SeqEncoder(input_size, d_model, seq_len, num_layers, 'gru', True)
    output = seq_encoder(x)
    print(output.shape)  # 输出形状

    seq_encoder = SeqEncoder(input_size, d_model, seq_len, num_layers, 'rnn', False)
    output = seq_encoder(x)
    print(output.shape)

    seq_encoder = SeqEncoder(input_size, d_model, seq_len, num_layers, 'lstm', False)
    output = seq_encoder(x)
    print(output.shape)  # 输出形状

    seq_encoder = SeqEncoder(input_size, d_model, seq_len, num_layers, 'gru', False)
    output = seq_encoder(x)
    print(output.shape)  # 输出形状

    ########## CNN ##########

    seq_encoder = SeqEncoder(input_size, d_model, seq_len, num_layers, 'cnn', False)
    output = seq_encoder(x)
    print(output.shape)  # 输出形状



