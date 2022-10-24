import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.linear = nn.Linear(base_model.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        hiddens = raw_outputs.last_hidden_state
        cls_feats = hiddens[:, 0, :]
        predicts = self.linear(self.dropout(cls_feats))
        return predicts


class Gru_Model(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.dropout = nn.Dropout(0.5)

        self.Rnn = nn.RNN(input_size=320,
                          hidden_size=64,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True
                          )
        self.Fn = nn.Linear(64, 2)

        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        text_ids = inputs['input_ids']

        # lstm_out, (h_n, h_c) = self.Lstm(datasets, None)
        # outputs = self.Fn(lstm_out[:, -1])
        outputs = 0
        return outputs


class Lstm_Model(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.Lstm = nn.LSTM(input_size=768,
                            hidden_size=320,
                            num_layers=1,
                            batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(320, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.ReLU())
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        hiddens = raw_outputs.last_hidden_state
        # Tokens ==> torch.Size([8,m,768]) and m <= 319
        tokens = hiddens[:, 1:, :]

        x, _ = self.Lstm(tokens)
        x = self.fc(x)
        outputs = x[:, -1, :]
        return outputs


class BiLstm_Model(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()

        self.Lstm = nn.LSTM(input_size=16,
                            hidden_size=64,
                            num_layers=1,
                            batch_first=True)
        self.Fn = nn.Linear(64 * 2, 2)

    def forward(self, datasets):
        lstm_out, _ = self.Lstm(datasets, None)
        outputs = self.Fn(lstm_out[:, -1, :])
        return outputs


class Rnn_Model(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()

        self.rnn = nn.RNN(input_size=16,
                          hidden_size=64,
                          num_layers=1,
                          batch_first=True)
        self.Fn = nn.Linear(64, 2)

    def forward(self, datasets):
        out = self.rnn(datasets, None)
        outputs = self.Fn(out)
        return outputs
