import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.linear = nn.Linear(base_model.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax()
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        # The pooler_output is made of CLS --> FNN --> Tanh
        # The last_hidden_state[:,0] is made of original CLS
        # Method one
        # cls_feats  = raw_outputs.pooler_output
        # Method two
        cls_feats = raw_outputs.last_hidden_state[:, 0, :]
        predicts = self.softmax(self.linear(self.dropout(cls_feats)))
        return predicts


class Gru_Model(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.Gru = nn.GRU(input_size=768,
                          hidden_size=320,
                          num_layers=1,
                          batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(320, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state

        gru_output, _ = self.Gru(tokens)
        outputs = gru_output[:, -1, :]
        outputs = self.fc(outputs)
        return outputs


# Try to use the softmax、relu、tanh and logistic
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
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state
        lstm_output, _ = self.Lstm(tokens)
        outputs = lstm_output[:, -1, :]
        outputs = self.fc(outputs)
        return outputs


class BiLstm_Model(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()

        self.BiLstm = nn.LSTM(input_size=16,
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
        self.base_model = base_model
        self.num_classes = num_classes
        self.Rnn = nn.RNN(input_size=768,
                          hidden_size=320,
                          num_layers=1,
                          batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(320, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        cls_feats = raw_outputs.last_hidden_state
        outputs, _ = self.Rnn(cls_feats)
        outputs = outputs[:, -1, :]
        outputs = self.fc(outputs)
        return outputs
