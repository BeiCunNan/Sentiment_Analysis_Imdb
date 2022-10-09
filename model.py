from torch import nn


class Transformer(nn.Modeule):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.linear = nn.Linear(base_model.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        for param in base_model.parameters():
            param.requires_grad = (True)

        def forward(self, *args, **kwargs):
            raw_outputs = self.base_model(*args, **kwargs)
            hiddens = raw_outputs.last_hidden_state
            cls_feats = hiddens[:, 0, :]
