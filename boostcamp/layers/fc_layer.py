import torch.nn as nn


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, activation) -> None:
        super().__init__()
        layers = [
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
            ]
        if activation == "leakyrelu":
            layers += [nn.LeakyReLU()]
        elif activation == "sigmoid":
            layers += [nn.Sigmoid()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
