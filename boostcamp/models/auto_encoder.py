import pytorch_lightning as pl
import torch
import torch.nn as nn

from boostcamp.layers import FullyConnectedLayer


class AutoEncoder(pl.LightningModule):
    def __init__(self, encoder_sizes, decoder_sizes, activation) -> None:
        super().__init__()
        self.encoder = ...
        self.decoder = ...
        self.loss_fn = self._set_loss_fn()
        self._set_encoder(encoder_sizes, activation)
        self._set_decoder(decoder_sizes, activation)

    def _set_encoder(self, encoder_sizes, activation):
        encoder_layers = []
        for in_features, out_features in zip(encoder_sizes[:-1], encoder_sizes[1:]):
            encoder_layers += [
                FullyConnectedLayer(
                    in_features=in_features,
                    out_features=out_features,
                    activation=activation,
                )
            ]
        self.encoder = nn.Sequential(*encoder_layers)

    def _set_decoder(self, decoder_sizes, activation):
        decoder_layers = []
        for in_features, out_features in zip(decoder_sizes[:-1], decoder_sizes[1:]):
            decoder_layers += [
                FullyConnectedLayer(
                    in_features=in_features,
                    out_features=out_features,
                    activation=activation,
                )
            ]
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

    def _set_loss_fn(self):
        return nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        test_loss = self.loss_fn(y_hat, y)
        self.log("test_loss", test_loss)
