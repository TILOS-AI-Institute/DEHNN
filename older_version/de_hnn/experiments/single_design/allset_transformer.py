"""Allset transformer class."""

import torch

from allset_transformer_layer import AllSetTransformerLayer


class AllSetTransformer(torch.nn.Module):
    """AllSet Neural Network Module.

    A module that combines multiple AllSet layers [1]_ to form a neural network.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    n_layers : int, default: 2
        Number of AllSet layers in the network.
    heads : int, default: 4
        Number of attention heads.
    dropout : float, default: 0.2
        Dropout probability.
    mlp_num_layers : int, default: 2
        Number of layers in the MLP.
    mlp_dropout: float, default: 0.2
        Dropout probability in the MLP.

    References
    ----------
    .. [1] Chien, Pan, Peng and Milenkovic.
        You are AllSet: a multiset function framework for hypergraph neural networks.
        ICLR 2022.
        https://arxiv.org/abs/2106.13264
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        n_layers=1,
        heads=4,
        dropout=0.2,
        mlp_num_layers=2,
        mlp_dropout=0.2,
    ):
        super().__init__()
        layers = [
            AllSetTransformerLayer(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                dropout=dropout,
                heads=heads,
                mlp_num_layers=mlp_num_layers,
                mlp_dropout=mlp_dropout,
            )
        ]

        for _ in range(n_layers - 1):
            layers.append(
                AllSetTransformerLayer(
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    dropout=dropout,
                    heads=heads,
                    mlp_num_layers=mlp_num_layers,
                    mlp_dropout=mlp_dropout,
                )
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x_0, incidence_1):
        """
        Forward computation.

        Parameters
        ----------
        x_0 : torch.Tensor
            Input features.
        incidence_1 : torch.Tensor
            Edge list (of size (2, |E|)).

        Returns
        -------
        torch.Tensor
            Output prediction.
        """
        for layer in self.layers:
            x_0, x_1 = layer(x_0, incidence_1)

        return x_0, x_1
