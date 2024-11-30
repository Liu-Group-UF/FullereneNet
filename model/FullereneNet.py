import math
from typing import Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, PairTensor
from torch_geometric.utils import softmax


class FullereneNetConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        aggr: str = 'add',
        concat: bool = True,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        **kwargs
    ):
        super().__init__(node_dim=0, aggr=aggr, **kwargs)

        # Initialize class variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        # Define layers for query, key, value, and edge attribute transformations
        self.lin_query = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_key = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_value = nn.Linear(in_channels, heads * out_channels, bias=bias)
        # Edge transformation is conditional on the presence of edge features
        self.lin_edge = (
            nn.Linear(edge_dim, heads * out_channels, bias=bias)
            if edge_dim is not None
            else None
        )

        self.attn_additive = nn.Sequential(
            nn.Linear(3 * out_channels, out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, 1, bias=False),
        )

        self.attn_additive_2 = nn.Sequential(
            nn.Linear(3 * out_channels, out_channels),
            nn.Softplus(),
            nn.Linear(out_channels, 1, bias=False),
        )
        # Linear layer to combine multi-head outputs
        self.lin_concat = nn.Linear(heads * out_channels, out_channels, bias=bias)

        self.line_update = nn.Linear(out_channels * 3, out_channels * 3, bias=bias)
        self.lin_fea = nn.Linear(in_channels, out_channels, bias=bias)

        self.bn = nn.BatchNorm1d(out_channels)

        # Layer Normalizations
        self.attn_layer_norm = nn.LayerNorm(out_channels)
        self.layer_norm = nn.LayerNorm(out_channels * heads)

        # Activation function
        self.sigmoid = nn.Sigmoid()

        # **Edge gating mechanism**
        if edge_dim is not None:
            self.edge_gate = nn.Linear(edge_dim, heads)
        else:
            self.edge_gate = None

        # Reset parameters of the model
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_concat.reset_parameters()
        self.line_update.reset_parameters()
        self.lin_fea.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        for layer in self.attn_additive:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()      
        for layer in self.attn_additive_2:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()  
        self.attn_layer_norm.reset_parameters()
        self.layer_norm.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # Propagate messages using the defined message and aggregation scheme
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Process the output based on the concatenation flag
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
            out = self.lin_concat(out)
        else:
            out = out.mean(dim=1)
        out = F.silu(self.bn(out))
        out += self.lin_fea(x[0])
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Optional[Tensor], index:Tensor) -> Tensor:
        # Compute queries, keys, values for each head
        q_i = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        k_j = self.lin_key(x_j).view(-1, self.heads, self.out_channels)
        v_j = self.lin_value(x_j).view(-1, self.heads, self.out_channels)

        # Edge features transformation
        if edge_attr is not None and self.lin_edge is not None:
            e_ij = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            # Edge-conditioned keys and values
            k_j = k_j + e_ij
            v_j = v_j + e_ij
        else:
            e_ij = torch.zeros_like(k_j)  # For consistency

        # **Additive Attention Component**
        additive_input = torch.cat([q_i, k_j, e_ij], dim=-1)
        additive_scores = self.attn_additive(additive_input).squeeze(-1)
        additive_scores_2 = self.attn_additive_2(additive_input).squeeze(-1)
        additive_scores = additive_scores + additive_scores_2

        # **Multiplicative Attention Component**
        multiplicative_scores = (q_i * k_j).sum(dim=-1) / math.sqrt(self.out_channels)

        # **Combine Attention Components**
        alpha = additive_scores + multiplicative_scores

        # Apply nonlinear activation
        # alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = F.silu(alpha)

        # Softmax over source nodes (neighbors)
        alpha = softmax(alpha, index=index, num_nodes=x_i.size(0))

        # Apply dropout to attention coefficients
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # print("alpha", alpha.shape)


        # Compute attention-weighted values
        out = v_j * alpha.unsqueeze(-1)
        out = out.view(-1, self.heads * self.out_channels)
        out = self.layer_norm(out)
        return out

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}({self.in_channels}, '
            f'{self.out_channels}, heads={self.heads})'
        )


class FullereneNet(nn.Module):
    """
    FullereneNet model for predicting fullerenes properties.

    Args:
        atom_input_features (int): Number of features for each atom in the input.
        node_fea (int): Number of features for each node (atom) after embedding.
        edge_fea (int): Number of features for each edge in the input.
        conv_layers (int): Number of convolutional layers in the model.
        hidden_layer (int): Number of features in the hidden layer of the fully connected network.
        heads (int): Number of attention heads in each MatformerConv layer.
        dropout (float): Dropout probability. (default: 0)
        classification (bool): If True, the model will be configured for a classification task. (default: False)
    """

    def __init__(
        self,
        atom_input_features: int = 92,
        node_fea: int = 64,
        edge_fea: int = 41,
        conv_layers: int = 5,
        hidden_layer: int = 128,
        heads: int = 4,
        dropout: float = 0.0,
        classification: bool = False,
    ):
        super().__init__()
        self.classification = classification
        self.dropout = dropout

        # Embedding layers for atoms and edges
        self.atom_embedding = nn.Linear(atom_input_features, node_fea)
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_fea, node_fea),
            nn.Softplus(),
            nn.Linear(node_fea, node_fea),
        )

        # Attention layers
        self.att_layers = nn.ModuleList(
            [
                FullereneNetConv(
                    in_channels=node_fea,
                    out_channels=node_fea,
                    heads=heads,
                    edge_dim=node_fea,
                    dropout=dropout,
                )
                for _ in range(conv_layers)
            ]
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(node_fea, hidden_layer),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Output layer
        if self.classification:
            self.fc_out = nn.Linear(hidden_layer, 2)  # Binary classification
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(hidden_layer, 1)  # Regression

    def forward(self, data) -> Tensor:
        """
        Forward pass of the Matformer model.

        Args:
            data: Input data containing node features, edge features, edge indices, and batch information.

        Returns:
            Tensor: The output of the model, either class logits or regression values.
        """
        node_fea = data.x
        edge_fea = data.edge_attr
        edge_index = data.edge_index
        batch = data.batch

        # Embedding transformations
        node_fea = self.atom_embedding(node_fea)
        if edge_fea is not None:
            edge_fea = self.edge_embedding(edge_fea)

        # Apply attention layers
        for conv_layer in self.att_layers:
            node_fea = conv_layer(x=node_fea, edge_index=edge_index, edge_attr=edge_fea)

        # Pooling layer for graph-level prediction
        crystal_fea = global_mean_pool(node_fea, batch)
        # crystal_fea = self.set2set(node_fea, batch)

        # Fully connected layers
        crystal_fea = self.fc(crystal_fea)
        out = self.fc_out(crystal_fea)

        # Apply softmax for classification
        if self.classification:
            out = self.softmax(out)

        return out
