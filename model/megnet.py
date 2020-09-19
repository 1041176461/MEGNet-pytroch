import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import Set2Set
from collections import OrderedDict
from ..layers.MEGBlock import MEGBlock
from ..utils.tools import gedge_index, gnode_index

class MEGNet(nn.Module):
    """
    The MEGNet implementation composed of MEGBlock, Set2Set and some linear layers as described in the paper

    Chen, Chi; Ye, Weike Ye; Zuo, Yunxing; Zheng, Chen; Ong, Shyue Ping.
    Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals,
    2018, arXiv preprint. [arXiv:1812.05055](https://arxiv.org/abs/1812.05055)
    """
    def __init__(self, no_global=True, pool_method='mean', activation=nn.ReLU,
                 processing_steps=10, num_lstm_layers=1, **kwargs):
        """
        Args:
            no_global (bool): if True, defalut global information will be set zeros.
                            If False, you need to set parameter 'global_state' for global information.
            pool_method (str): 'mean', 'sum', 'max' or 'min', determines how information is gathered to nodes from neighboring edges
            activation (subclass of torch.nn): activation function, e.g. nn.ReLU, nn.Tanh
            processing_steps (int): the paramter of Set2Set means number of iterations T. (default: 10)
            num_lstm_layers (int): the parameter of Set2Set means number of recurrent layers,
                        .e.g, setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM,
                        with the second LSTM taking in outputs of the first LSTM and computing the final results. (default: 1)

        **kwargs:
            global_state (Tensor): initial global information
        """
        super(MEGNet, self).__init__()
        self.no_global = no_global
        self.pool_method = pool_method
        self.activation = activation
        self.act_obj = self.activation()
        self.processing_steps = processing_steps
        self.num_lstm_layers = num_lstm_layers
        self.others = {}
        for key, value in kwargs.items():
            self.others[key] = value
        if "global_state" in self.others.keys():
            self.global_state = self.others["global_state"]
        else:
            self.global_state = []
        self.units_phi = [[10, 12, 13], [10, 14, 16], [10, 15, 17]]
        self.block = nn.Sequential(
                    MEGBlock(units_phi=self.units_phi, no_global=self.no_global, pool_method=self.pool_method,
                               activation=self.activation, global_state=self.global_state),
                    MEGBlock(units_phi=self.units_phi, no_global=self.no_global, pool_method=self.pool_method,
                               activation=self.activation, global_state=self.global_state)
        )
    def _set2set_init(self, in_channels):
        return Set2Set(in_channels, self.processing_steps, self.num_lstm_layers)

    def _out_init(self, in_channels, out_channels):
        return nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features=in_channels, out_features=32)),
            ('relu1', self.act_obj),
            ('linear2', nn.Linear(in_features=32, out_features=16)),
            ('relu2', self.act_obj),
            ('out', nn.Linear(in_features=16, out_features=out_channels))
        ]))

    def forward(self, data):
        data = self.block(data)

        self.edge_set2set = self._set2set_init(data.num_edge_features)
        gedge = gedge_index(data)
        edge_res = self.edge_set2set(data.edge_attr, gedge)
        self.node_set2set = self._set2set_init(data.num_node_features)
        gnode = gnode_index(data)
        node_res = self.node_set2set(data.x, gnode)

        cat_res = torch.cat((edge_res, node_res, data.global_state), dim=1)
        out_layer = self._out_init(cat_res.shape[1], data.y.shape[1])
        return out_layer(cat_res)










