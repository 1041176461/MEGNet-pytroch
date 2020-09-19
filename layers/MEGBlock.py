import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch
from torch_scatter import scatter
from MEGNet.layers.GNBlock import GNBlock
from MEGNet.utils.tools import gedge_index, gnode_index

class MEGBlock(GNBlock):
    """
        The MEGBlock implementation based on GNBlock as described in the paper

        Chen, Chi; Ye, Weike Ye; Zuo, Yunxing; Zheng, Chen; Ong, Shyue Ping.
        Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals,
        2018, arXiv preprint. [arXiv:1812.05055](https://arxiv.org/abs/1812.05055)

        Method:
            phi_e(data): update function for edges and returns updated edges attributes e_p
            rho_e_v(e_p, data): aggregate updated edges e_p to per node attributes, b_e_p
            phi_v(b_e_p, data): update the nodes attributes by the results from previous step b_e_p and all the data
                                returns v_p.
            rho_e_u(e_p, data): aggregate edges to global attributes
            rho_v_u(v_p, data): aggregate nodes to global attributes
        """
    def __init__(self, units_phi, units_before=[[64, 64, 64], [32, 32, 32]],
                 no_global=True, pool_method='mean', activation=nn.ReLU, use_bias=True, **kwargs):
        """
            Args:
                units_phi (list): the hidden layer in_features for update fully-connected neural network.
                            Its shape is (3, M), and 3 elements correspond to edge, node and global information respectively.
                            M represents the number of layers of each update function.
                units_before (list of integers): the hidden layer infeatures for fully-connected neural network before Graph network.
                            Its shape is (N, 3), in which 'N' represents the number of layers. Each layer has three elements corresponding
                            to edge, node and global_state respectively.
                no_global (bool): if True, defalut global information will be set zeros. If False, you need to set parameter 'global_state' for
                            global information.
                pool_method (str): 'mean', 'sum', 'max' or 'min', determines how information is gathered to nodes from neighboring edges
                activation (subclass of torch.nn): The activation function used for each sub-neural network, e.g. nn.ReLU, nn.Tanh
                use_bias (bool): default: True. Whether to use the bias term in the neural network.

            **kwargs:
                global_state (Tensor): initial global information
        """
        super(MEGBlock, self).__init__()
        self.units_e = units_phi[0]
        self.units_v = units_phi[1]
        self.units_u = units_phi[2]
        self.units_before = np.array(units_before)
        self.no_global = no_global
        self.pool_method = pool_method
        self.activation = activation
        self.use_bias = use_bias
        self.others = {}
        for key, value in kwargs.items():
            self.others[key] = value

    def _state_init(self, data):
        if 'global_state' not in data.keys:
            if self.no_global:
                if isinstance(data, Batch):
                    global_state = torch.zeros((data.num_graphs, self.units_u[0]), dtype=torch.float)
                elif not isinstance(data, Batch):
                    global_state = torch.zeros((1, self.units_u[0]))
                else:
                    raise TypeError(
                        "The type of 'batch' must be torch_geometric.data.Data or torch_geometric.data.Batch")
            else:
                assert 'global_state' in self.others.keys(), \
                    "The parameter 'no_global' is set to be False, you must set parameters 'global_state' for updating global information."
                global_state = self.others['global_state']
            data.global_state = global_state
        return data

    def _mlp(self, name, units, in_features, out_features):
        units.append(out_features)
        n = len(units)
        self.layers = nn.Sequential()
        for i in range(n):
            if i == 0:
                self.layers.add_module(f'{name}-{i}-linear', nn.Linear(in_features=in_features, out_features=units[i], bias=self.use_bias))
            else:
                self.layers.add_module(f'{name}-{i}-activation', self.activation())
                self.layers.add_module(f'{name}-{i}-linear', nn.Linear(in_features=units[i-1], out_features=units[i], bias=self.use_bias))

    def _first_block(self, data):
        units_e_1, units_v_1, units_u_1 = np.split(self.units_before, 3, axis=1)
        units_e_1, units_v_1, units_u_1 = units_e_1.flatten().tolist(), units_v_1.flatten().tolist(), units_u_1.flatten().tolist()
        e_size, v_size, u_size = data.num_edge_features, data.num_node_features, data.global_state.shape[1]
        self._mlp('first-e_p', units_e_1, e_size, e_size)
        data.edge_attr = self.layers(data.edge_attr)
        self._mlp('first-v_p', units_v_1, v_size, v_size)
        data.x = self.layers(data.x)
        u_layer = self._mlp('first-u_p', units_u_1, u_size, u_size)
        data.global_state = self.layers(data.global_state)
        return data

    def phi_e(self, data):
        """
        Edge update function
        Args:
            data (torch_geometric.data type)
        Returns:
            output tensor
        """
        gedge = gedge_index(data)
        u_expand = data.global_state[gedge]
        fr, fs = data.x[data.edge_index]
        inputs = torch.cat([fr, fs, data.edge_attr], dim=1)
        in_features = inputs.shape[1]
        out_features = data.num_edge_features
        self._mlp('phi_e', self.units_e, in_features, out_features)
        return self.layers(inputs)

    def rho_e_v(self, e_p, data):
        """
        Reduce edge attributes to node attribute
        Args:
            e_p: updated edge
            data (torch_geometric.data type)
        Returns: summed tensor
        """
        return scatter(e_p, data.edge_index[0], dim=0, reduce=self.pool_method)

    def phi_v(self, b_ei_p, data):
        """
        Node update function
        Args:
            b_ei_p (tensor): edge aggregated tensor
            data (torch_geometric.data type)
        Returns: updated node tensor
        """
        gnode = gnode_index(data)
        u_expand = data.global_state[gnode]
        inputs = torch.cat([b_ei_p, data.x, u_expand], dim=1)
        in_features = inputs.shape[1]
        out_features = data.num_node_features
        self._mlp('phi_v', self.units_v, in_features, out_features)
        return self.layers(inputs)

    def rho_e_u(self, e_p, data):
        """
        aggregate edge to state
        Args:
            e_p (tensor): edge tensor
            data (tuple of tensors): other graph input tensors
        Returns: edge aggregated tensor for states
        """
        gedge = gedge_index(data)
        return scatter(e_p, gedge, dim=0, reduce=self.pool_method)

    def rho_v_u(self, v_p, data):
        """
        aggregate node to state
        Args:
            v_p (tf.Tensor): updated node attributes
            data (Sequence): list or tuple for the graph inputs
        Returns:
            node to global aggregated tensor
        """
        gnode = gnode_index(data)
        return scatter(v_p, gnode, dim=0, reduce=self.pool_method)

    def phi_u(self, b_e_p, b_v_p, data):
        """
        State update function
        Args:
            b_e_p (Tensor): edge to global aggregated tensor
            b_v_p (Tensor): node to global aggregated tensor
            data (Sequence): list or tuple for the graph inputs
        Returns:
            updated global attributes
        """
        inputs = torch.cat([b_e_p, b_v_p, data.global_state], dim=1)
        in_features = inputs.shape[1]
        out_features = data.global_state.shape[1]
        self._mlp('phi_u', self.units_u, in_features, out_features)
        return self.layers(inputs)

    def forward(self, data):
        data = self._state_init(data)
        i_data = data.clone()

        data = self._first_block(data)
        r_data = data.clone()

        data = self.build(data)
        data.x = data.x + r_data.x + i_data.x
        data.edge_attr = data.edge_attr + r_data.edge_attr + i_data.edge_attr
        data.global_state = data.global_state + r_data.global_state + i_data.global_state
        return data