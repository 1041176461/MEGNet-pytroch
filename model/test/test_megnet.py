import pytest
import torch
import torch.nn as nn
from torch_geometric.datasets import QM9
from torch_geometric.data import Data, DataLoader
from MEGNet.layers.MEGBlock import MEGBlock
from MEGNet.model.megnet import MEGNet

class Test_MEGNet:
    @classmethod
    def setup_class(cls):
        cls.units_phi = [[10, 12, 13], [10, 14, 16], [10, 15, 17]]
        cls.batch_size = 10
        cls.dataset = QM9(root="D:\\MachineLearing\\Project\\GN\\venv\\MEGNet\\data\\QM9")
        cls.loader = DataLoader(cls.dataset, batch_size=cls.batch_size, shuffle=False)

    @pytest.mark.parametrize('no_global', [True, False])
    def test_block(self, no_global):
        batch = next(iter(self.loader))
        u = torch.rand((batch.num_graphs, 6))
        MEG = MEGBlock(units_phi=self.units_phi, no_global=no_global,
                       activation=nn.ReLU, global_state=u)
        new_batch = MEG(batch)
        assert new_batch.num_edge_features == batch.num_edge_features
        assert new_batch.num_edges == batch.num_edges
        assert new_batch.num_node_features == batch.num_node_features
        assert new_batch.num_nodes == batch.num_nodes
        assert len(list(MEG.parameters())) != 0

    @pytest.mark.parametrize('no_global', [True, False])
    def test_net(self, no_global):
        batch = next(iter(self.loader))
        u = torch.rand((batch.num_graphs, 6))
        network = MEGNet(no_global=no_global, activation=nn.ReLU, global_state=u)
        res = network(batch)
        assert  res.shape == batch.y.shape
        assert len(list(network.parameters())) != 0

if __name__ == '__main__':
    pytest.main(['-s', '--html=report.html', 'test_megnet.py'])
