import torch.nn as nn
from torch_geometric.data import Data

class GNBlock(nn.Module):
    """
        Implementation of a graph network layer. Current implementation is based on neural
        networks for each update function, and sum or mean and other reduce methods for each
        aggregation function.

        Method:
            phi_e(data): update function for edges and returns updated edges attributes e_p
            rho_e_v(e_p, data): aggregate updated edges e_p to per node attributes, b_e_p
            phi_v(b_e_p, data): update the nodes attributes by the results from previous step b_e_p and all the data
                                returns v_p.
            rho_e_u(e_p, data): aggregate edges to global attributes
            rho_v_u(v_p, data): aggregate nodes to global attributes
            build(data): organize the whole graph, and return a torch_geometric.data.Data object
    """
    def __init__(self):
        super(GNBlock, self).__init__()

    def phi_e(self, data):
        """
        Edge update function
        Args:
            data (torch_geometric.data type)
        Returns:
            output tensor
        """
        return NotImplementedError

    def rho_e_v(self, e_p, data):
        """
        Reduce edge attributes to node attribute
        Args:
            e_p: updated edge
            data (torch_geometric.data type)
        Returns: summed tensor
        """
        return NotImplementedError

    def phi_v(self, b_ei_p, data):
        """
        Node update function
        Args:
            b_ei_p (tensor): edge aggregated tensor
            data (torch_geometric.data type)
        Returns: updated node tensor
        """
        return NotImplementedError

    def rho_e_u(self, e_p, data):
        """
        aggregate edge to state
        Args:
            e_p (tensor): edge tensor
            data (tuple of tensors): other graph input tensors
        Returns: edge aggregated tensor for states
        """
        return NotImplementedError

    def rho_v_u(self, v_p, data):
        """
        aggregate node to state
        Args:
            v_p (tf.Tensor): updated node attributes
            data (Sequence): list or tuple for the graph inputs
        Returns:
            node to global aggregated tensor
        """
        return NotImplementedError

    def phi_u(self, b_e_p, b_v_p, data):
        """
        State update function
        Args:
            b_e_p (Tensor): edge to global aggregated tensor
            b_v_p (Tensor): node to global aggregated tensor
            data (Sequence): list or tuple for the graph inputs
        Returns:
            updated globa attributes
        """
        return NotImplementedError

    def build(self, data):
        e_p = self.phi_e(data)
        b_ei_p = self.rho_e_v(e_p, data)
        v_p = self.phi_v(b_ei_p, data)
        b_e_p = self.rho_e_u(e_p, data)
        b_v_p = self.rho_v_u(v_p, data)
        u_p = self.phi_u(b_e_p, b_v_p, data)
        data.x = v_p
        data.edge_attr = e_p
        data.global_state = u_p
        return data

    def forward(self, data):
        return self.build(data)