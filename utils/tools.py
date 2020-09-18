import torch
from torch_geometric.data import Batch

def gedge_index(batch):
    """
    A tensor maps each edge to its respective graph identifier.
    :param:
        batch (torch_geometric.data.Data or torch_geometric.data.Batch)
    :return:
        res (Tensor)
    """
    res = torch.tensor([], dtype=torch.long)
    if not isinstance(batch, Batch):
        res = torch.zeros((batch.num_edges, ), dtype=torch.long)
    elif isinstance(batch, Batch):
        data_list = batch.to_data_list()
        for i, data in enumerate(data_list):
            index = torch.full((data.num_edges, ), fill_value=i, dtype=torch.long)
            res = torch.cat((res, index))
    else:
        raise TypeError("The type of 'batch' must be torch_geometric.data.Data or torch_geometric.data.Batch")
    return res

def gnode_index(batch):
    """
    A tensor maps each node to its respective graph identifier.
    :param:
        batch (torch_geometric.data.Data or torch_geometric.data.Batch)
    :return:
        res (Tensor)
    """
    if not isinstance(batch, Batch):
        return torch.zeros((batch.num_nodes, ), dtype=torch.long)
    elif isinstance(batch, Batch):
        return batch.batch