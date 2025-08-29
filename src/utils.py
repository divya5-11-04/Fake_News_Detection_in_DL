import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def create_data_loader(input_ids, attention_masks, labels, batch_size=32, sampler_type="random"):
    """
    Wraps tensors into a DataLoader.
    Args:
        input_ids, attention_masks, labels : torch tensors
        batch_size : int
        sampler_type : "random" for training, "sequential" for validation/test
    """
    dataset = TensorDataset(input_ids, attention_masks, labels)

    if sampler_type == "random":
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)