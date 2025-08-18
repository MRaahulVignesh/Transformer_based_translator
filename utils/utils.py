import torch


def causal_mask(size):
    """
    Create a causal mask for decoder self-attention.
    
    Args:
        size (int): Size of the sequence
        
    Returns:
        torch.Tensor: Causal mask of shape (1, size, size)
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0