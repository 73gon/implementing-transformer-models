import torch.nn as nn
from torch.optim import AdamW


def create_optimizer_with_weight_decay(
    model: nn.Module,
    learning_rate: float = 0.0,
    weight_decay: float = 0.01,
) -> AdamW:
    """
    Create an AdamW optimizer with proper weight decay handling.

    Bias and LayerNorm parameters are excluded from weight decay to prevent
    the normalization layers from being regularized, which can harm training.

    Args:
        model: PyTorch model to optimize
        learning_rate: Initial learning rate (will be set by scheduler)
        weight_decay: Weight decay coefficient for parameters (default: 0.01)

    Returns:
        AdamW optimizer with two parameter groups:
        - params_with_decay: All parameters except bias and LayerNorm
        - params_without_decay: Bias and LayerNorm parameters
    """
    # Collect parameters with weight decay and without weight decay
    params_with_decay = []
    params_without_decay = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            # Parameters to exclude from weight decay:
            # - bias parameters
            # - layer norm parameters (weight and bias)
            if 'bias' in name or 'layer_norm' in name or 'LayerNorm' in name:
                params_without_decay.append(param)
            else:
                params_with_decay.append(param)

    # Create optimizer with parameter groups
    optimizer = AdamW([
        {'params': params_with_decay, 'weight_decay': weight_decay},
        {'params': params_without_decay, 'weight_decay': 0.0},
    ], lr=learning_rate)

    return optimizer
