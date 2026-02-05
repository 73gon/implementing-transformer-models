import torch.optim


class TransformerLRScheduler:
    """
    Learning rate scheduler from "Attention is all you need" paper.

    The schedule increases the learning rate linearly for the first warmup_steps steps,
    then decreases it proportionally to the inverse square root of the step number.

    Formula: lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))

    This scheduler has a built-in warmup phase which is crucial for training transformers
    to avoid gradient instability at the beginning of training.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int = 4000):
        """
        Args:
            optimizer: PyTorch optimizer instance
            d_model: Dimensionality of the model (embedding dimension)
            warmup_steps: Number of steps to linearly increase the learning rate
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        """Update the learning rate based on the current step."""
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """
        Calculate the learning rate for the current step.

        Returns:
            The learning rate for the current step
        """
        step = self.current_step

        # Avoid division by zero
        if step == 0:
            step = 1

        # Formula from "Attention is all you need"
        # arg1: step^(-0.5)
        # arg2: step * warmup_steps^(-1.5)
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))

        return (self.d_model ** (-0.5)) * min(arg1, arg2)
