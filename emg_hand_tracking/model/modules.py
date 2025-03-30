from torch import nn, Tensor
import torch


class Unsqueeze(nn.Module):
    def __init__(self, i):
        self.i = i

    def forward(self, x):
        return x.unsqueeze(self.i)


class WindowedApply(nn.Module):
    def __init__(self, window_len: int, step: int, f: nn.Module):
        """
        Applies a function `f` to sliding windows extracted along the last dimension of an input tensor.

        Args:
            window_len (int): Length of each window along the last dimension.
            step (int): Step size (stride) between successive windows.
            f (nn.Module): A module (or callable) that processes a window. It should accept input of
                           shape (B, *A, window_len) and return output of shape (B, *F).
        """
        super().__init__()
        self.window_len = window_len
        self.step = step
        self.f = f

    def forward(self, x: Tensor) -> Tensor:
        """
        Input: (B, *A, T)
        Output: (B, W, *F), where W is the number of windows produced
        """
        assert x.dim() >= 2
        assert (
            x.shape[-1] >= self.window_len
        ), "The size of the time dimension must be at least window_len."

        # Use unfold on the last dimension to extract sliding windows
        # After unfold, x_unf has shape: (B, *A, W, window_len),
        x_unf = x.unfold(-1, self.window_len, self.step)

        # Let *A be any dimensions between the batch and time dimensions
        # We want to permute x_unf to have shape: (B, W, *A, window_len)
        x_unf_dim = x_unf.dim()
        A_len = x_unf_dim - 3
        assert A_len >= 0

        # Create a permutation: [0, W, A1, ..., Ak, window_len]
        # NOTE: becomes identity permutation if A_len is 0
        permute_order = [0, A_len + 1] + list(range(1, A_len + 1)) + [x_unf_dim - 1]
        x_perm = x_unf.permute(permute_order)

        # Now, x_perm has shape: (B, W, *A, window_len)
        # We merge the first two dimensions (B and W) to process all windows in one batch
        B, W = x_perm.shape[0], x_perm.shape[1]
        remaining_shape = x_perm.shape[2:]  # (*A, window_len)
        x_windows = x_perm.reshape(B * W, *remaining_shape)

        # Apply the function/module f on each window
        # f should accept input of shape (B, *A, window_len) where B here is B*W,
        # and return output of shape (B*W, *F)
        out = self.f(x_windows)

        # Reshape the output back to (B, W, *F)
        return out.reshape(B, W, *out.shape[1:])


class WeightedMean(nn.Module):
    def __init__(self, len: int):
        super().__init__()
        self.l = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.k = nn.Parameter(torch.rand(len - 1))

    @property
    def normalized_weights(self):
        weights = torch.cat([self.k, self.l.unsqueeze(0)])
        return weights / weights.sum()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, len, *A), where `len` matches the length of the filter.

        Returns:
            Tensor: Weighted mean of shape (B, *A).
        """
        assert (
            x.shape[1] == self.k.shape[0] + 1
        ), "Input length must match filter length."
        return torch.sum(
            x * self.normalized_weights.view(1, -1, *([1] * (x.dim() - 2))),
            dim=1,
        )
