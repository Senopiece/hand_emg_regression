import torch.nn.functional as F
from torch import nn, Tensor
import torch


def make_spike_weights(p, num_slices, alpha=1.0, gamma=10.0):
    """
    Create strictly positive, differentiable, and sharper logits for each possible slice
    based on a continuous parameter.

    Args:
        p (Tensor): A scalar tensor in [0, 1] representing the normalized starting position.
        num_slices (int): The number of possible slices.
        alpha (float): Steepness factor.
        gamma (float): Exponent factor to sharpen the spike (gamma > 1).

    Returns:
        Tensor: Logits of shape (num_slices,) that are >0 and have a sharper spike at the index closest to p*(num_slices-1).
    """
    p_scaled = p * (
        num_slices - 1
    )  # Scale the normalized position to the index range [0, num_slices - 1]
    indices = torch.arange(num_slices, device=p.device, dtype=p.dtype)
    return alpha * (((num_slices - 1) - torch.abs(indices - p_scaled)) ** gamma)


class Unsqueeze(nn.Module):
    def __init__(self, i):
        super().__init__()
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


# class LearnablePatternSimilarity(nn.Module):
#     def __init__(self, input_len: int, eps=1e-8):
#         super().__init__()
#         self.l = nn.Parameter(torch.tensor(1.0), requires_grad=False)
#         self.k = nn.Parameter(torch.rand(input_len - 1))
#         self.eps = nn.Parameter(torch.tensor(eps), requires_grad=False)

#     @property
#     def pattern(self):
#         return torch.cat([self.k, self.l.unsqueeze(0)])

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: shape (B, N), where B = batch size, N = input length
#         returns: shape (B,), correlation
#         """
#         y = self.pattern  # shape: (N,)
#         assert x.shape[1] == y.shape[0], "Input length must match pattern length"

#         x_mean = x.mean(dim=1, keepdim=True)
#         y_mean = y.mean()

#         x_centered = x - x_mean  # shape: (B, N)
#         y_centered = y - y_mean  # shape: (N,)

#         numerator = (x_centered * y_centered).sum(dim=1)  # shape: (B,)
#         x_norm = x_centered.norm(dim=1)  # shape: (B,)
#         y_norm = y_centered.norm()  # scalar

#         denominator = x_norm * y_norm + self.eps  # shape: (B,)
#         corr = numerator / denominator  # shape: (B,)

#         return corr

# class LearnableSlice(nn.Module):
#     def __init__(self, window_len: int, sigma: float = 1.0):
#         """
#         Extracts a differentiable window slice from an input tensor along the second dimension.

#         Args:
#             window_len (int): Length of the window to extract along the time dimension.
#             sigma (float): Standard deviation for the Gaussian used in soft window extraction.
#                            Lower values yield a window closer to a hard slice.
#         """
#         super().__init__()
#         self.window_len = window_len
#         self.sigma = sigma
#         self.start_idx = nn.Parameter(torch.rand())

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x (Tensor): Input tensor of shape (B, input_len, *A).

#         Returns:
#             Tensor: Soft-selected slice of shape (B, window_len, *A).
#         """
#         # Create a tensor of indices for the time dimension.
#         t = torch.arange(x.shape[1], device=x.device).float()  # shape: (input_len,)

#         # For each window index j (0 <= j < window_len), the desired input index is start_idx + j.
#         j = torch.arange(
#             self.window_len, device=x.device
#         ).float()  # shape: (window_len,)
#         desired_positions = self.start_idx + j  # shape: (window_len,)

#         # Compute Gaussian weights for each window index over all input positions.
#         # W will have shape (window_len, input_len)
#         W = torch.exp(
#             -((t.unsqueeze(0) - desired_positions.unsqueeze(1)) ** 2)
#             / (2 * self.sigma**2)
#         )
#         W = W / (W.sum(dim=1, keepdim=True) + 1e-8)

#         # Compute weighted sum along the time dimension.
#         # x has shape (B, input_len, *A) and W.t() has shape (input_len, window_len).
#         # The tensordot contracts over the time dimension (dim 1 of x).
#         y = torch.tensordot(x, W.t(), dims=([1], [0]))  # type: ignore
#         # y has shape (B, *A, window_len)

#         # Permute y so that the window_len dimension is the second dimension:
#         # Desired output shape: (B, window_len, *A)
#         return y.permute(0, y.dim() - 1, *range(1, y.dim() - 1))

# class LearnableSlice(nn.Module):
#     def __init__(self, output_len: int, temperature: float = 1e-3):
#         """
#         Args:
#             output_len (int): The desired length of the output slice.
#             temperature (float): Temperature for the softmax. A small value makes the softmax near one-hot.
#         """
#         super().__init__()
#         self.output_len = output_len
#         self.temperature = temperature

#         self.s = nn.Parameter(torch.rand())

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x (Tensor): Input tensor of shape (B, input_len, *A).

#         Returns:
#             Tensor: Selected slice (B, output_len, *A), computed as a weighted sum over all possible slices.
#         """
#         num_slices = x.shape[1] - self.output_len + 1
#         logits = make_spike_weights(F.sigmoid(self.s), num_slices)

#         # Compute softmax weights with a small temperature to encourage near one-hot behavior.
#         weights = F.softmax(logits / self.temperature, dim=0)  # Shape: (num_slices,)

#         # Generate all possible slices from the input tensor.
#         # Each slice is taken along the second dimension.
#         slices = [x[:, i : i + self.output_len, ...] for i in range(num_slices)]
#         # Stack the slices to get a tensor of shape (num_slices, B, output_len, *A)
#         slices_stacked = torch.stack(slices, dim=0)

#         # Reshape weights for broadcasting:
#         # Create a shape like (num_slices, 1, 1, ..., 1) with as many 1's as needed.
#         weight_shape = [num_slices] + [1] * (slices_stacked.dim() - 1)
#         weights = weights.view(*weight_shape)

#         # Multiply each slice by its corresponding weight and sum over all slices.
#         # The resulting tensor has shape (B, output_len, *A)
#         return (weights * slices_stacked).sum(dim=0)


# class LearnableSlice2(nn.Module):
#     def __init__(self, output_len: int):
#         """
#         Args:
#             output_len (int): The desired length of the output slice.
#         """
#         super().__init__()
#         self.output_len = output_len
#         self.s = nn.Parameter(torch.rand())

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x (Tensor): Input tensor of shape (B, input_len, *A).

#         Returns:
#             Tensor: Interpolated slice of shape (B, output_len, *A).
#         """
#         B = x.shape[0]
#         # Map the scalar s to a valid starting index in [0, input_len - output_len]
#         start = torch.sigmoid(self.s) * (x.shape[1] - self.output_len)

#         # Create continuous positions for the output slice:
#         # positions = start, start + 1, ..., start + output_len - 1
#         pos = start + torch.arange(self.output_len, device=x.device, dtype=x.dtype)

#         # For each continuous position, compute the floor and ceil indices
#         pos_floor = torch.floor(pos).long()
#         pos_ceil = pos_floor + 1
#         # The interpolation weight is the fractional part
#         w = pos - pos_floor.float()

#         # Clamp pos_ceil to ensure it does not exceed valid index range
#         pos_ceil = pos_ceil.clamp(max=x.shape[1] - 1)

#         # Expand indices to match the batch dimension
#         pos_floor = pos_floor.unsqueeze(0).expand(B, -1)
#         pos_ceil = pos_ceil.unsqueeze(0).expand(B, -1)

#         # Create a batch index for advanced indexing
#         batch_idx = (
#             torch.arange(B, device=x.device).unsqueeze(1).expand(B, self.output_len)
#         )

#         # Gather values from x at floor and ceil indices along the slicing dimension (dim=1)
#         x_floor = x[batch_idx, pos_floor]  # shape: (B, output_len, *A)
#         x_ceil = x[batch_idx, pos_ceil]  # shape: (B, output_len, *A)

#         # Reshape the weights to broadcast correctly over additional dimensions
#         w = w.view(1, self.output_len, *([1] * (x.dim() - 2)))

#         # Perform linear interpolation
#         return (1 - w) * x_floor + w * x_ceil


class LearnableSlice3(nn.Module):
    def __init__(self, window_len: int, sigma: float = 1.0):
        """
        Extracts a differentiable window slice from an input tensor along the second dimension.

        Args:
            window_len (int): Length of the window to extract along the time dimension.
            sigma (float): Standard deviation for the Gaussian used in soft window extraction.
                           Lower values yield a window closer to a hard slice.
        """
        super().__init__()
        self.window_len = window_len
        self.sigma = sigma
        self.start_idx = nn.Parameter(torch.rand())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, input_len, *A).

        Returns:
            Tensor: Soft-selected slice of shape (B, window_len, *A).
        """
        # Create a tensor of indices for the time dimension.
        t = torch.arange(x.shape[1], device=x.device).float()  # shape: (input_len,)

        # For each window index j (0 <= j < window_len), the desired input index is start_idx + j.
        j = torch.arange(
            self.window_len, device=x.device
        ).float()  # shape: (window_len,)
        desired_positions = self.start_idx + j  # shape: (window_len,)

        # Compute Gaussian weights for each window index over all input positions.
        # W will have shape (window_len, input_len)
        W = torch.exp(
            -((t.unsqueeze(0) - desired_positions.unsqueeze(1)) ** 2)
            / (2 * self.sigma**2)
        )
        W = W / (W.sum(dim=1, keepdim=True) + 1e-8)

        # Compute weighted sum along the time dimension.
        # x has shape (B, input_len, *A) and W.t() has shape (input_len, window_len).
        # The tensordot contracts over the time dimension (dim 1 of x).
        y = torch.tensordot(x, W.t(), dims=([1], [0]))  # type: ignore
        # y has shape (B, *A, window_len)

        # Permute y so that the window_len dimension is the second dimension:
        # Desired output shape: (B, window_len, *A)
        return y.permute(0, y.dim() - 1, *range(1, y.dim() - 1))


class ExtractLearnableSlices(nn.Module):
    def __init__(
        self, n: int, width: int, sigma_time: float = 1.0, sigma_channel: float = 0.1
    ):
        """
        Extracts n learnable slices from an input tensor along the time dimension.
        Each slice is determined by two learnable parameters:
          - A channel parameter (in [0,1]) that softly selects channels.
          - A time offset (in [0,1]) that determines the starting index of the time window.

        Args:
            n (int): Number of slices to extract.
            width (int): Length of the time window to extract for each slice.
            sigma_time (float): Std. deviation for the Gaussian used in soft time slicing.
            sigma_channel (float): Std. deviation for the Gaussian used in soft channel selection.
        """
        super().__init__()
        self.n = n
        self.width = width
        self.sigma_time = sigma_time
        self.sigma_channel = sigma_channel
        # Learnable parameters, one per slice.
        self.channel_params = nn.Parameter(torch.rand(n))
        self.offset_params = nn.Parameter(torch.rand(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, input_len).

        Returns:
            Tensor: Output tensor of shape (B, n, width) with soft-selected slices.
        """
        B, C, input_len = x.shape

        # ----- Soft Channel Selection (Vectorized) -----
        # Map each channel parameter (in [0,1]) to a desired channel index in [0, C-1].
        desired_channels = torch.sigmoid(self.channel_params) * (C - 1)  # (n,)
        channels = torch.arange(C, device=x.device).float()  # (C,)
        # Compute Gaussian weights for channels: shape (n, C)
        # Each row corresponds to one slice.
        channel_diff = channels.unsqueeze(0) - desired_channels.unsqueeze(1)  # (n, C)
        channel_weights = torch.exp(-(channel_diff**2) / (2 * self.sigma_channel**2))
        channel_weights = channel_weights / (
            channel_weights.sum(dim=1, keepdim=True) + 1e-8
        )
        # Apply channel weights:
        # Rearrange x to (B, input_len, C) and then perform matrix multiplication with (C, n)
        # to obtain a weighted sum over channels for each slice.
        x_channel = torch.matmul(
            x.transpose(1, 2), channel_weights.T
        )  # (B, input_len, n)
        x_channel = x_channel.transpose(1, 2)  # (B, n, input_len)

        # ----- Soft Time Slicing (Vectorized) -----
        # Map each offset parameter (in [0,1]) to a starting time index in [0, input_len - width].
        start = torch.sigmoid(self.offset_params) * (input_len - self.width)  # (n,)
        # Create time indices and window indices.
        t = torch.arange(input_len, device=x.device).float()  # (input_len,)
        j = torch.arange(self.width, device=x.device).float()  # (width,)
        # For each slice, compute desired positions: shape (n, width)
        desired_positions = start.unsqueeze(1) + j.unsqueeze(0)  # (n, width)

        # Build Gaussian weights for the time dimension:
        # Expand t to (1, 1, input_len) and desired_positions to (n, width, 1)
        t_expanded = t.view(1, 1, input_len)
        desired_positions_expanded = desired_positions.unsqueeze(2)
        # Compute weights: shape (n, width, input_len)
        W = torch.exp(
            -((t_expanded - desired_positions_expanded) ** 2) / (2 * self.sigma_time**2)
        )
        W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)

        # Apply time slicing weights to the channel-selected signal.
        # x_channel is (B, n, input_len) and we want a weighted sum over the input_len dimension.
        # Transpose W to (n, input_len, width) so that each slice's weights can be applied.
        # Using einsum, we compute, for each batch and slice:
        # out[b, i, :] = x_channel[b, i, :] dot (W[i].T)
        return torch.einsum("bni,niw->bnw", x_channel, W.transpose(1, 2))


class LearnablePatternSimilarity(nn.Module):
    def __init__(self, n: int, width: int, eps: float = 1e-8):
        """
        Computes similarity (via centered correlation) between the input and n learnable patterns.
        Each learnable pattern is of length `width` and is defined as the concatenation of a
        learnable parameter vector and a fixed scalar.

        Args:
            n (int): Number of learnable patterns.
            width (int): Length of each pattern.
            eps (float): Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        # For each pattern, create a learnable vector of length (width - 1) and a fixed scalar.
        self.k = nn.Parameter(torch.rand(n, width - 1))
        self.l = nn.Parameter(torch.ones(n, 1), requires_grad=False)

    @property
    def patterns(self) -> torch.Tensor:
        """
        Returns:
            Tensor: Learnable patterns of shape (n, width).
        """
        return torch.cat([self.k, self.l], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the correlation similarity between the input and each learnable pattern.

        Args:
            x (Tensor): Input tensor of shape (B, *A, width).

        Returns:
            Tensor: Similarity scores of shape (B, *A, n), one for each pattern.
        """
        # Get patterns: shape (n, width)
        patterns = self.patterns
        # Compute mean and center patterns along the width dimension.
        p_mean = patterns.mean(dim=-1, keepdim=True)  # (n, 1)
        p_centered = patterns - p_mean  # (n, width)
        p_norm = p_centered.norm(dim=-1, keepdim=True)  # (n, 1)

        # Center the input along its last dimension (width)
        x_mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - x_mean  # (B, *A, width)

        # Reshape x_centered to (N, width) where N = product of batch and other dims.
        orig_shape = x.shape[:-1]  # (B, *A)
        num_vectors = x.numel() // x.shape[-1]
        x_flat = x_centered.reshape(num_vectors, x.shape[-1])  # (N, width)
        x_norm = x_flat.norm(dim=-1, keepdim=True)  # (N, 1)

        # Compute dot product between each input vector and each pattern.
        # p_centered is (n, width), so its transpose is (width, n).
        dot = torch.matmul(x_flat, p_centered.T)  # (N, n)
        denominator = x_norm * p_norm.T + self.eps  # (N, n)
        corr = dot / denominator  # (N, n)

        # Reshape back to (B, *A, n)
        out_shape = orig_shape + (patterns.shape[0],)
        return corr.reshape(out_shape)
