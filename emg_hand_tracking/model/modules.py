from torch import nn, Tensor
import torch


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


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


class LearnablePatternCosSimilarity(nn.Module):
    def __init__(self, n: int, width: int, eps: float = 1e-8):
        """
        Computes cos similarity between the input and n learnable patterns.

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
        p_norm = patterns.norm(dim=-1, keepdim=True)  # (n, 1)

        # Reshape x_centered to (N, width) where N = product of batch and other dims.
        orig_shape = x.shape[:-1]  # (B, *A)
        num_vectors = x.numel() // x.shape[-1]
        x_flat = x.reshape(num_vectors, x.shape[-1])  # (N, width)
        x_norm = x_flat.norm(dim=-1, keepdim=True)  # (N, 1)

        # Compute dot product between each input vector and each pattern.
        # p_centered is (n, width), so its transpose is (width, n).
        dot = torch.matmul(x_flat, patterns.T)  # (N, n)
        denominator = x_norm * p_norm.T + self.eps  # (N, n)
        corr = dot / denominator  # (N, n)

        # Reshape back to (B, *A, n)
        out_shape = orig_shape + (patterns.shape[0],)
        return corr.reshape(out_shape)


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


class LearnablePatternUnnormSimilarity(nn.Module):
    def __init__(self, n: int, width: int):
        """
        Computes similarity (via centered dot) between the input and n learnable patterns.

        Args:
            n (int): Number of learnable patterns.
            width (int): Length of each pattern.
            eps (float): Small constant for numerical stability.
        """
        super().__init__()
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

        # Center the input along its last dimension (width)
        x_mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - x_mean  # (B, *A, width)

        # Reshape x_centered to (N, width) where N = product of batch and other dims.
        orig_shape = x.shape[:-1]  # (B, *A)
        num_vectors = x.numel() // x.shape[-1]
        x_flat = x_centered.reshape(num_vectors, x.shape[-1])  # (N, width)

        # Compute dot product between each input vector and each pattern.
        # p_centered is (n, width), so its transpose is (width, n).
        dot = torch.matmul(x_flat, p_centered.T)  # (N, n)

        # Reshape back to (B, *A, n)
        out_shape = orig_shape + (patterns.shape[0],)
        return dot.reshape(out_shape)


class LearnablePatternDot(nn.Module):
    def __init__(self, n: int, width: int, eps: float = 1e-8):
        """
        Computes cos similarity between the input and n learnable patterns.

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

        # Reshape x_centered to (N, width) where N = product of batch and other dims.
        orig_shape = x.shape[:-1]  # (B, *A)
        num_vectors = x.numel() // x.shape[-1]
        x_flat = x.reshape(num_vectors, x.shape[-1])  # (N, width)

        # Compute dot product between each input vector and each pattern.
        # p_centered is (n, width), so its transpose is (width, n).
        dot = torch.matmul(x_flat, patterns.T)  # (N, n)
        corr = dot  # (N, n)

        # Reshape back to (B, *A, n)
        out_shape = orig_shape + (patterns.shape[0],)
        return corr.reshape(out_shape)


class ExtractLearnableSlices(nn.Module):
    def __init__(self, n: int, width: int):
        super().__init__()
        self.n = n
        self.width = width
        self.channel_params = nn.Parameter(torch.rand(n))
        self.offset_params = nn.Parameter(torch.rand(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape

        # --- Channel Interpolation ---
        desired_channels = torch.sigmoid(self.channel_params) * (C - 1)
        floor_channels = torch.floor(desired_channels).long()
        ceil_channels = (floor_channels + 1).clamp(max=C - 1)
        w_channel = (desired_channels - floor_channels.float()).view(1, -1, 1)

        floor_indices = floor_channels.view(1, -1, 1).expand(B, -1, L)
        ceil_indices = ceil_channels.view(1, -1, 1).expand(B, -1, L)
        x_floor_channel = torch.gather(x, dim=1, index=floor_indices)
        x_ceil_channel = torch.gather(x, dim=1, index=ceil_indices)
        x_channel = torch.lerp(x_floor_channel, x_ceil_channel, w_channel)

        # --- Time Interpolation ---
        t0 = torch.sigmoid(self.offset_params) * (L - self.width)
        j = torch.arange(self.width, device=x.device, dtype=x.dtype)
        pos = t0.unsqueeze(1) + j.unsqueeze(0)
        pos_floor = torch.floor(pos).long()
        w_time = (pos - pos_floor.float()).unsqueeze(0)  # shape: (1, n, width)

        pos_floor_exp = pos_floor.unsqueeze(0).expand(B, -1, -1)
        x_floor_time = torch.gather(x_channel, dim=2, index=pos_floor_exp)

        # For the ceiling, use the same trick:
        pos_ceil = (pos_floor + 1).clamp(max=L - 1)
        pos_ceil_exp = pos_ceil.unsqueeze(0).expand(B, -1, -1)
        x_ceil_time = torch.gather(x_channel, dim=2, index=pos_ceil_exp)

        return torch.lerp(x_floor_time, x_ceil_time, w_time)


class Parallel(nn.Module):
    """
    Applies multiple modules in parallel to the same input and concatenates their outputs.
    """

    def __init__(self, *modules):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x):
        # Apply each module in parallel and concatenate their outputs along the last dimension
        outputs = [module(x) for module in self.modules_list]
        return torch.cat(outputs, dim=-1)


class Variance(nn.Module):
    """
    Computes the variance of the input tensor along the last dimension.
    """

    def forward(self, x):
        return x.var(dim=-1, unbiased=False)  # Variance along the last dimension


class Mean(nn.Module):
    """
    Computes the mean of the input tensor along the last dimension.
    """

    def forward(self, x):
        return x.mean(dim=-1)  # Mean along the last dimension


class Max(nn.Module):
    """
    Computes the maximum value of the input tensor along the last dimension.
    """

    def forward(self, x):
        return x.max(dim=-1).values  # Max along the last dimension


class StdDev(nn.Module):
    """
    Computes the standard deviation of the input tensor along the last dimension.
    """

    def forward(self, x):
        return x.std(
            dim=-1, unbiased=False
        )  # Standard deviation along the last dimension
