import torch
from torch import nn


class ZeroMean(nn.Module):
    """
    Parameterization layer enforcing per-dimension zero-mean.

    Learns θ ∈ ℝ^{N×D}, exposes b ∈ ℝ^{(N+1)×D} such that:

        b[:-1] = θ
        b[-1]  = -θ.sum(0)

    This ensures:
        b.sum(0) = 0
        b.mean(0) = 0
    """

    def __init__(self, nrows: int, ndims: int, init_std: float = 0.1):
        """
        Args:
            nrows: number of learnable rows (θ has shape nrows x ndims)
            ndims: number of dimensions (columns)
            init_std: std for random initialization
        """
        super().__init__()
        theta = torch.randn(nrows, ndims) * init_std
        self.theta = nn.Parameter(theta)

    def forward(self) -> torch.Tensor:
        """
        Construct b from θ.
        Returns:
            b of shape (nrows+1, ndims)
        """
        last = -self.theta.sum(dim=0, keepdim=True)
        return torch.cat([self.theta, last], dim=0)

    @property
    def value(self) -> torch.Tensor:
        """Convenience alias."""
        return self.forward()

    def project_from_b(self, b: torch.Tensor):
        """
        Take any b of shape (nrows+1, ndims), center it,
        and update θ to match the first nrows rows.

        This is needed for refresh/initialization steps.

        Args:
            b: tensor of shape (nrows+1, ndims)
        """
        assert b.dim() == 2, "b must be 2D"
        nrows_plus, ndims = b.shape
        assert nrows_plus == self.theta.shape[0] + 1
        assert ndims == self.theta.shape[1]

        # Center b per dimension
        b_centered = b - b.mean(dim=0, keepdim=True)

        # Write into θ
        with torch.no_grad():
            self.theta.copy_(b_centered[:-1])


class FixedFirstIntercept(nn.Module):
    """
    Represents intercepts b in R^{ny x D} with gauge fixing b[0, d] = 0.
    Stores parameters only for i >= 1.
    """
    def __init__(self, ny, dim):
        super().__init__()
        self.ny = ny
        self.dim = dim
        self.theta = nn.Parameter(torch.zeros(ny-1, dim))  # b[1:], free

    @property
    def value(self):
        # prepend the fixed zero row
        zero_row = torch.zeros(1, self.dim, device=self.theta.device, dtype=self.theta.dtype)
        return torch.cat([zero_row, self.theta], dim=0)   # (ny, dim)

    def project_from_b_(self, b):
        """
        Project a full b (ny,dim) back into theta, with gauge b[0,:] = 0.
        """
        with torch.no_grad():
            assert b.shape == (self.ny, self.dim)
            self.theta.copy_(b[1:, :])    # simply ignore the first row
