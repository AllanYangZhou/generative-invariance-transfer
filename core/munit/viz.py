import numpy as np
import torch
from torchvision.utils import make_grid


def to_np(tens):
    """Convert pytorch tensor to numpy for matplotlib imshow."""
    return tens.numpy().transpose(1, 2, 0)


@torch.no_grad()
def sample_style_normal(gen, inputs, num_samples=10, centered=False, stddev=1):
    # prior: If True, sample style~N(0,1). Else style~N(0,1)+orig_style.
    """Sample style noise from N(0,1).

    Args:
      centered: If true, style_rand ~ N(0,1). Else, style_rand ~ N(0,1) + style_orig.
    """
    img_dims = inputs.shape[-3:]
    content, style = gen.encode(inputs)
    samples = []
    for _ in range(num_samples):
        sample_style = stddev * torch.randn_like(style)
        if not centered:
            sample_style = sample_style + style
        samples.append(gen.decode(content, sample_style))
    return torch.stack(samples, 1).reshape(-1, *img_dims)


@torch.no_grad()
def make_2d_grid(gen, inp, p1, p2, mean_style, delta=0.3, num_samples=10):
    """
    Make 2d grid of visualizations using first 2 principal components.
    Args:
      inp: shape (1, 3, H, W)
      p1: First principal component, (1, 8, 1, 1)
      p2: Second principal component, (1, 8, 1, 1)
      mean_style: Mean style vector on training dset, (1, 8, 1, 1)
    """
    content, style = gen.encode(inp)
    grid = np.linspace(-delta, delta, num=num_samples)
    out_grid = []
    for i in range(num_samples):
        for j in range(num_samples):
            style_sample = mean_style + grid[i] * p1 + grid[j] * p2
            out_grid.append(gen.decode(content, style_sample).cpu())
    return make_grid(torch.cat(out_grid, 0), nrow=num_samples)


def pca(style_mat):
    # style_mat.shape == (B, d)
    # TODO: naive impl of PCA, should use SVD.
    cov = (style_mat - style_mat.mean(0)).transpose(0, 1) @ (style_mat - style_mat.mean(0))
    eigval, eigvec = torch.eig(cov, eigenvectors=True)
    print(eigval[:, 0])
    return eigvec[:, eigval[:, 0].argsort(descending=True)]
