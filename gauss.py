import torch
import torch.nn.functional as F
import math


def create_gaussian_kernel(sigma, kernel_size):
    x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    y = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def gaussmap(input_tensor):
    W = 45
    sigma = W / (2 * math.sqrt(2 * math.log(2)))
    kernel_size = 1 + 2 * (int(round(6 * sigma)) // 2)

    with torch.no_grad():
        kernel = create_gaussian_kernel(sigma, kernel_size).to(input_tensor.device)

    padding = kernel_size // 2
    blurred = F.conv2d(input_tensor, kernel, padding=padding, groups=1)

    batch_size = blurred.shape[0]
    min_vals = blurred.view(batch_size, -1).min(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
    max_vals = blurred.view(batch_size, -1).max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
    epsilon = 1e-6
    normalized = (blurred - min_vals) / (max_vals - min_vals + epsilon) * 255

    return normalized