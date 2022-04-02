import torch

@torch.jit.script
@torch.no_grad()
def hog(image: torch.Tensor, num_bins: int = 9, cell_size: int = 8, norm_p: float=2.):
    assert num_bins > 0
    B, C, H, W = image.size()
    assert H % cell_size == 0, W % cell_size == 0
    image = image.view(B * C, 1, H, W)  # separate channels

    # compute gradient for each channel using convolution
    kernel_x = torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=image.dtype, device=image.device)
    kernel = torch.stack([kernel_x, kernel_x.T]).view(2, 1, 3, 3)  # [out_channels, in_channels, kH, kW]

    grad = torch.conv2d(input=image, weight=kernel, bias=None, stride=1, padding=1)
    grad_x, grad_y = grad[:, 0].contiguous(), grad[:, 1].contiguous()

    # compute discrete gradient
    grad_scale: torch.Tensor = torch.linalg.norm(grad, dim=1, keepdim=True)
    grad_angle = torch.atan2(grad_y, grad_x).rad2deg_().div_(180.0 / num_bins)
    phase = grad_angle.remainder_(num_bins).view(B * C, 1, H, W)  # now ≥ 0
    phase_floor = phase.floor_().long()
    phase_frac = phase.frac()
    discrete = image.new_zeros(B * C, num_bins, H, W)
    discrete.scatter_(dim=1, index=phase_floor, src=grad_scale.mul(1.0 - phase_frac))
    discrete.scatter_add_(dim=1, index=phase_floor.add_(1).remainder_(num_bins), src=grad_scale.mul(phase_frac))

    # cell pooling for histogram
    hist: torch.Tensor = torch._C._nn.avg_pool2d(discrete, cell_size)

    # normalization
    denom: torch.Tensor = torch.linalg.vector_norm(hist, ord=norm_p, dim=1, keepdim=True)
    hist_normalized = hist.div_(denom.clamp_min_(1e-5))

    return hist_normalized.reshape(B, C * num_bins, H // cell_size, W // cell_size)


if __name__ == '__main__':
    from PIL import Image
    import torch.nn.functional as F
    import torchvision.transforms as T
    from skimage import draw
    import numpy as np

    img = T.ToTensor()(Image.open('dog.jpg').convert('RGB').resize([256, 256])).mean(dim=0, keepdim=True)
    print(img.shape)
    f = hog(img[None]).view(9, 32, 32)
    contrast_f = F.relu(f - f.mean(dim=0, keepdim=True))
    hist = contrast_f.numpy()
    s_row = s_col = 256
    c_row = c_col = 8
    n_cells_row = n_cells_col = 32
    orientations = 9
    radius = min(c_row, c_col) // 2 - 1
    orientations_arr = np.arange(orientations)
    # set dr_arr, dc_arr to correspond to midpoints of orientation bins
    orientation_bin_midpoints = (
        np.pi * (orientations_arr + .5) / orientations)
    dr_arr = radius * np.sin(orientation_bin_midpoints)
    dc_arr = radius * np.cos(orientation_bin_midpoints)
    hog_image = np.zeros((s_row, s_col), dtype=np.float32)
    for r in range(n_cells_row):
        for c in range(n_cells_col):
            for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                centre = tuple([r * c_row + c_row // 2,
                                c * c_col + c_col // 2])
                rr, cc = draw.line(int(centre[0] - dc),
                                    int(centre[1] + dr),
                                    int(centre[0] + dc),
                                    int(centre[1] - dr))
                hog_image[rr, cc] += hist[o, r, c]
    T.ToPILImage()(torch.as_tensor(hog_image).view(1, 256, 256)).show()
