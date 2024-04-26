import os
import torch
import random
import imageio
import cv2
import numpy as np
from PIL import Image
from typing import List
import torch.nn.functional as F


def color_with_shade(color: List[float], z_normals: torch.Tensor, light_coef=0.7):
    normals_with_light = (light_coef + (1 - light_coef) * z_normals.detach())
    shaded_color = torch.tensor(color).view(1, 3, 1, 1).to(z_normals.device) * normals_with_light
    return shaded_color


def tensor2numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor[0].permute(1, 2, 0).contiguous().clamp(0, 1).detach()  # [N, C, H, W ]-->[C, H, W]-->[H, W, C]
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return tensor


def pil2tensor(pil_img, device=torch.device('cpu')):
    tensor_chw = torch.Tensor(np.array(pil_img)).to(device).permute(2, 0, 1) / 255.0
    return tensor_chw.unsqueeze(0)


def save_tensor_image(tensor: torch.Tensor, save_path: str):
    if len(os.path.dirname(save_path)) > 0 and not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)  # [1, c, h, w]-->[c, h, w]
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)  
    tensor = tensor.permute(1, 2, 0).detach().cpu().numpy()  # [c, h, w]-->[h, w, c]
    Image.fromarray((tensor * 255).astype(np.uint8)).save(save_path)


def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w


def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d


def gaussian_blur(image:torch.Tensor, kernel_size:int, std:int) -> torch.Tensor:
    gaussian_filter = gkern(kernel_size, std=std)
    gaussian_filter /= gaussian_filter.sum()

    image = F.conv2d(image, gaussian_filter.unsqueeze(0).unsqueeze(0).cuda(), padding=kernel_size // 2)
    return image


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def save_video(video_frames, save_path):
    imageio.mimsave(save_path, video_frames, fps=25, quality=8, macro_block_size=1)


def split_grid_image(
        img: np.array,
        size: tuple
) -> np.array:
    """
    split grid image to batch image
    """
    H, W, C = img.shape
    row, col = size
    res = []

    h, w = H // row, W // col
    for i in range(row):
        for j in range(col):
            sub_img = img[i * h:(i + 1) * h, j * w:(j + 1) * w, ...]
            res.append(sub_img)
    return res


def inpaint_atlas(image, append_mask=None):
    src_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    src_h = src_hls[:, :, 0]
    tgt_range, thres = 150, 1
    lowerb = tgt_range - thres
    upperb = tgt_range + thres
    mask = cv2.inRange(src=src_h, lowerb=lowerb, upperb=upperb)

    if append_mask is not None:
        mask = np.clip(mask + append_mask[..., 0], 0, 1).astype(np.uint8)
    image_inpaint = cv2.inpaint(src=image, inpaintMask=mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    return image_inpaint


def mask_postprocess(depth_render: torch.Tensor, mask_render: torch.Tensor,
                     z_normal_render: torch.Tensor, z_normals_cache: torch.Tensor, uncolored_mask_render: torch.Tensor,
                     strict_projection=True, z_update_thr=0.2):
    uncolored_mask = torch.from_numpy(
        cv2.dilate(uncolored_mask_render[0, 0].detach().cpu().numpy(), np.ones((19, 19), np.uint8))).to(
        uncolored_mask_render.device).unsqueeze(0).unsqueeze(0)
    update_mask = uncolored_mask.clone()

    object_mask = torch.ones_like(update_mask)
    object_mask[depth_render == 0] = 0
    object_mask = torch.from_numpy(
        cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((7, 7), np.uint8))).to(
        object_mask.device).unsqueeze(0).unsqueeze(0)
    update_mask[torch.bitwise_and(object_mask == 0, uncolored_mask == 0)] = 0

    object_mask = torch.from_numpy(cv2.erode(mask_render[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))
                                   ).to(mask_render.device).unsqueeze(0).unsqueeze(0)
    render_update_mask = object_mask.clone()
    render_update_mask[update_mask == 0] = 0
    blurred_render_update_mask = torch.from_numpy(
        cv2.dilate(render_update_mask[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
        render_update_mask.device).unsqueeze(0).unsqueeze(0)
    blurred_render_update_mask = gaussian_blur(blurred_render_update_mask, 21, 16)
    blurred_render_update_mask[object_mask == 0] = 0
    if strict_projection:
        blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0
        z_was_better = z_normal_render + z_update_thr < z_normals_cache[:, :1, :, :]
        blurred_render_update_mask[z_was_better] = 0
    render_update_mask = blurred_render_update_mask

    return render_update_mask


def dilate_depth_outline(path, iters=5, dilate_kernel=3):
    ori_img = cv2.imread(path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

    img = ori_img
    for i in range(iters):
        _, mask = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        mask = mask / 255

        img_dilate = cv2.dilate(img, np.ones((dilate_kernel, dilate_kernel), np.uint8))

        img = (mask * img + (1 - mask) * img_dilate).astype(np.uint8)
    return img


def extract_bg_mask(img_path, mask_color=[204, 25, 204], dilate_kernel=5):
    """
    :param mask_color:  BGR
    :return:
    """
    img = cv2.imread(img_path)

    mask = (img == mask_color).all(axis=2).astype(np.float32)
    mask = mask[:,:,np.newaxis]

    mask = cv2.dilate(mask, np.ones((dilate_kernel, dilate_kernel), np.uint8))[:,:,np.newaxis]
    mask = (mask * 255).astype(np.uint8)
    return mask