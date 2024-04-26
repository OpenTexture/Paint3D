import cv2
import numpy as np


def repeat_pixel_2d(img, n):
    """
    :param img: (H,W)
    :param ratio:
    :return:
    """
    h, w = img.shape
    a = img[:, :, np.newaxis].repeat(n*n, axis=2)
    # unshuffle
    b = a.reshape(h, w, n, n, )
    c = b.transpose(0, 2, 1, 3)
    d = c.reshape(n * h, n * w)

    return d


def repeat_pixel(img, n):
    '''
    :param img: (H, W, C) or (H W)
    :param n:
    :return:
    '''
    if len(img.shape) == 2:
        return repeat_pixel_2d(img, n)
    elif len(img.shape) == 3:
        h,w,c = img.shape
        res_img = []
        for i in range(c):
            i_img = img[:,:,i]
            res_img.append(repeat_pixel_2d(i_img, n))
        res_img = np.stack(res_img, axis=2)
        return res_img
    else:
        raise NotImplementedError


def build_2d_gaussion(ksize=5):
    img = np.zeros((ksize, ksize))
    img[ksize // 2, ksize // 2] = 1
    k = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return k


def make_strided_arr(x, kernel_size=(3, 3), c_stride = 2):
    '''
    :param x:  arr has (H, W) shape
    :param kernel_size:
    :param c_stride:
    :return:
    '''
    view_shape = tuple(np.subtract(x.shape, kernel_size) // c_stride + 1) + kernel_size
    strides = tuple([c_stride * i for i in x.strides]) + x.strides
    sub_matrices = np.lib.stride_tricks.as_strided(x, view_shape, strides)
    return sub_matrices


def customize_down_sample_image(image, mask, ksize=9, ratio=2):
    """
    image - (H,W,C)
    mask - (H,W)
    ksize - download kernel size
    ratio - downsample ratio
    """
    assert ratio >= 1.0
    kernel = build_2d_gaussion(ksize)
    image_pad = cv2.copyMakeBorder(image, ksize // 2, ksize // 2, ksize // 2, ksize // 2, cv2.BORDER_CONSTANT,
                                   value=[0, 0, 0])
    mask_pad = cv2.copyMakeBorder(mask.astype(np.float64), ksize // 2, ksize // 2, ksize // 2, ksize // 2,
                                  cv2.BORDER_CONSTANT, value=[0])

    # make mask to sub_grid
    mask_strided = make_strided_arr(mask_pad, kernel_size=(ksize, ksize), c_stride=ratio)
    # print(mask_strided.shape)
    down_mask = mask_strided.sum(3).sum(2)
    down_mask[down_mask>0] = 1
    # print(down_mask.shape)

    # cal adaptive kernel and normalize it
    adaptive_kernel = mask_strided * kernel.reshape(1, 1, ksize, ksize)
    kernel_sum = adaptive_kernel.sum(3).sum(2)
    adaptive_kernel = adaptive_kernel / (kernel_sum[:, :, np.newaxis, np.newaxis] + 1e-12)
    # print(adaptive_kernel.shape)

    # Conv
    h, w, c = image_pad.shape
    down_img = []
    for i in range(c):
        i_img = image_pad[:,:,i]
        i_img = make_strided_arr(i_img, kernel_size=(ksize, ksize), c_stride=ratio)
        i_img_conv = np.einsum('klij,klij->kl', adaptive_kernel, i_img)
        down_img.append(i_img_conv)
    down_img = np.stack(down_img, axis=2)
    return down_img, down_mask


def fill_image_by_mipmap(image, mask=None, mask_color=None, ksize=9, ratio=2):
    """
        image - [h, w, 3] in opencv
        mask - [h, w] bool
        mask_color - default color for uncolored

        Return:
        image or albedo without hole
    """

    if mask is not None:
        mask_color = [-1, -1, -1]
        pass
    elif mask_color is not None:
        mask = ~(image == mask_color).all(axis=2)
    else:
        assert 0, "Need mask or mask color input"

    # build levels of mipmap
    mipmap_queue = []
    mask_queue = []
    mipmap = image
    mask_mipmap = mask.astype(np.float32)
    while (~ mask_mipmap.all()):
        mipmap, mask_mipmap = customize_down_sample_image(mipmap, mask_mipmap, ksize, ratio)
        mipmap_queue.append(mipmap)
        mask_queue.append(mask_mipmap)

    # for i, img in enumerate(mipmap_queue):
    #     cv2.imwrite(f"mipmap_{i}.png", img)

    # === fill bg
    mipmap_queue.insert(0, image)
    mask_queue.insert(0, mask.astype(np.float32))
    levels = len(mipmap_queue)
    for i in range(levels - 1, -1, -1):
        if i == levels - 1:
            img_blur = mipmap_queue[i]
            continue
        mipmap, mask = mipmap_queue[i], mask_queue[i]

        # fuse mipmap and img_blur
        # print(np.unique(mask))
        mask_blur = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = mask * mask_blur
        mask = mask[:,:,np.newaxis].repeat(3, axis=2)

        h,w,c = img_blur.shape
        img_blur = cv2.resize(img_blur, (w*ratio,h*ratio), interpolation=cv2.INTER_LINEAR)
        # img_blur = repeat_pixel(img_blur, n=ratio)

        img_blur = mipmap * mask + (1 - mask) * img_blur
    return img_blur

