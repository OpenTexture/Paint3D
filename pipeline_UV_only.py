import sys
import argparse
import os
import cv2
from tqdm import tqdm
import torch
import torchvision
import time
import numpy as np
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf

from controlnet.diffusers_cnet_txt2img import txt2imgControlNet
from controlnet.diffusers_cnet_inpaint import inpaintControlNet
from controlnet.diffusers_cnet_img2img import img2imgControlNet
from paint3d.dataset import init_dataloaders
from paint3d import utils
from paint3d.models.textured_mesh import TexturedMeshModel
from paint3d.trainer import dr_eval, forward_texturing


def UV_gen(sd_cfg, cnet, mesh_model, outdir,):
    print(f"rendering texture and position map")
    UV_pos = mesh_model.UV_pos_render()
    UV_pos_path = os.path.join(outdir, f"UV_pos.png")
    utils.save_tensor_image(UV_pos.permute(0, 3, 1, 2), UV_pos_path)

    # UV inpaint
    p_cfg = sd_cfg.txt2img
    p_cfg.controlnet_units[0].condition_image_path = UV_pos_path

    images = cnet.infernece(config=p_cfg)
    for i, img in enumerate(images):
        save_path = os.path.join(outdir, f'UV_gen_res_{i}.png')
        img.save(save_path)
    return images


def init_process(opt):
    outdir = opt.outdir
    os.makedirs(outdir, exist_ok=True)

    pathdir, filename = Path(opt.render_config).parent, Path(opt.render_config).stem
    sys.path.append(str(pathdir))
    render_cfg = __import__(filename, ).TrainConfig()
    utils.seed_everything(render_cfg.optim.seed)

    sd_cfg = OmegaConf.load(opt.sd_config)
    render_cfg.log.exp_path = str(outdir)
    if opt.prompt is not None:
        sd_cfg.inpaint.prompt = opt.prompt
    if opt.mesh_path is not None:
        render_cfg.guide.shape_path = opt.mesh_path
    if opt.texture_path is not None:
        render_cfg.guide.initial_texture = opt.texture_path
        img = Image.open(opt.texture_path)
        render_cfg.guide.texture_resolution = img.size
    return sd_cfg, render_cfg


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sd_config",
        type=str,
        default="stable-diffusion/v2-inpainting-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--render_config",
        type=str,
        default=" ",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="prompt",
        default=None,
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        help="path of mesh",
        default=None,
    )
    parser.add_argument(
        "--texture_path",
        type=str,
        help="path of texture image",
        default=None,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/inpainting-samples"
    )

    opt = parser.parse_args()
    return opt


def main():
    print("Texture diffusion")
    opt = parse()
    sd_cfg, render_cfg = init_process(opt)

    # ===  1. create model
    device = torch.device("cuda")
    mesh_model = TexturedMeshModel(cfg=render_cfg, device=device,).to(device)
    dataloaders = init_dataloaders(render_cfg, device)

    UV_cnet = txt2imgControlNet(sd_cfg.txt2img)


    total_start = time.time()
    # ===  2. UV gen
    start_t = time.time()
    _ = UV_gen(
        sd_cfg=sd_cfg,
        cnet=UV_cnet,
        mesh_model=mesh_model,
        outdir=opt.outdir,
    )
    print(f"UV gen time: {time.time() - start_t}")


if __name__ == '__main__':
    main()
