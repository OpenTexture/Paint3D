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
from paint3d import utils
from paint3d.models.textured_mesh import TexturedMeshModel
from paint3d.dataset import init_dataloaders
from paint3d.trainer import dr_eval, forward_texturing


def inpaint_viewpoint(sd_cfg, cnet, save_result_dir, mesh_model, dataloaders, inpaint_view_ids=[(5, 6)]):
    # projection
    print(f"Project inpaint view {inpaint_view_ids}...")
    view_angle_info = {i:data for i, data in enumerate(dataloaders['train'])}
    inpaint_used_key = ["image", "depth", "uncolored_mask"]
    for i, one_batch_id in tqdm(enumerate(inpaint_view_ids)):
        one_batch_img = []
        for view_id in one_batch_id:
            data = view_angle_info[view_id]
            theta, phi, radius = data['theta'], data['phi'], data['radius']
            outputs = mesh_model.render(theta=theta, phi=phi, radius=radius)
            view_img_info = [outputs[k] for k in inpaint_used_key]
            one_batch_img.append(view_img_info)

        for i, img in enumerate(zip(*one_batch_img)):
            img = torch.cat(img, dim=3)
            if img.size(1) == 1:
                img = img.repeat(1, 3, 1, 1)
            t = '_'.join(map(str, one_batch_id))
            name = inpaint_used_key[i]
            if name == "uncolored_mask":
                img[img>0] = 1
            save_path = os.path.join(save_result_dir, f"view_{t}_{name}.png")
            utils.save_tensor_image(img, save_path=save_path)

    # inpaint view point
    txt_cfg = sd_cfg.txt2img
    img_cfg = sd_cfg.inpaint
    copy_list = ["prompt", "negative_prompt", "seed", ]
    for k in copy_list:
        img_cfg[k] = txt_cfg[k]

    for i, one_batch_id in tqdm(enumerate(inpaint_view_ids)):
        t = '_'.join(map(str, one_batch_id))
        rgb_path = os.path.join(save_result_dir, f"view_{t}_{inpaint_used_key[0]}.png")
        depth_path = os.path.join(save_result_dir, f"view_{t}_{inpaint_used_key[1]}.png")
        mask_path = os.path.join(save_result_dir, f"view_{t}_{inpaint_used_key[2]}.png")

        # pre-processing inpaint mask: dilate
        mask = cv2.imread(mask_path)
        dilate_kernel = 10
        mask = cv2.dilate(mask, np.ones((dilate_kernel, dilate_kernel), np.uint8))
        mask_path = os.path.join(save_result_dir, f"view_{t}_{inpaint_used_key[2]}_d{dilate_kernel}.png")
        cv2.imwrite(mask_path, mask)

        img_cfg.image_path = rgb_path
        img_cfg.mask_path =  mask_path
        img_cfg.controlnet_units[0].condition_image_path = depth_path
        images = cnet.infernece(config=img_cfg)
        for i, img in enumerate(images):
            save_path = os.path.join(save_result_dir, f"view_{t}_rgb_inpaint_{i}.png")
            img.save(save_path)
    return images


def gen_init_view(sd_cfg, cnet, mesh_model, dataloaders, outdir, view_ids=[]):
    print(f"Project init view {view_ids}...")
    init_depth_map = []
    view_angle_info = {i: data for i, data in enumerate(dataloaders['train'])}
    for view_id in view_ids:
        data = view_angle_info[view_id]
        theta, phi, radius = data['theta'], data['phi'], data['radius']
        outputs = mesh_model.render(theta=theta, phi=phi, radius=radius)
        depth_render = outputs['depth']
        init_depth_map.append(depth_render)

    init_depth_map = torch.cat(init_depth_map, dim=0).repeat(1, 3, 1, 1)
    init_depth_map = torchvision.utils.make_grid(init_depth_map, nrow=2, padding=0)
    save_path = os.path.join(outdir, f"init_depth_render.png")
    utils.save_tensor_image(init_depth_map.unsqueeze(0), save_path=save_path)

    # post-processing depth，dilate
    depth_dilated = utils.dilate_depth_outline(save_path, iters=5, dilate_kernel=3)
    save_path = os.path.join(outdir, f"init_depth_dilated.png")
    cv2.imwrite(save_path, depth_dilated)

    print("Generating init view...")
    p_cfg = sd_cfg.txt2img
    p_cfg.controlnet_units[0].condition_image_path = save_path

    images = cnet.infernece(config=p_cfg)
    for i, img in enumerate(images):
        save_path = os.path.join(outdir, f'init-img-{i}.png')
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
        sd_cfg.txt2img.prompt = opt.prompt
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
    print("Depth-based 3D Texturing")
    opt = parse()
    sd_cfg, render_cfg = init_process(opt)

    # ===  1. create model and data
    device = torch.device("cuda")
    dataloaders = init_dataloaders(render_cfg, device)
    mesh_model = TexturedMeshModel(cfg=render_cfg, device=device,).to(device)

    depth_cnet = txt2imgControlNet(sd_cfg.txt2img)
    inpaint_cnet = inpaintControlNet(sd_cfg.inpaint)


    # ===  2. init view generation
    total_start = time.time()
    start_t = time.time()
    init_images = gen_init_view(
        sd_cfg=sd_cfg,
        cnet=depth_cnet,
        mesh_model=mesh_model,
        dataloaders=dataloaders,
        outdir=opt.outdir,
        view_ids=render_cfg.render.views_init,
    )
    print(f"init view generation time: {time.time() - start_t}")
    # init_image_paths = Path(opt.outdir)
    # init_image_paths = list(init_image_paths.glob("init-img-*.png"))
    # init_image_paths.sort()
    # init_images = [Image.open(str(p)) for p in init_image_paths]


    for i, init_image in enumerate(init_images):
        outdir = Path(opt.outdir) / f"res-{i}"
        outdir.mkdir(exist_ok=True)
        #  back-projection init view
        start_t = time.time()
        mesh_model.initial_texture_path = None
        mesh_model.refresh_texture()
        view_imgs = utils.split_grid_image(img=np.array(init_image), size=(1, 2))
        forward_texturing(
            cfg=render_cfg,
            dataloaders=dataloaders,
            mesh_model=mesh_model,
            save_result_dir=outdir,
            device=device,
            view_imgs=view_imgs,
            view_ids=render_cfg.render.views_init,
            verbose=False,
        )
        print(f"init DR time: {time.time() - start_t}")

        # === 3. depth based inpaint
        for view_group in render_cfg.render.views_inpaint:   # cloth 4 view
            start_t = time.time()
            print("View inpainting ...")
            outdir = Path(opt.outdir) / f"res-{i}"
            outdir.mkdir(exist_ok=True)
            inpainted_images = inpaint_viewpoint(
                sd_cfg=sd_cfg,
                cnet=inpaint_cnet,
                save_result_dir=outdir,
                mesh_model=mesh_model,
                dataloaders=dataloaders,
                inpaint_view_ids=[view_group],
            )
            print(f"inpaint view generation time: {time.time() - start_t}")


            start_t = time.time()
            view_imgs = []
            for img_t in inpainted_images:
                view_imgs.extend(utils.split_grid_image(img=np.array(img_t), size=(1, 2)))
            forward_texturing(
                cfg=render_cfg,
                dataloaders=dataloaders,
                mesh_model=mesh_model,
                save_result_dir=outdir,
                device=device,
                view_imgs=view_imgs,
                view_ids=view_group,
                verbose=False,
            )
            print(f"inpaint DR time: {time.time() - start_t}")


        print(f"total processed time:{time.time() - total_start}")
        mesh_model.initial_texture_path = f"{outdir}/albedo.png"
        mesh_model.refresh_texture()
        dr_eval(
            cfg=render_cfg,
            dataloaders=dataloaders,
            mesh_model=mesh_model,
            save_result_dir=outdir,
            valset=True,
            verbose=False,
        )
        mesh_model.empty_texture_cache()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
