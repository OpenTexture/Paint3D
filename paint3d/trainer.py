import os
import sys
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

sys.path.append("../")
from paint3d import utils
import paint3d.dataset as dataset
from paint3d.config.train_config_paint3d import TrainConfig
from paint3d.models.textured_mesh import TexturedMeshModel


def forward_texturing(cfg, dataloaders, mesh_model, save_result_dir, device,
                      view_imgs=[], view_ids=[], verbose=False):
    assert len(view_imgs) == len(view_ids), "the number of view_imgs should equal to the number of view_ids"

    view_info = {}
    for view_id, view_img in zip(view_ids, view_imgs):
        view_info[view_id] = {"img": view_img}

    for view_id, data in tqdm(enumerate(dataloaders['train'])):
        if view_id not in view_ids:
            continue
        theta, phi, radius = data['theta'], data['phi'], data['radius']
        view_info[view_id]["pos"] = (theta, phi, radius)

    for view_id in view_ids:
        view_img = view_info[view_id]["img"]
        theta, phi, radius = view_info[view_id]["pos"]
        target_sd = utils.pil2tensor(Image.fromarray(view_img).convert('RGB').resize((cfg.render.grid_size, cfg.render.grid_size), resample=Image.BILINEAR), device)
        mesh_model.forward_texturing(theta=theta, phi=phi, radius=radius, view_target=target_sd,
                                     save_result_dir=save_result_dir, view_id=view_id, verbose=verbose)

    mesh_model.export_mesh(save_result_dir)


def dr_train(cfg, dataloaders, mesh_model, save_result_dir, device, view_imgs=[], view_ids=[], verbose=False):
    assert len(view_imgs) == len(view_ids), "the number of view_imgs should equal to the number of view_ids"
    view_map = {}
    for k, v in zip(view_ids, range(len(view_ids))):
        view_map[k] = v

    for view_id, data in tqdm(enumerate(dataloaders['train'])):
        if view_id not in view_ids:
            continue

        theta, phi, radius = data['theta'], data['phi'], data['radius']
        outputs = mesh_model.render(theta=theta, phi=phi, radius=radius)
        render_cache, uncolored_mask_render = outputs['render_cache'], outputs['uncolored_mask']
        rgb_render, depth_render, mask_render = outputs['image'], outputs['depth'], outputs['mask']
        z_normal_render = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        z_normal_show = torch.ones_like(z_normal_render)*(1-mask_render)+z_normal_render*mask_render

        meta_output = mesh_model.render(use_meta_texture=True, render_cache=render_cache)
        z_normals_cache = meta_output['image'].clamp(0, 1)
        z_normals_cache[:, 0, :, :] = torch.max(z_normals_cache[:, 0, :, :], z_normal_render[:, 0, :, :])
        uncolored_mask_updated = torch.from_numpy(
            cv2.erode(uncolored_mask_render[0, 0].detach().cpu().numpy(), np.ones((11, 11), np.uint8))).to(
            uncolored_mask_render.device).unsqueeze(0).unsqueeze(0)

        target_sd = utils.pil2tensor(Image.fromarray(view_imgs[view_map[view_id]]).convert('RGB').
                                     resize((cfg.render.grid_size, cfg.render.grid_size),
                                            resample=Image.BILINEAR), device)

        if verbose:
            utils.save_tensor_image(target_sd, os.path.join(save_result_dir, f"view{view_id}_target_sd.png"))
            utils.save_tensor_image(target_sd * uncolored_mask_render,
                                    os.path.join(save_result_dir, f"view{view_id}_target_sd_r.png"))
            utils.save_tensor_image(target_sd * uncolored_mask_updated,
                                os.path.join(save_result_dir, f"view{view_id}_target_sd_u.png"))

        optimizer = torch.optim.Adam(mesh_model.get_params(), lr=cfg.optim.lr, betas=(0.9, 0.99), eps=1e-15)
        for _ in tqdm(range(cfg.optim.train_step), desc='rendering texture of {}th view [theta:{}, phi:{}, radius:{}]'.
                format(view_id, theta, phi, radius)):
            optimizer.zero_grad()
            outputs = mesh_model.render(render_cache=render_cache)
            rgb_render = outputs['image']
            uncolored_mask = uncolored_mask_updated.flatten()
            masked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)[:, :, uncolored_mask > 0]
            masked_target = target_sd.reshape(1, target_sd.shape[1], -1)[:, :, uncolored_mask > 0]
            uncolored_mask = uncolored_mask[uncolored_mask > 0]

            loss = ((masked_pred - masked_target.detach()).pow(2) * uncolored_mask).mean() + (
                    (masked_pred - masked_pred.detach()).pow(2) * (1 - uncolored_mask)).mean()

            meta_outputs = mesh_model.render(use_meta_texture=True, render_cache=render_cache)
            current_z_normals = meta_outputs['image']
            current_z_mask = meta_outputs['mask'].flatten()
            masked_current_z_normals = current_z_normals.reshape(1, current_z_normals.shape[1], -1)[:, :,
                                       current_z_mask == 1][:, :1]
            masked_last_z_normals = z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :,
                                    current_z_mask == 1][:, :1]
            loss += (masked_current_z_normals - masked_last_z_normals.detach()).pow(2).mean()
            loss.backward()
            optimizer.step()

        if verbose:
            pass
    mesh_model.export_mesh(save_result_dir)


@torch.no_grad()
def dr_eval(cfg, dataloaders, mesh_model, save_result_dir, valset=False, verbose=False):
    all_render_out_frames = []
    all_render_rgb_frames = []
    mesh_model.renderer.clear_seen_faces()
    val_dataset = dataloaders['val_large'] if valset else dataloaders['train']
    for i, data in tqdm(enumerate(val_dataset), desc="Evalating textured mesh~"):
        phi = data['phi']
        phi = float(phi + 2 * np.pi if phi < 0 else phi)

        outputs = mesh_model.render(theta=data['theta'], phi=phi, radius=data['radius'], dims=(1024, 1024))
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        mask, uncolored_masks = outputs['mask'], outputs['uncolored_mask']
        color_with_shade_img = utils.color_with_shade([0.85, 0.85, 0.85], z_normals=z_normals, light_coef=0.3)
        rgb_render = outputs['image'] * (1 - uncolored_masks) + color_with_shade_img * uncolored_masks

        z_normals = torch.ones_like(z_normals) * (1 - mask) + z_normals * mask
        render_outs = utils.tensor2numpy(torch.cat([rgb_render,
                                                    outputs['depth'].repeat(1, 3, 1, 1),
                                                    z_normals.repeat(1, 3, 1, 1),
                                                    ], dim=3))
        all_render_out_frames.append(render_outs)
        all_render_rgb_frames.append(utils.tensor2numpy(rgb_render))

        if verbose:
            save_path = os.path.join(save_result_dir, f"view{i}_rgb_render.png")
            utils.save_tensor_image(rgb_render, save_path=save_path)

            save_path = os.path.join(save_result_dir, f"view{i}_depth_render.png")
            utils.save_tensor_image(outputs['depth'].repeat(1, 3, 1, 1), save_path=save_path)

            save_path = os.path.join(save_result_dir, f"view{i}_uncolored_mask.png")
            utils.save_tensor_image(uncolored_masks.repeat(1, 3, 1, 1), save_path=save_path)

            save_path = os.path.join(save_result_dir, f"view{i}_normal.png")
            utils.save_tensor_image(z_normals.repeat(1, 3, 1, 1), save_path=save_path)

    utils.save_video(np.stack(all_render_out_frames, axis=0), os.path.join(save_result_dir, "render_all.mp4"))
    utils.save_video(np.stack(all_render_rgb_frames, axis=0), os.path.join(save_result_dir, "render_rgb.mp4"))


if __name__ == '__main__':
    device = torch.device("cuda")
    cfg = TrainConfig()
    if not os.path.exists(cfg.log.exp_path):
        os.makedirs(cfg.log.exp_path)

    utils.seed_everything(cfg.optim.seed)
    dataloaders = dataset.init_dataloaders(cfg, device)

    mesh_model = TexturedMeshModel(cfg=cfg, device=device)
    mesh_model = mesh_model.to(device)

    debug_view_image_dir = "xxx"
    debug_view_image_names = os.listdir(debug_view_image_dir)
    debug_view_image_names.sort()
    view_images, view_ids = [], []
    for image_name in ["0006.png", "0007.png", "0001.png", "00016.png"]:
        view_image = cv2.imread(os.path.join(debug_view_image_dir, image_name))
        view_images.append(cv2.cvtColor(view_image, cv2.COLOR_BGR2RGB))
        view_ids.append(int(image_name[3:image_name.find(".")])-1)

    # 1.test forward texturing
    forward_texturing(cfg, dataloaders, mesh_model, cfg.log.exp_path, device,
                      view_imgs=view_images, view_ids=view_ids, post_process=True)

    # 2.test DR texturing
    # dr_train(cfg, dataloaders, mesh_model, cfg.log.exp_path, device,
    #          view_imgs=view_images, view_ids=view_ids, post_process=True)

    # eval
    dr_eval(cfg, dataloaders, mesh_model, cfg.log.exp_path, valset=True, verbose=False)

