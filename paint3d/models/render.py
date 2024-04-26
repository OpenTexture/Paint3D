import kaolin as kal
import torch
import numpy as np
import trimesh

from paint3d.utils import save_tensor_image


def prepare_vertices(vertices, faces, intrinsics, camera_rot=None, camera_trans=None, camera_transform=None):
    if camera_transform is None:
        assert camera_trans is not None and camera_rot is not None, \
            "camera_transform or camera_trans and camera_rot must be defined"
        vertices_camera = kal.render.camera.rotate_translate_points(vertices, camera_rot, camera_trans)
    else:
        assert camera_trans is None and camera_rot is None, \
            "camera_trans and camera_rot must be None when camera_transform is defined"
        padded_vertices = torch.nn.functional.pad(vertices, (0, 1), mode='constant', value=1.)
        vertices_camera = (padded_vertices @ camera_transform)

    vertices_image = intrinsics.transform(vertices_camera)[:, :, :2]

    face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(vertices_camera, faces)
    face_vertices_image = kal.ops.mesh.index_vertices_by_faces(vertices_image, faces)
    face_normals = kal.ops.mesh.face_normals(face_vertices_camera, unit=True)
    return face_vertices_camera, face_vertices_image, face_normals


class Renderer:
    def __init__(self, render_cfg, mesh_face_num, device):
        self.device = device
        project_mode = render_cfg.projection_mode
        fov_para = render_cfg.fov_para
        self.render_angle_thres = render_cfg.render_angle_thres
        self.calcu_uncolored_mode = render_cfg.calcu_uncolored_mode
        self.dims = (render_cfg.grid_size, render_cfg.grid_size)
        self.look_at_height = render_cfg.look_at_height
        self.interpolation_mode = render_cfg.texture_interpolation_mode
        assert self.interpolation_mode in ['nearest', 'bilinear', 'bicubic'], \
            f'no interpolation mode: {self.interpolation_mode}'
        assert self.calcu_uncolored_mode in ['WarpGrid', 'FACE_ID', 'DIFF'], \
            f'no uncolored mask caculation mode: {self.calcu_uncolored_mode}'
        assert project_mode in ['Pinhole', 'Orthographic'], \
            f'no projecttion mode: {project_mode}'

        if project_mode == "Pinhole":
            self.intrinsics = kal.render.camera.PinholeIntrinsics.from_fov(width=1200, height=1200, fov=fov_para, device=device)
        elif project_mode == "Orthographic":
            self.intrinsics = kal.render.camera.OrthographicIntrinsics.from_frustum(width=1200, height=1200, near=-800,
                                                                                    far=800, fov_distance=fov_para, device=device)
        self.mesh_face_num = mesh_face_num + 1
        self.seen_faces = torch.zeros(1, self.mesh_face_num, 1, device=self.device)

    def clear_seen_faces(self):
        self.seen_faces = torch.zeros(1, self.mesh_face_num, 1, device=self.device)

    def get_camera_from_view(self, theta, phi, radius):
        x = radius * torch.sin(theta) * torch.sin(phi)
        y = radius * torch.cos(theta)
        z = radius * torch.sin(theta) * torch.cos(phi)

        pos = torch.tensor([x, y, z]).unsqueeze(0)  
        look_at = torch.zeros_like(pos)  
        look_at[:, 1] = self.look_at_height  
        direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)  
        camera_transform = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
        return camera_transform

    def normalize_depth(self, depth_map):
        assert depth_map.max() <= 0.0, 'depth map should be negative'  # 相机视野中不存在物体, 就会报错
        object_mask = depth_map != 0
        min_val = 0.5
        depth_map[object_mask] = ((1 - min_val) * (depth_map[object_mask] - depth_map[object_mask].min()) / (
                depth_map[object_mask].max() - depth_map[object_mask].min())) + min_val
        return depth_map

    def UV_pos_render(self, verts, faces, uv_face_attr, texture_dims,):
        """
        :param verts: (V, 3)
        :param faces: (F, 3)
        :param uv_face_attr: shape (1, F, 3, 2), range [0, 1]
        :param theta:
        :param phi:
        :param radius:
        :param view_target:
        :param texture_dims:
        :return:
        """
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        mesh_out_of_range = False
        if x.min() < -1 or x.max() > 1 or y.min() < -1 or y.max() > 1 or z.min() < -1 or z.max() > 1:
            mesh_out_of_range = True

        face_vertices_world = kal.ops.mesh.index_vertices_by_faces(verts.unsqueeze(0), faces)
        face_vertices_z = torch.zeros_like(face_vertices_world[:, :, :, -1], device=self.device)
        uv_position, face_idx = kal.render.mesh.rasterize(texture_dims[0], texture_dims[1], face_vertices_z, uv_face_attr * 2 - 1,
                                                face_features=face_vertices_world, )
        uv_position = torch.clamp(uv_position, -1, 1)

        uv_position = uv_position / 2 + 0.5
        uv_position[face_idx == -1] = 0
        return uv_position


    def forward_texturing_render(self, verts, faces, uv_face_attr, theta, phi, radius,
                                 view_target, uncolored_mask, texture_dims,):
        """
        :param verts: (V, 3)
        :param faces: (F, 3)
        :param uv_face_attr: shape (1, F, 3, 2), range [0, 1]
        :param theta:
        :param phi:
        :param radius:
        :param view_target:
        :param texture_dims:
        :return:
        """
        camera_transform = self.get_camera_from_view(torch.tensor(theta), torch.tensor(phi), radius, ).to(self.device)
        face_vertices_camera, face_vertices_image, face_normals = prepare_vertices(
            verts.to(self.device), faces.to(self.device), intrinsics=self.intrinsics,
            camera_transform=camera_transform)
        face_vertices_z = face_vertices_camera[:, :, :, -1]
        _, face_idx = kal.render.mesh.rasterize(texture_dims[0], texture_dims[1], face_vertices_z, face_vertices_image,
                                                face_features=face_vertices_z.unsqueeze(3),)
        valid_face_idx = list(np.unique(face_idx.data.cpu().numpy()))
        valid_face_idx.remove(-1)
        seen_cons_from_view = torch.zeros(1, self.mesh_face_num - 1, device=self.device)
        seen_cons_from_view[:, valid_face_idx] = 1  

        vertex_normals = trimesh.geometry.mean_vertex_normals(vertex_count=verts.size(0), faces=faces.cpu(),
                                                              face_normals=face_normals[0].cpu(),)  # V,3
        vertex_normals = torch.from_numpy(vertex_normals).unsqueeze(0).float().to(self.device)
        face_vertices_normals = kal.ops.mesh.index_vertices_by_faces(vertex_normals, faces)

        face_vertices_normal_z = face_vertices_normals[:, :, :, 2:3]  # [1, F, 3, n]
        normal_map, _ = kal.render.mesh.rasterize(texture_dims[0], texture_dims[1], face_vertices_z,
                                                  uv_face_attr * 2 - 1,
                                                  face_features=face_vertices_normal_z,
                                                  valid_faces=seen_cons_from_view.to(torch.bool))
        cos_thres = np.cos(self.render_angle_thres / 180 * np.pi)
        normal_map[normal_map < cos_thres] = 0
        valid_face_mask = normal_map.clone()
        valid_face_mask[valid_face_mask > 0] = 1

        uv3d = torch.zeros_like(face_vertices_camera[:, :, :, -1], device=self.device)
        texture_h, texture_w = texture_dims[0], texture_dims[1]
        uv_features_inv, face_idx_inv = kal.render.mesh.rasterize(texture_h, texture_w, uv3d, uv_face_attr * 2 - 1,
                                                                  face_features=face_vertices_image,
                                                                  valid_faces = seen_cons_from_view.to(torch.bool)
                                                                  )
        # mapping
        uv_features_inv = (uv_features_inv + 1) / 2
        cur_texture_map = kal.render.mesh.texture_mapping(uv_features_inv, view_target, mode=self.interpolation_mode)
        cur_texture_map = cur_texture_map * valid_face_mask
        cur_texture_update_area = kal.render.mesh.texture_mapping(uv_features_inv, uncolored_mask.repeat(1,3,1,1), mode=self.interpolation_mode)
        cur_texture_update_area[cur_texture_update_area > 0] = 1
        cur_texture_update_area = cur_texture_update_area * valid_face_mask

        return cur_texture_map.permute(0, 3, 1, 2,), cur_texture_update_area.permute(0, 3, 1, 2,), normal_map.permute(0, 3, 1, 2)


    def render_single_view_texture(self, verts, faces, uv_face_attr, texture_map, theta, phi, radius,
                                   render_cache=None, dims=None, texture_default_color=[0.8, 0.1, 0.8]):
        dims = self.dims if dims is None else dims

        if render_cache is None:
            camera_transform = self.get_camera_from_view(torch.tensor(theta), torch.tensor(phi), radius,).to(self.device)
            face_vertices_camera, face_vertices_image, face_normals = prepare_vertices(
                verts.to(self.device), faces.to(self.device), intrinsics=self.intrinsics,
                camera_transform=camera_transform)
            face_features_rasterized, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                                           face_vertices_image, [face_vertices_camera[:, :, :, -1:], uv_face_attr])
            depth_map = self.normalize_depth(face_features_rasterized[0])  
            uv_features = face_features_rasterized[1].detach()
        else:
            face_normals, uv_features, face_idx, depth_map = render_cache['face_normals'], render_cache['uv_features'],\
                                                             render_cache['face_idx'], render_cache['depth_map']
        mask = (face_idx > -1).float()[..., None]
        image_features = kal.render.mesh.texture_mapping(uv_features, texture_map, mode=self.interpolation_mode)
        uncolored_mask = None

        if render_cache is None:
            if self.calcu_uncolored_mode == "WarpGrid":
                texture_map_diff = (texture_map.detach() - torch.tensor(texture_default_color).view(1, 3, 1, 1).
                                    to(self.device)).abs().sum(axis=1)
                uncolored_texture_map = (texture_map_diff < 0.1).float().unsqueeze(0)

                uncolored_mask = kal.render.mesh.texture_mapping(uv_features, uncolored_texture_map,
                                                                 mode=self.interpolation_mode)
            elif self.calcu_uncolored_mode == "FACE_ID":
                check = (face_idx > -1) & (face_idx < self.mesh_face_num - 1)
                cur_seen_faces = torch.zeros(1, self.mesh_face_num, 1, device=self.device)
                cur_seen_faces[:, face_idx[check].view(-1)] = 1.0
                cur_seen_faces = ((cur_seen_faces - self.seen_faces) > 0).float()
                uncolored_mask = cur_seen_faces[0][face_idx, :]
                self.seen_faces = ((cur_seen_faces + self.seen_faces) > 0).float()

            elif self.calcu_uncolored_mode == "DIFF":
                diff = (image_features.permute(0, 3, 1, 2).clamp(0, 1).detach() - torch.tensor(texture_default_color).
                        view(1, 3, 1, 1).to(self.device)).abs().sum(axis=1)
                uncolored_mask = (diff < 0.1).float().unsqueeze(-1).clamp(0, 1).detach()
            uncolored_mask = (uncolored_mask * mask + 0 * (1 - mask)).permute(0, 3, 1, 2)

        image_features = image_features * mask + 1 * (1 - mask)
        normals_image = face_normals[0][face_idx, :]

        render_cache = {'uv_features': uv_features, 'face_normals': face_normals, 'face_idx': face_idx, 'depth_map':depth_map}
        return image_features.permute(0, 3, 1, 2), depth_map.permute(0, 3, 1, 2), \
               mask.permute(0, 3, 1, 2), uncolored_mask, normals_image.permute(0, 3, 1, 2), render_cache
