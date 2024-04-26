import os
import cv2
import json
import torch
import trimesh
import kaolin as kal
from loguru import logger


class Mesh:
    def __init__(self, mesh_path, device, target_scale=1.0, mesh_dy=0.0,
                 remove_mesh_part_names=None, remove_unsupported_buffers=None, intermediate_dir=None):
        # from https://github.com/threedle/text2mesh
        self.material_cvt, self.material_num, org_mesh_path, is_convert = None, 1, mesh_path, False
        if not mesh_path.endswith(".obj") and not mesh_path.endswith(".off"):
            if mesh_path.endswith(".gltf"):
                mesh_path = self.preprocess_gltf(mesh_path, remove_mesh_part_names, remove_unsupported_buffers)
            mesh_temp = trimesh.load(mesh_path, force='mesh', process=True, maintain_order=True)
            mesh_path = os.path.splitext(mesh_path)[0] + "_cvt.obj"
            mesh_temp.export(mesh_path)
            merge_texture_path = os.path.join(os.path.dirname(mesh_path), "material_0.png")
            if os.path.exists(merge_texture_path):
                self.material_cvt = cv2.imread(merge_texture_path)
                self.material_num = self.material_cvt.shape[1] // self.material_cvt.shape[0]
            logger.info("Converting current mesh model to obj file with {} material~".format(self.material_num))
            is_convert = True

        if ".obj" in mesh_path:
            try:
                mesh = kal.io.obj.import_mesh(mesh_path, with_normals=True, with_materials=True)
            except:
                mesh = kal.io.obj.import_mesh(mesh_path, with_normals=True, with_materials=False)
        elif ".off" in mesh_path:
            mesh = kal.io.off.import_mesh(mesh_path)
        else:
            raise ValueError(f"{mesh_path} extension not implemented in mesh reader.")

        self.vertices = mesh.vertices.to(device)    
        self.faces = mesh.faces.to(device)          
        try:
            self.vt = mesh.uvs                          
            self.ft = mesh.face_uvs_idx                 
        except AttributeError:
            self.vt = None
            self.ft = None
        self.mesh_path = mesh_path
        self.normalize_mesh(target_scale=target_scale, mesh_dy=mesh_dy)

        if is_convert and intermediate_dir is not None:
            if not os.path.exists(intermediate_dir):
                os.makedirs(intermediate_dir)
            if os.path.exists(os.path.splitext(org_mesh_path)[0] + "_removed.gltf"):
                os.system("mv {} {}".format(os.path.splitext(org_mesh_path)[0] + "_removed.gltf", intermediate_dir))
            if mesh_path.endswith("_cvt.obj"):
                os.system("mv {} {}".format(mesh_path, intermediate_dir))
            os.system("mv {} {}".format(os.path.join(os.path.dirname(mesh_path), "material.mtl"), intermediate_dir))
            if os.path.exists(merge_texture_path):
                os.system("mv {} {}".format(os.path.join(os.path.dirname(mesh_path), "material_0.png"), intermediate_dir))

    def preprocess_gltf(self, mesh_path, remove_mesh_part_names, remove_unsupported_buffers):
        with open(mesh_path, "r") as fr:
            gltf_json = json.load(fr)
            if remove_mesh_part_names is not None:
                temp_primitives = []
                for primitive in gltf_json["meshes"][0]["primitives"]:
                    if_append, material_id = True, primitive["material"]
                    material_name = gltf_json["materials"][material_id]["name"]
                    for remove_mesh_part_name in remove_mesh_part_names:
                        if material_name.find(remove_mesh_part_name) >= 0:
                            if_append = False
                            break
                    if if_append:
                        temp_primitives.append(primitive)
                gltf_json["meshes"][0]["primitives"] = temp_primitives
                logger.info("Deleting mesh with materials named '{}' from gltf model ~".format(remove_mesh_part_names))

            if remove_unsupported_buffers is not None:
                temp_buffers = []
                for buffer in gltf_json["buffers"]:
                    if_append = True
                    for unsupported_buffer in remove_unsupported_buffers:
                        if buffer["uri"].find(unsupported_buffer) >= 0:
                            if_append = False
                            break
                    if if_append:
                        temp_buffers.append(buffer)
                gltf_json["buffers"] = temp_buffers
                logger.info("Deleting unspported buffers within uri {} from gltf model ~".format(remove_unsupported_buffers))
            updated_mesh_path = os.path.splitext(mesh_path)[0] + "_removed.gltf"
            with open(updated_mesh_path, "w") as fw:
                json.dump(gltf_json, fw, indent=4)
        return updated_mesh_path

    def normalize_mesh(self, target_scale=1.0, mesh_dy=0.0):

        verts = self.vertices
        center = verts.mean(dim=0)
        verts = verts - center
        scale = torch.max(torch.norm(verts, p=2, dim=1))   
        verts = verts / scale
        verts *= target_scale    
        verts[:, 1] += mesh_dy   
        self.vertices = verts

