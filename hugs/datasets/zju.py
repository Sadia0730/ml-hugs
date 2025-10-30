import os
import torch
import cv2
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
import trimesh

# from hugs.smplx import SMPL
from hugs.models.modules.smpl_layer import SMPL
from hugs.utils.graphics import get_projection_matrix



class ZJUMoCapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        seq,
        split='train',
        render_mode='human',
        cloth_upper='tshirt',
        cloth_lower='pants',
        cloth_dir='assets/snug',
        dataset_path='data/zju_mocap/processed',
    ):
        dataset_path = f'{dataset_path}/{seq}'
        
        # Cloth directory setup (similar to NeuMan)
        self.cloth_dir = f"{cloth_dir}/{seq}"
        self.cloth_upper = cloth_upper  # Upper garment type
        self.cloth_lower = cloth_lower  # Lower garment type
        print(f"[Dataset] Cloth config: upper='{cloth_upper}', lower='{cloth_lower}', dir='{self.cloth_dir}'")
        
        if split == 'train':
            self.images_list = sorted(glob.glob(f"{dataset_path}/images/train_*.png"))
        elif split == 'val':
            self.images_list = sorted(glob.glob(f"{dataset_path}/images/val_*.png"))
        else:
            self.images_list = sorted(glob.glob(f"{dataset_path}/images/*.png"))
        
        self.num_frames = len(self.images_list)
        self.split = split
        
        self.smpl_params = torch.load(f'{dataset_path}/smpl_params.pt')
        self.cam_params = torch.load(f'{dataset_path}/cameras.pt')
        
        self.init_human_pcd = None
        
        self.mode = render_mode
        
        # Load neutral cloth meshes
        # Neutral upper garment
        neutral_upper_path = f"assets/meshes/{cloth_upper}.obj"
        if os.path.exists(neutral_upper_path):
            upper_mesh = trimesh.load(neutral_upper_path, process=False)
            self.cloth_vertices_shirt = torch.from_numpy(upper_mesh.vertices).float()
            self.cloth_faces_shirt = torch.from_numpy(upper_mesh.faces).long()
            print(f"[Dataset] Loaded neutral upper garment mesh from {neutral_upper_path} ({upper_mesh.vertices.shape[0]} vertices)")
        else:
            print(f"[Dataset] WARNING: Neutral upper garment not found: {neutral_upper_path}")
            self.cloth_vertices_shirt, self.cloth_faces_shirt = None, None
        
        # Neutral lower garment
        neutral_lower_path = f"assets/meshes/{cloth_lower}.obj"
        if os.path.exists(neutral_lower_path):
            lower_mesh = trimesh.load(neutral_lower_path, process=False)
            self.cloth_vertices_pants = torch.from_numpy(lower_mesh.vertices).float()
            self.cloth_faces_pants = torch.from_numpy(lower_mesh.faces).long()
            print(f"[Dataset] Loaded neutral lower garment mesh from {neutral_lower_path} ({lower_mesh.vertices.shape[0]} vertices)")
        else:
            print(f"[Dataset] WARNING: Neutral lower garment not found: {neutral_lower_path}")
            self.cloth_vertices_pants, self.cloth_faces_pants = None, None
        
        # Combine cloth vertices and faces
        cloth_vertices = []
        cloth_faces = []
        face_offset = 0
        
        if self.cloth_vertices_shirt is not None:
            cloth_vertices.append(self.cloth_vertices_shirt)
            cloth_faces.append(self.cloth_faces_shirt)
            face_offset = self.cloth_vertices_shirt.shape[0]
        
        if self.cloth_vertices_pants is not None:
            # shift pants faces so they don't overlap with shirt vertices
            cloth_vertices.append(self.cloth_vertices_pants)
            cloth_faces.append(self.cloth_faces_pants + face_offset)
        
        if len(cloth_vertices) > 0:
            self.cloth_vertices = torch.cat(cloth_vertices, dim=0)
            self.cloth_faces = torch.cat(cloth_faces, dim=0)
        else:
            self.cloth_vertices, self.cloth_faces = None, None

        if split == "animation":
            global_orient = np.array([[np.pi, 0, 0]])
            body_pose = np.zeros((1, 69))
            body_pose[:, 2] = 0.5
            body_pose[:, 5] = -0.5
            transl = np.array([[0., 0., 5]])

            self.betas = np.zeros(10).astype(np.float32)
            self.body_pose = body_pose.astype(np.float32)
            self.global_orient = global_orient.astype(np.float32)
            self.transl = transl.astype(np.float32)
            self.num_frames = 360

            pose_sequence = "data/animation/aist_demo.npz"
            anim = np.load(pose_sequence)
            # self.anim_poses = anim['poses'][:, joints_to_use].astype(np.float32)
            self.anim_poses = anim['poses'][:, :72].astype(np.float32)
        
        self.cached_data = None
        if self.cached_data is None:
            self.load_data_to_cuda()

    def __len__(self):
        if self.split in ['train', 'val']:
            return self.num_frames
        elif self.split == 'animation':
            return self.num_frames + self.anim_poses.shape[0]
    
    def get_single_item(self, i):
        
        # if self.split == 'val':
        #     i = i + self.num_frames - 10
        
        img_path = self.images_list[i]
        msk_path = img_path.replace('images', 'masks')
        img = np.asarray(Image.open(img_path))
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE) / 255
        # msk = np.asarray(Image.open(msk_path)) / 255.
        
        key = img_path.split('/')[-1].split('.')[0]
        
        # get bbox from mask
        rows = np.any(msk, axis=0)
        cols = np.any(msk, axis=1)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        bbox = np.array([xmin, ymin, xmax, ymax])

        img = (img[..., :3] / 255).astype(np.float32)
        msk = msk.astype(np.float32)
        # import ipdb; ipdb.set_trace()
        img = img.transpose(2, 0, 1)
        
        K = self.cam_params[key]['intrinsics']
        w2c = self.cam_params[key]['extrinsics']
        
        width = img.shape[2]
        height = img.shape[1]
        
        fovx = 2 * np.arctan(width / (2 * K[0, 0]))
        fovy = 2 * np.arctan(height / (2 * K[1, 1]))
        zfar = 100.0
        znear = 0.01
        
        world_view_transform = w2c.T
        projection_matrix = get_projection_matrix(znear, zfar, fovx, fovy).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        datum = {
            # NeRF
            "rgb": torch.from_numpy(img).float(),
            "mask": torch.from_numpy(msk).float(),
            "bbox": torch.from_numpy(bbox).float(),

            "fovx": fovx,
            "fovy": fovy,
            "image_height": height,
            "image_width": width,
            "world_view_transform": world_view_transform,
            "full_proj_transform": full_proj_transform,
            "camera_center": camera_center,
            "cam_intrinsics": K,

            "betas": self.smpl_params[key]["betas"][0],
            "global_orient": self.smpl_params[key]["global_orient"][0],
            "body_pose": self.smpl_params[key]["body_pose"][0],
            "transl": self.smpl_params[key]["transl"][0],
            "smpl_scale": torch.tensor(1.0).float(),
        }
        datum["near"] = znear
        datum["far"] = zfar
        
        # Load per-frame cloth meshes
        # Extract frame index from key (e.g., "train_000" -> 0)
        frame_idx = int(key.split('_')[-1])
        
        # Load lower garment (configurable: pants, shorts, skirt, etc.)
        lower_path = os.path.join(self.cloth_dir, f"{frame_idx:04d}_{self.cloth_lower}.obj")
        if os.path.exists(lower_path):
            lower_mesh = trimesh.load(lower_path, process=False)
            lower_vertices = torch.from_numpy(lower_mesh.vertices).float()
            datum["cloth_pants"] = lower_vertices
        
        # Load upper garment (configurable: tshirt, top, shirt, hoodie, etc.)
        upper_path = os.path.join(self.cloth_dir, f"{frame_idx:04d}_{self.cloth_upper}.obj")
        if os.path.exists(upper_path):
            upper_mesh = trimesh.load(upper_path, process=False)
            upper_vertices = torch.from_numpy(upper_mesh.vertices).float()
            datum["cloth_shirt"] = upper_vertices
        
        # Concatenate all garments into one tensor
        cloth_vertices = []
        if "cloth_pants" in datum:
            cloth_vertices.append(datum["cloth_pants"])
        if "cloth_shirt" in datum:
            cloth_vertices.append(datum["cloth_shirt"])
        
        if len(cloth_vertices) > 0:
            # Concatenate all garments into one tensor
            datum["cloth_gt"] = torch.cat(cloth_vertices, dim=0)  # [N, 3]
        else:
            # No cloth GT available - cloth training will use only regularization losses
            pass
        
        return datum
    
    def load_data_to_cuda(self):
        self.cached_data = []
        for i in tqdm(range(self.__len__())):
            if self.split == 'animation':
                datum = self.get_single_item_anim(i)
            else:
                datum = self.get_single_item(i)
            for k, v in datum.items():
                if isinstance(v, torch.Tensor):
                    datum[k] = v.to("cuda")
            self.cached_data.append(datum)
                
    def __getitem__(self, idx):
        if self.cached_data is None:
            if self.split == 'animation':
                return self.get_single_item_anim(idx)
            else:
                return self.get_single_item(idx, is_src=True)
        else:
            return self.cached_data[idx]