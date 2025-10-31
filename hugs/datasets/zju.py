import os
import torch
import cv2
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from hugs.models.modules.smpl_layer import SMPL
from hugs.utils.graphics import get_projection_matrix_center


class ZJUDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        seq,
        split='train',
        render_mode='human',   
    ):
        dataset_path = f'data/zju_mocap/processed/{seq}'
        
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
        projection_matrix = get_projection_matrix_center(znear, zfar, K[0, 0], K[1, 1], K[0, 2], K[1, 2], width, height).transpose(0,1)
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