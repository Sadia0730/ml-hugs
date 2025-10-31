import os
import cv2
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

from hugs.models.modules.smpl_layer import SMPL
from hugs.utils.render import overlay_smpl_3dgs

MODEL_DIR = 'data/smpl'


def prepare_dir(output_path, name):
    out_dir = os.path.join(output_path, name)
    os.makedirs(out_dir, exist_ok=True)

    return out_dir


def load_image(path):
    return Image.open(path)


def save_image(img, path):
    mode = 'RGB' if len(img.shape) == 3 else 'L'
    img = Image.fromarray(img, mode)
    img.save(path)


def split_path(file_path):
    file_dir, file_name = os.path.split(file_path)
    file_base_name, file_ext = os.path.splitext(file_name)
    return file_dir, file_base_name, file_ext


def get_mask(subject_dir, img_name):
    msk_path = os.path.join(subject_dir, 'mask',
                            img_name)[:-4] + '.png'

    msk = np.array(load_image(msk_path))
    msk = (msk != 0).astype(np.uint8)

    msk_path = os.path.join(subject_dir, 'mask_cihp',
                            img_name)[:-4] + '.png'
    msk_cihp = np.array(load_image(msk_path))
    msk_cihp = (msk_cihp != 0).astype(np.uint8)

    msk = (msk | msk_cihp).astype(np.uint8)
    msk[msk == 1] = 255

    return msk


def apply_global_tfm_to_camera(E, Rh, Th):
    r""" Get camera extrinsics that considers global transformation.

    Args:
        - E: Array (3, 3)
        - Rh: Array (3, )
        - Th: Array (3, )
        
    Returns:
        - Array (3, 3)
    """

    global_tfms = np.eye(4)  #(4, 4)
    global_rot = cv2.Rodrigues(Rh)[0].T
    global_trans = Th
    global_tfms[:3, :3] = global_rot
    global_tfms[:3, 3] = -global_rot.dot(global_trans)
    return E.dot(np.linalg.inv(global_tfms))


def get_single_view_results(subject, subject_dir, smpl_params_dir, select_view, out_img_dir, out_mask_dir, out_smpl_dir, max_frames, gender='neutral'):
    anno_path = os.path.join(subject_dir, 'annots.npy')
    annots = np.load(anno_path, allow_pickle=True).item()
    
    # load cameras
    cams = annots['cams']
    cam_Ks = np.array(cams['K'])[select_view].astype('float32')
    cam_Rs = np.array(cams['R'])[select_view].astype('float32')
    cam_Ts = np.array(cams['T'])[select_view].astype('float32') / 1000.
    cam_Ds = np.array(cams['D'])[select_view].astype('float32')

    K = cam_Ks     #(3, 3)
    D = cam_Ds[:, 0]
    E = np.eye(4)  #(4, 4)
    cam_T = cam_Ts[:3, 0]
    E[:3, :3] = cam_Rs
    E[:3, 3]= cam_T
    
    # load image paths
    img_path_frames_views = annots['ims']
    img_paths = np.array([
        np.array(multi_view_paths['ims'])[select_view] \
            for multi_view_paths in img_path_frames_views
    ])
    frame_idxs = np.arange(len(img_paths))
    
    if select_view > 0:
        img_paths = img_paths[0::30]
        frame_idxs = frame_idxs[0::30]
        
    prefix = 'train' if select_view == 0 else 'val'
        
    if max_frames > 0:
        img_paths = img_paths[:max_frames]

    # copy config file
    # copyfile(FLAGS.cfg, os.path.join(output_path, 'config.yaml'))

    smpl_model = SMPL(MODEL_DIR, gender=gender)

    cameras = {}
    smpl_params_out = {}
    all_betas = []
    for idx, ipath in enumerate(tqdm(img_paths)):
        out_name = f'{prefix}_{select_view:03d}_{idx:06d}'

        img_path = os.path.join(subject_dir, ipath)
    
        # load image
        img = np.array(load_image(img_path))

        if subject in ['313', '315']:
            _, image_basename, _ = split_path(img_path)
            start = image_basename.find(')_')
            smpl_idx = int(image_basename[start+2: start+6])
        else:
            smpl_idx = frame_idxs[idx]

        # load smpl parameters
        smpl_params = np.load(
            os.path.join(smpl_params_dir, f"{smpl_idx}.npy"),
            allow_pickle=True).item()

        betas = smpl_params['shapes']
        body_pose = smpl_params['poses'][:, 3:]
        global_orient = np.zeros((1, 3))
        transl = np.zeros((1, 3))
        Rh = smpl_params['Rh']
        Th = smpl_params['Th']
        
        all_betas.append(betas)

        n_E = apply_global_tfm_to_camera(E.copy(), Rh[0], Th[0])
        
        # write camera info
        cameras[out_name] = {
                'intrinsics': K,
                'extrinsics': n_E,
                'distortions': D
        }
        
        smpl_params_out[out_name] = {
            'betas': betas,
            'body_pose': body_pose,
            'global_orient': global_orient,
            'transl': transl,
        }

        # load and write mask
        mask = get_mask(subject_dir, ipath)
        save_image(mask, os.path.join(out_mask_dir, out_name+'.png'))

        # write image
        out_image_path = os.path.join(out_img_dir, '{}.png'.format(out_name))
        save_image(img, out_image_path)

        for k, v in smpl_params_out[out_name].items():
            smpl_params_out[out_name][k] = torch.from_numpy(v).float()
        
        for k, v in cameras[out_name].items():
            cameras[out_name][k] = torch.from_numpy(v).float()
        
        if args.render:
            smpl_out = smpl_model(**smpl_params_out[out_name])
            vertices = smpl_out.vertices[0].detach()
            np2th = lambda x: torch.from_numpy(x).float()
            smpl_img, _ = overlay_smpl_3dgs(
                img, 
                verts=vertices, 
                cam_R=np2th(n_E[:3, :3]), 
                cam_t=np2th(n_E[:3, 3]), 
                cam_int=np2th(K),
                faces=smpl_model.faces_tensor,
                device='cuda',
                vertex_color=None,
                get_face_colors=False,
            )
            final_img = np.concatenate([img, smpl_img], axis=0,)
            
            Image.fromarray(final_img).save(f'{out_smpl_dir}/{out_name}.png')
    return smpl_params_out, cameras


def main(args):
    subject = args.subject
    gender = args.gender
    max_frames = args.max_frames

    dataset_dir = 'data/zju_mocap'
    subject_dir = os.path.join(dataset_dir, f"CoreView_{subject}")
    smpl_params_dir = os.path.join(subject_dir, "new_params")

    output_path = f'data/zju_mocap/processed/{subject}'
    
    os.makedirs(output_path, exist_ok=True)
    out_img_dir  = prepare_dir(output_path, 'images')
    out_mask_dir = prepare_dir(output_path, 'masks')
    out_smpl_dir = prepare_dir(output_path, 'smpl_render')
    os.makedirs(out_smpl_dir, exist_ok=True)
    
    smpl_params_out, cameras = {}, {}
    for view in tqdm(range(0, 23), 'Processing views:'):
        smpl_params, cams = get_single_view_results(
            subject, subject_dir, smpl_params_dir, view, 
            out_img_dir, out_mask_dir, out_smpl_dir, max_frames, gender=gender)
        smpl_params_out.update(smpl_params)
        cameras.update(cams)
    
    torch.save(smpl_params_out, os.path.join(output_path, 'smpl_params.pt'))
    torch.save(cameras, os.path.join(output_path, 'cameras.pt'))


# [377, 386, 387, 392, 393, 394]
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, default='377')
    parser.add_argument('--gender', type=str, default='neutral')
    parser.add_argument('--max_frames', type=int, default=-1)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    print(args)
    
    mf = {
        '377': 570, 
        '386': 540, 
        '387': 540, 
        '392': -1, 
        '393': -1, 
        '394': 475,
    }
    if args.subject == 'all':
        for subject in mf.keys():
            args.subject = subject
            args.max_frames = mf[args.subject]
            main(args)
    else:
        args.max_frames = mf[args.subject]
        main(args)