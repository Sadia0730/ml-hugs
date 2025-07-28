# HUGS Feature-NonRigid-FiLM-Optimized Branch Flowchart

## 🚀 Main Entry Point
```
main.py
├── Parse CLI arguments (--cfg_file, --cfg_id)
├── Load config file (OmegaConf.load)
├── Expand configs (get_cfg_items) - handles list-valued hyperparameters
├── For each expanded config:
    └── main(cfg)
        ├── safe_state(seed)
        ├── get_logger(cfg) - setup logging directories
        ├── GaussianTrainer(cfg)
        ├── trainer.train() (if not eval)
        ├── trainer.save_ckpt()
        ├── trainer.validate()
        ├── trainer.animate() (if human mode)
        └── trainer.render_canonical() (a_pose, da_pose)
```

## 🏗️ GaussianTrainer Initialization
```
GaussianTrainer.__init__(cfg)
├── Load Datasets:
│   ├── train_dataset = NeumanDataset(seq, 'train', ...)
│   ├── val_dataset = NeumanDataset(seq, 'val', ...)
│   └── anim_dataset = NeumanDataset(seq, 'anim', ...)
├── Initialize LPIPS loss
├── Model Creation:
│   ├── HUGS_TRIMLP (if cfg.human.name == 'hugs_trimlp')
│   │   ├── TriPlane(n_features=32, res=256)
│   │   ├── AppearanceDecoder(n_features=96)
│   │   ├── GeometryDecoder(n_features=96)
│   │   ├── DeformationDecoder(n_features=96)
│   │   ├── FiLM Components (if num_frames > 0):
│   │   │   ├── frame_emb = nn.Embedding(num_frames, 32)
│   │   │   └── film_layer = nn.Linear(32, 2*96)
│   │   └── SMPL Layer
│   └── SceneGS (if scene mode)
├── Setup Optimizers
├── Load Checkpoints (if available)
└── Initialize SMPL Parameters
```

## 🎯 Training Loop
```
train()
├── RandomIndexIterator for data sampling
├── For t_iter in range(num_steps):
│   ├── Update learning rates
│   ├── Sample random data index
│   ├── Get data = train_dataset[rnd_idx]
│   ├── Forward Pass:
│   │   ├── human_gs_out = human_gs.forward(...)
│   │   └── scene_gs_out = scene_gs.forward() (if scene mode)
│   ├── Rendering:
│   │   └── render_pkg = render_human_scene(...)
│   ├── Loss Computation:
│   │   └── loss, loss_dict = loss_fn(...)
│   ├── Backward Pass:
│   │   └── loss.backward()
│   ├── Optimization:
│   │   ├── human_gs.optimizer.step()
│   │   └── scene_gs.optimizer.step() (if scene mode)
│   ├── Densification (if conditions met):
│   │   ├── human_densification(...)
│   │   └── scene_densification(...)
│   ├── Save Progress Images (every 1000 steps)
│   ├── Save Checkpoints (every save_ckpt_interval)
│   ├── Validation (every val_interval)
│   ├── Animation (every anim_interval)
│   └── Canonical Rendering (every 1000 steps)
```

## 🧠 HUGS_TRIMLP Forward Pass
```
HUGS_TRIMLP.forward(...)
├── TriPlane Feature Extraction:
│   ├── tri_feats = self.triplane(self.get_xyz)
│   └── Output: [N, 96] features (3 planes × 32 features)
├── FiLM Modulation (if enabled):
│   ├── frame_idx = dataset_idx
│   ├── cond = self.frame_emb(frame_idx)
│   ├── gamma_beta = self.film_layer(cond)
│   ├── gamma = gamma_beta[:96], beta = gamma_beta[96:]
│   └── tri_feats = tri_feats * (1 + gamma) + beta
├── Decoder Networks:
│   ├── appearance_out = self.appearance_dec(tri_feats)
│   ├── geometry_out = self.geometry_dec(tri_feats)
│   └── deformation_out = self.deformation_dec(tri_feats) (if use_deformer)
├── Gaussian Parameters:
│   ├── xyz_offsets = geometry_out['xyz']
│   ├── gs_rot6d = geometry_out['rotations']
│   ├── gs_scales = geometry_out['scales']
│   ├── gs_xyz = self.get_xyz + xyz_offsets
│   ├── gs_rotmat = rotation_6d_to_matrix(gs_rot6d)
│   ├── gs_rotq = matrix_to_quaternion(gs_rotmat)
│   ├── gs_opacity = appearance_out['opacity']
│   └── gs_shs = appearance_out['shs']
├── Non-Rigid Deformation (if use_deformer):
│   ├── lbs_weights = deformation_out['lbs_weights']
│   ├── posedirs = deformation_out['posedirs']
│   ├── SMPL forward pass
│   ├── LBS transformation: vitruvian → t-pose → posed
│   └── deformed_xyz = LBS(deformed_xyz, lbs_weights, posedirs)
└── Return: {xyz, rotq, scales, opacity, shs, active_sh_degree}
```

## 🎨 TriPlane Module
```
TriPlane.forward(x)
├── Normalize coordinates: x = (x - center) / scale + 0.5
├── Grid Sampling:
│   ├── feat_xy = F.grid_sample(plane_xy, coords[..., [0, 1]])
│   ├── feat_xz = F.grid_sample(plane_xz, coords[..., [0, 2]])
│   └── feat_yz = F.grid_sample(plane_yz, coords[..., [1, 2]])
├── Concatenate features: feat = torch.cat([feat_xy, feat_xz, feat_yz], dim=1)
└── Return: [N, 96] features
```

## 🎭 Rendering Pipeline
```
render_human_scene(...)
├── Combine Gaussian Parameters:
│   ├── human_gs_out: {xyz, rotq, scales, opacity, shs}
│   └── scene_gs_out: {xyz, rotq, scales, opacity, shs}
├── Concatenate for rendering:
│   ├── means3D = torch.cat([human_xyz, scene_xyz])
│   ├── opacity = torch.cat([human_opacity, scene_opacity])
│   ├── scales = torch.cat([human_scales, scene_scales])
│   └── rotations = torch.cat([human_rotq, scene_rotq])
├── Gaussian Rasterization:
│   ├── rasterizer = GaussianRasterizer(settings)
│   └── rendered_image, radii = rasterizer(...)
└── Return: {render, viewspace_points, visibility_filter, radii}
```

## 📊 Loss Computation
```
HumanSceneLoss.forward(...)
├── Extract ground truth and predicted images
├── Apply masks based on render_mode
├── Compute Losses:
│   ├── L1 Loss: l1_loss(pred_img, gt_image)
│   ├── SSIM Loss: 1.0 - ssim(pred_img, gt_image)
│   ├── LPIPS Loss: lpips(pred_img, gt_image)
│   ├── LBS Loss: lbs_regularization (if use_deformer)
│   └── Human Separation Loss (if human_scene mode)
├── Weighted combination: total_loss = Σ(weight_i × loss_i)
└── Return: {total_loss, loss_dict, extras_dict}
```

## 🔄 Densification Process
```
human_densification(...)
├── Update max_radii2D based on visibility
├── Add densification stats
├── Densify and Prune (if conditions met):
│   ├── densify_and_split(grads, threshold)
│   ├── densify_and_clone(grads, threshold)
│   └── prune_points(mask)
└── Update optimizer with new parameters
```

## 📸 Validation & Animation
```
validate()
├── Set models to eval mode
├── For each validation sample:
│   ├── Forward pass with validation data
│   ├── Render image
│   ├── Compute metrics (PSNR, SSIM, LPIPS)
│   └── Save validation images
└── Save metrics to file

animate()
├── Load animation dataset
├── For each animation frame:
│   ├── Forward pass with animation data
│   ├── Render image
│   └── Save animation frame
└── Create video from frames

render_canonical()
├── Generate rotating camera parameters
├── For each camera angle:
│   ├── Forward pass with canonical pose
│   ├── Render image
│   └── Save canonical view
└── Create video from canonical views
```

## 🗂️ Data Flow
```
NeumanDataset
├── Load SMPL parameters (betas, body_pose, global_orient, transl)
├── Load camera parameters (intrinsics, extrinsics)
├── Load RGB images and masks
├── Cache data to CUDA
└── __getitem__: return data sample

Data Sample Structure:
{
    'rgb': [3, H, W],
    'mask': [H, W],
    'betas': [10],
    'body_pose': [23*3],
    'global_orient': [3],
    'transl': [3],
    'smpl_scale': scalar,
    'camera_center': [3],
    'world_view_transform': [4, 4],
    'full_proj_transform': [4, 4],
    'fovx': scalar,
    'fovy': scalar,
    'image_height': int,
    'image_width': int
}
```

## 🔧 Key Optimizations in Feature-NonRigid-FiLM-Optimized

### 1. **FiLM Modulation**
- **Purpose**: Per-frame feature modulation for better temporal consistency
- **Implementation**: 
  - Frame embedding: `nn.Embedding(num_frames, 32)`
  - FiLM layer: `nn.Linear(32, 2*96)` → gamma and beta
  - Modulation: `features * (1 + gamma) + beta`

### 2. **Non-Rigid Deformation**
- **Purpose**: Handle complex human motion and deformation
- **Components**:
  - DeformationDecoder: Predicts LBS weights and pose-dependent deformations
  - LBS transformation: vitruvian pose → t-pose → posed
  - Posedirs: Pose-dependent vertex offsets

### 3. **TriPlane Feature Representation**
- **Purpose**: Efficient 3D feature encoding
- **Structure**: 3 orthogonal planes (XY, XZ, YZ) with 32 features each
- **Output**: 96-dimensional feature vector per point

### 4. **Multi-Decoder Architecture**
- **AppearanceDecoder**: Predicts opacity and spherical harmonics
- **GeometryDecoder**: Predicts position offsets, rotations, and scales
- **DeformationDecoder**: Predicts LBS weights and pose-dependent deformations

## 🎯 Training Modes

### Human Mode (`mode='human'`)
- Train only human model
- Render only human Gaussians
- Background color handling

### Scene Mode (`mode='scene'`)
- Train only scene model
- Render only scene Gaussians
- Point cloud initialization

### Human-Scene Mode (`mode='human_scene'`)
- Train both human and scene models
- Render combined scene
- Separate human/scene loss computation

## 📁 Output Structure
```
output/
├── human/
│   └── neuman/
│       └── {sequence}/
│           └── hugs_trimlp/
│               └── {exp_name}/
│                   └── {timestamp}/
│                       ├── train/          # Training progress images
│                       ├── val/            # Validation images
│                       ├── anim/           # Animation frames
│                       ├── canon/          # Canonical views
│                       ├── meshes/         # Gaussian splat files
│                       ├── ckpt/           # Model checkpoints
│                       └── logs/           # Training logs
```

## 🔄 Configuration System
```
Config Expansion (get_cfg_items)
├── Load base config from YAML
├── Expand list-valued parameters
├── Create Cartesian product of all combinations
├── Generate experiment names
└── Return list of configs to run

Example:
- dataset.seq: ['citron', 'parkinglot']
- human.sh_degree: [0, 1]
→ Creates 4 experiments: citron-sh0, citron-sh1, parkinglot-sh0, parkinglot-sh1
```

This flowchart represents the complete architecture and data flow of your `feature-nonrigid-film-optimized` branch, showing how the FiLM modulation, non-rigid deformation, and TriPlane features work together to create a sophisticated human rendering system. 