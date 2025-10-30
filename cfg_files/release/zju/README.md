# ZJU-MoCap Dataset Configuration

This directory contains configuration files for training HUGS on the ZJU-MoCap dataset.

## Dataset Structure

The ZJU-MoCap dataset should be organized as follows:

```
data/zju_mocap/
├── CoreView_313/
│   ├── Camera_B1/
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   └── ...
│   ├── Camera_B2/
│   ├── ... (up to Camera_B23)
│   ├── mask/
│   │   ├── Camera_B1/
│   │   │   ├── 000000.png
│   │   │   └── ...
│   │   └── ...
│   └── annots.npy  # Contains SMPL params + camera calibration
├── CoreView_315/
├── CoreView_377/
└── ...
```

## Available Subjects

The following subjects are available in ZJU-MoCap:
- `313`, `315`, `377`, `386`, `387`, `390`, `392`, `393`, `394`

## Configuration Files

- **`hugs_human_scene.yaml`**: Joint human and scene training (recommended)
- **`hugs_human.yaml`**: Human-only training
- **`hugs_scene.yaml`**: Scene-only training

## Usage

### 1. Joint Human and Scene Training (Recommended)

Train on subject 313 using camera 0:

```bash
python main.py --cfg_file cfg_files/release/zju/hugs_human_scene.yaml dataset.subject=313 dataset.camera_id=0
```

Train on a different subject (e.g., 377):

```bash
python main.py --cfg_file cfg_files/release/zju/hugs_human_scene.yaml dataset.subject=377
```

### 2. Human-Only Training

```bash
python main.py --cfg_file cfg_files/release/zju/hugs_human.yaml dataset.subject=313
```

### 3. Scene-Only Training

```bash
python main.py --cfg_file cfg_files/release/zju/hugs_scene.yaml dataset.subject=313
```

## Camera Selection

ZJU-MoCap has ~20 cameras per subject. By default, we use camera 0 (Camera_B1) for training.

To use a different camera:

```bash
python main.py --cfg_file cfg_files/release/zju/hugs_human_scene.yaml dataset.subject=313 dataset.camera_id=5
```

**Note:** Camera ID mapping:
- `camera_id=0` → `Camera_B1`
- `camera_id=1` → `Camera_B2`
- etc.

## Key Differences from NeuMan

1. **No Cloth GT**: ZJU-MoCap does not provide per-frame cloth meshes, so cloth training relies only on regularization losses (no `cloth_gt` supervision).

2. **SMPL Format**: ZJU uses a different SMPL parameter format than NeuMan. The dataset loader handles this conversion automatically.

3. **Point Cloud Initialization**: Since ZJU doesn't have COLMAP reconstruction, the initial point cloud is created from SMPL mesh vertices.

4. **Single Camera Training**: While ZJU has multi-view data, this implementation uses a single camera for training (specified by `camera_id`).

## Training Parameters

All training parameters are kept the same as NeuMan for consistency:
- **Iterations**: 20,000
- **LR scheduling**: Same as NeuMan
- **Densification**: Same schedule
- **Loss weights**: Same values

## Expected Data Files

The `annots.npy` file should contain:
- `cams`: Dictionary with keys `K`, `R`, `T`, `D` (camera parameters)
- `poses`: SMPL poses [num_frames, 72]
- `betas`: SMPL shape parameters [10] or [1, 10]
- `trans`: SMPL translation [num_frames, 3]

## Troubleshooting

### "ZJU-MoCap dataset not found"
- Ensure the dataset is located at `data/zju_mocap/CoreView_{subject}/`
- Check that the subject ID is correct (e.g., "313", not "313.0")

### "No images found"
- Verify that images exist in `CoreView_{subject}/Camera_B{camera_id+1}/`
- Check file extension (should be `.jpg`)

### "No masks found"
- Masks are optional; the code will use all-ones masks if not available
- If you have masks, ensure they're in `mask/Camera_B{camera_id+1}/` and are `.png` files

## Example Training Command

Full training on subject 313 with all default settings:

```bash
python main.py --cfg_file cfg_files/release/zju/hugs_human_scene.yaml \
    dataset.subject=313 \
    dataset.camera_id=0 \
    exp_name=zju_313_full
```

## Validation and Evaluation

Validation runs automatically every 1000 iterations. Results are saved in:
```
output/human_scene/zju/{subject}/hugs_trimlp/{exp_name}/
├── val/           # Validation images
├── ckpts/         # Model checkpoints
└── results_train.json  # Metrics (PSNR, SSIM, LPIPS, FID)
```

