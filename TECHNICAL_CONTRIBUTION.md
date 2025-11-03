# HUGS Technical Overview and Contributions

## 1. Method Overview

### 1.1 Core Architecture
HUGS (Human Gaussian Splatting) is a neural rendering framework that models dynamic human subjects using 3D Gaussian Splatting with SMPL-based deformation. The method extends traditional Gaussian Splatting to handle articulated human motion by integrating:

1. **Tri-Plane Feature Representation**: A neural feature volume organized as three orthogonal 2D planes (XY, XZ, YZ) that encode spatial information for each Gaussian point.

2. **Learned Gaussian Attributes**: Each Gaussian point learns:
   - Position (`xyz`) with per-point offsets
   - Rotation (6D rotation representation)
   - Scale (3D anisotropic scaling)
   - Opacity (sigmoid-activated)
   - Spherical Harmonics (SH) coefficients for view-dependent appearance

3. **Modular Decoder Architecture**:
   - **TriPlane Module** (`hugs/models/modules/triplane.py`): Encodes spatial features via grid sampling from three orthogonal planes
   - **Geometry Decoder** (`hugs/models/modules/decoders.py`): Predicts position offsets, rotations, and scales from triplane features
   - **Appearance Decoder**: Predicts opacity and SH coefficients for color
   - **Deformation Decoder**: Predicts Linear Blend Skinning (LBS) weights and pose-dependent blendshapes for SMPL deformation

4. **SMPL Integration**: 
   - Uses SMPL body model parameters (betas, global_orient, body_pose, transl) for pose-driven deformation
   - Applies Linear Blend Skinning (LBS) to transform canonical-space Gaussians to posed space
   - Computes LBS weights via K-nearest neighbor search between Gaussian positions and SMPL template vertices

### 1.2 Training Pipeline

**Initialization**:
- Gaussians initialized from SMPL mesh vertices or point cloud
- Initial scales computed from edge lengths or spatial proximity
- SH coefficients initialized to neutral gray

**Forward Pass**:
```
Canonical Space → TriPlane Features → Decoders → Gaussian Attributes
                    ↓
SMPL Pose → LBS Deformation → Posed Space → Rendering
```

**Loss Functions**:
- **L1 Loss**: Pixel-wise reconstruction error
- **SSIM Loss**: Structural similarity for perceptual quality
- **LPIPS Loss**: Learned perceptual image patch similarity
- **LBS Regularization**: Encourages predicted LBS weights to match SMPL ground truth
- **Densification**: Periodic Gaussian splitting/cloning based on view-space gradients

**Densification Strategy**:
- Every N iterations (configurable, default 600), Gaussians with high view-space gradients are split or cloned
- Splitting: Large Gaussians split into smaller ones based on scale
- Cloning: Duplicates Gaussians with high gradient magnitudes
- Pruning: Removes Gaussians with low opacity

### 1.3 Rendering
- Uses differentiable Gaussian rasterization (adapted from 3DGS)
- Renders 3D Gaussians to 2D images via alpha blending
- Supports separate rendering for human body and static scene
- Compositing: Human rendered with background color, scene rendered separately, then composited

---

## 2. Problem Statement

### 2.1 Limitations of Base HUGS
The original HUGS framework focuses exclusively on human body modeling and does not account for clothing as a separate, deformable entity. This limitation leads to:

1. **Inability to model clothing independently**: Garments are baked into body geometry, preventing:
   - Realistic cloth physics simulation
   - Separate clothing manipulation
   - Fine-grained cloth detail preservation
   - Accurate cloth-body interaction modeling

2. **Limited generalization**: Cannot handle:
   - Different clothing types without retraining
   - Per-frame cloth deformation from physics simulators (e.g., SNUG)
   - Cloth-specific losses (simulation, ARAP, mask-based)

3. **Dataset compatibility**: Only supports NeuMan dataset format, lacking:
   - ZJU-MoCap dataset integration
   - Configurable dataset paths and cloth directories
   - Per-frame cloth mesh loading from external sources

4. **Rendering limitations**: No support for:
   - Multi-layer rendering (body + cloth + scene)
   - Depth-aware cloth compositing
   - Cloth visibility matte computation

### 2.2 Challenges in Cloth Integration
1. **Coordinate space consistency**: Cloth meshes must align with body Gaussians in the same canonical space
2. **LBS weight computation**: Cloth vertices require their own LBS weight predictions, distinct from body topology
3. **Separate rendering passes**: Cloth must be rendered with proper depth ordering relative to body and scene
4. **Loss design**: Requires physics-based losses (simulation, ARAP) and mask-based supervision
5. **Gaussian initialization**: Cloth mesh must be converted to Gaussian representation with appropriate scales and rotations

---

## 3. Technical Contributions

### 3.1 Cloth-Aware Gaussian Modeling

#### 3.1.1 Cloth Gaussian Initialization (`hugs/models/hugs_trimlp.py:128-208`)

**Implementation**: `initialize_cloth()` method

**Key Features**:
- Loads neutral cloth meshes (shirt, pants) from OBJ files
- Converts mesh vertices to Gaussian representation:
  - Position: Cloth vertices as Gaussian centers
  - Scale: Computed from mesh edge lengths via `torch.log(torch.max(edge_len))`
  - Rotation: Aligned to mesh vertex normals using `torch_rotation_matrix_from_vectors()`
  - Opacity: Initialized to 0.1 for gradual appearance
  - SH colors: Neutral gray initialization
- Stores cloth topology (edges, faces) for ARAP loss computation
- Precomputes cloth LBS weight table for consistent weight assignment

**Technical Detail**:
```python
# Scale computation from edge lengths
for v in range(verts_world.shape[0]):
    selected_edges = torch.any(edges == v, dim=-1)
    if selected_edges.sum() > 0:
        e = edges[selected_edges]
        edge_len = torch.norm(verts_world[e[:, 0]] - verts_world[e[:, 1]], dim=-1)
        scales[v] = torch.log(torch.max(edge_len)) * init_scale_multiplier
```

**Storage Structure**:
- `self.cloth_gaussians`: Dictionary containing cloth Gaussian parameters (`xyz`, `scales`, `rot6d_canon`, `shs`, `opacity`, `edges`, `faces`)
- Separate gradient accumulators for cloth: `cloth_xyz_gradient_accum`, `cloth_denom`, `cloth_max_radii2D`

#### 3.1.2 Cloth Forward Pass (`hugs/models/hugs_trimlp.py:707-886`)

**Implementation**: `forward_cloth()` method

**Architecture**: Mirrors `forward_body()` with identical pipeline:
1. **Feature Extraction**: TriPlane queries at cloth vertex positions
2. **Attribute Prediction**: Geometry/appearance decoders predict offsets, rotations, scales, opacity, SH
3. **SMPL Deformation**: Uses same LBS deformation as body Gaussians
4. **Transform Application**: Applies SMPL scale, translation, and external transforms

**Key Differences**:
- Uses `self.cloth_gaussians["xyz"]` as base positions (instead of body `self._xyz`)
- Uses precomputed `self.cloth_lbs_weights_table` for GT LBS weights
- Queries cloth template (`self.cloth_tpose_template`) instead of body template

**LBS Weight Computation**:
```python
_, gt_lbs_weights = smpl_lbsweight_top_k(
    lbs_weights=self.cloth_lbs_weights_table,     # Precomputed cloth weights
    points=gs_xyz.unsqueeze(0),                   # Current cloth positions
    template_points=self.cloth_tpose_template.unsqueeze(0),  # Original template
    K=6,
)
```

#### 3.1.3 Unified Forward Pass (`hugs/models/hugs_trimlp.py:679-705`)

**Implementation**: Modified `forward()` method to return both body and cloth outputs:

```python
def forward(self, ...):
    body_out = self.forward_body(...)
    cloth_out = None
    if self.cloth_gaussians is not None:
        cloth_out = self.forward_cloth(...)
    return {"body": body_out, "cloth": cloth_out} if cloth_out else body_out
```

**Design Choice**: Returns dictionary to maintain backward compatibility while enabling cloth-aware training.

---

### 3.2 Cloth-Aware Rendering (`hugs/renderer/gs_renderer.py`)

#### 3.2.1 Multi-Pass Rendering Pipeline

**Implementation**: Enhanced `render_human_scene()` function

**Rendering Steps**:
1. **Base Render**: Render body + scene Gaussians together (depth blending)
2. **Cloth Color Pass**: Render cloth Gaussians with SH coefficients (`_render_colors_only()`)
3. **Cloth Visibility Matte**: Compute depth-aware visibility of cloth vs. body+scene (`_render_visibility_matte()`)
4. **Compositing**: Blend cloth over base: `final = cloth_rgb * cloth_vis + base_rgb * (1 - cloth_vis)`

**Technical Detail**:
```python
# Depth-aware cloth compositing
cloth_vis = _render_visibility_matte(
    cloth=cloth_gs_out, 
    blockers={body+scene Gaussians},
    ...
)
comp = cloth_rgb * cloth_vis + base_rgb * (1.0 - cloth_vis)
```

**Benefits**:
- Preserves depth ordering: cloth occludes body/scene where closer
- Allows transparent cloth regions (visibility < 1.0)
- Maintains render package with separate body/scene/cloth visibility filters for densification

#### 3.2.2 Cloth-Specific Rendering Functions

**`_render_colors_only()`**: 
- Renders cloth Gaussians with SH-based colors
- Returns RGB image and rendering info (visibility filter, radii, viewspace points)

**`_render_visibility_matte()`**:
- Renders cloth opacity vs. blocker Gaussians (body+scene)
- Uses depth testing to determine cloth visibility
- Returns H×W×1 visibility matte (0=occluded, 1=visible)

---

### 3.3 Cloth-Specific Loss Functions (`hugs/losses/loss.py`)

#### 3.3.1 Simulation Loss (`l_cloth_sim_w`)

**Purpose**: Enforces per-frame cloth mesh alignment with ground truth from physics simulator (SNUG)

**Implementation**: `simulation_loss()` from `losses/utils.py`
```python
loss_cloth_sim = F.mse_loss(cloth_pred, cloth_gt)
```

**Where**:
- `cloth_pred`: Deformed cloth Gaussians from `forward_cloth()` → `cloth_gs_out["xyz"]`
- `cloth_gt`: Ground truth cloth mesh vertices from dataloader → `data["cloth_gt"]`

#### 3.3.2 ARAP Loss (`l_cloth_arap_w`)

**Purpose**: Preserves local cloth shape during deformation using As-Rigid-As-Possible energy

**Implementation**: `arap_loss()` from `losses/utils.py`
- Computes edge length preservation: `||deformed_edge - original_edge||²`
- Encourages locally rigid transformations while allowing global deformation

**Usage**: Requires cloth edges from initialization: `human_gs_init_values["cloth_edges"]`

#### 3.3.3 Mask Loss (`l_cloth_mask_w`)

**Purpose**: Ensures cloth renders only in masked regions (where clothing should appear)

**Implementation**: `mask_loss()` from `losses/utils.py`
- Computes binary cross-entropy between rendered cloth mask and ground truth mask
- Prevents cloth from appearing outside garment regions

#### 3.3.4 Cloth LBS Regularization (`l_lbs_w`)

**Purpose**: Encourages predicted cloth LBS weights to match precomputed ground truth weights

**Implementation**: Similar to body LBS loss
```python
loss_cloth_lbs = F.mse_loss(
    cloth_gs_out["lbs_weights"], 
    cloth_gs_out["gt_lbs_weights"].detach()
)
```

**Key Feature**: Uses separate cloth LBS weight table, not body weights (due to different topologies)

#### 3.3.5 Opacity Entropy Regularization (`l_opacity_entropy_w`)

**Purpose**: Encourages Gaussians to be fully opaque (1.0) or transparent (0.0), reducing "milky" mid-opacity artifacts

**Implementation**:
```python
opacity = cloth_gs_out['opacity'].clamp(1e-6, 1-1e-6)
entropy = -(opacity * torch.log(opacity) + (1-opacity) * torch.log(1-opacity))
loss = entropy.mean()
```

#### 3.3.6 Total Variation Loss (`l_tv_w`)

**Purpose**: Penalizes high-frequency noise in rendered images, encouraging smooth rendering

**Implementation**: `total_variation_loss()` from `losses/utils.py`
- Computes pixel-wise differences between neighboring pixels
- Reduces rendering artifacts and improves visual quality

---

### 3.4 Dataset Integration

#### 3.4.1 ZJU-MoCap Dataset Support (`hugs/datasets/zju.py`)

**Implementation**: New `ZJUMoCapDataset` class

**Key Features**:
1. **Cloth Mesh Loading**:
   - Loads neutral cloth meshes at initialization (shirt, pants from `assets/meshes/`)
   - Loads per-frame cloth meshes during `get_single_item()` from `assets/snug/{seq}/`
   - Combines shirt + pants into single cloth mesh with face index remapping

2. **Configurable Paths**:
   - `dataset_path`: Base path for ZJU data (default: `data/zju_mocap/processed`)
   - `cloth_dir`: Base path for cloth meshes (default: `assets/snug`)
   - `cloth_upper`, `cloth_lower`: Garment type names (default: `tshirt`, `pants`)

3. **Camera Projection**:
   - Uses `get_projection_matrix_center()` for accurate projection matrix from intrinsics
   - Properly handles camera parameters (K matrix, width, height, znear, zfar)

4. **Data Structure**:
   ```python
   datum = {
       "rgb": image,
       "mask": mask,
       "smpl_params": {betas, global_orient, body_pose, transl},
       "cameras": {K, R, T, ...},
       "cloth_gt": per_frame_cloth_vertices,  # Combined shirt + pants
       ...
   }
   ```

#### 3.4.2 NeuMan Dataset Enhancement (`hugs/datasets/neuman.py`)

**Changes**:
- Added `cloth_dir` and `dataset_path` parameters for configurability
- Maintains backward compatibility with default `NEUMAN_PATH`

#### 3.4.3 Trainer Integration (`hugs/trainer/gs_trainer.py`)

**ZJU Dataset Loading**:
```python
elif cfg.dataset.name == 'zju':
    dataset = ZJUDataset(
        cfg.dataset.seq,
        split='train',
        render_mode=cfg.mode,
        cloth_upper=getattr(cfg.dataset, 'cloth_upper', 'tshirt'),
        cloth_lower=getattr(cfg.dataset, 'cloth_lower', 'pants'),
        cloth_dir=getattr(cfg.dataset, 'cloth_dir', 'assets/snug'),
        dataset_path=getattr(cfg.dataset, 'dataset_path', 'data/zju_mocap/processed'),
    )
```

**Cloth Initialization in Trainer**:
- Checks if cloth vertices exist in first training sample
- Calls `human_gs.initialize_cloth(cloth_vertices, cloth_faces)` if available
- Sets up cloth-specific optimizers (position, opacity, scaling, rotation, features)

---

### 3.5 Cloth Densification (`hugs/models/hugs_trimlp.py`)

**Implementation**: Cloth-specific densification methods mirroring body densification

**Key Methods**:
- `cloth_densify_and_split()`: Splits large cloth Gaussians
- `cloth_densify_and_clone()`: Clones high-gradient cloth Gaussians
- `cloth_prune_points()`: Removes low-opacity cloth Gaussians
- `_ensure_cloth_group_exists()`: Creates optimizer parameter groups for cloth

**Integration**: Called during training at densification intervals (same as body), ensuring cloth Gaussians adapt to reconstruction needs.

---

### 3.6 Configuration and YAML Support

**Created YAML Files**:
- `cfg_files/release/zju/hugs_human.yaml`: ZJU human-only training
- `cfg_files/release/zju/hugs_human_scene.yaml`: ZJU human+scene training
- `cfg_files/release/zju/hugs_scene.yaml`: ZJU scene-only training

**Enhanced Existing Files**:
- `cfg_files/release/neuman/hugs_*.yaml`: Added `dataset_path` and `cloth_dir` parameters

**Key Configuration Parameters**:
```yaml
dataset:
  name: zju
  dataset_path: "data/zju_mocap/processed"
  cloth_upper: "top"
  cloth_lower: "pants"
  cloth_dir: "assets/snug"

human:
  loss:
    cloth_sim_w: 1.0      # Simulation loss weight
    cloth_arap_w: 0.5     # ARAP loss weight
    cloth_mask_w: 1.0     # Mask loss weight
    opacity_entropy_w: 0.0
    tv_w: 0.01
```

---

## 4. Technical Highlights

### 4.1 Architecture Decisions

1. **Unified TriPlane for Body and Cloth**: Both body and cloth Gaussians use the same triplane feature volume, ensuring spatial consistency and shared learning.

2. **Separate Gaussian Storage**: Cloth Gaussians stored in `self.cloth_gaussians` dictionary, allowing independent optimization and densification.

3. **LBS Weight Precomputation**: Cloth LBS weights precomputed from template mesh, avoiding runtime KNN search errors and ensuring topological consistency.

4. **Multi-Pass Rendering**: Three-pass rendering (base, cloth color, cloth visibility) enables proper depth compositing without modifying core rasterizer.

5. **Physics-Informed Losses**: Integration of simulation loss (ground truth alignment) and ARAP loss (local shape preservation) bridges neural rendering with physics simulation.

### 4.2 Numerical Stability

- **LBS Weight Validation**: Warnings logged when LBS weights don't sum to 1.0 (indicates NaN/invalid positions)
- **Gradient Accumulation**: Separate gradient accumulators for cloth prevent interference with body optimization
- **Scale Multiplier**: Per-Gaussian scaling multiplier allows independent scale optimization

### 4.3 Extensibility

- **Modular Design**: Cloth integration doesn't modify core body rendering/deformation logic
- **Optional Cloth**: System degrades gracefully when cloth meshes unavailable (original HUGS behavior preserved)
- **Configurable Losses**: All cloth loss weights configurable via YAML (can be set to 0 to disable)

---

## 5. Files Modified/Created

### Core Model Files:
- `hugs/models/hugs_trimlp.py`: Added cloth initialization, forward pass, densification
- `hugs/models/modules/decoders.py`: (No changes - reused for cloth)

### Renderer Files:
- `hugs/renderer/gs_renderer.py`: Added multi-pass cloth rendering, visibility matte computation

### Loss Files:
- `hugs/losses/loss.py`: Added cloth-specific losses (simulation, ARAP, mask, LBS regularization, opacity entropy, TV)
- `hugs/losses/utils.py`: Implementation of simulation_loss, arap_loss, mask_loss, total_variation_loss

### Dataset Files:
- `hugs/datasets/zju.py`: New ZJU dataset with cloth mesh loading
- `hugs/datasets/neuman.py`: Enhanced with configurable paths

### Trainer Files:
- `hugs/trainer/gs_trainer.py`: Added ZJU dataset loading, cloth initialization, cloth densification calls

### Configuration Files:
- `cfg_files/release/zju/hugs_*.yaml`: New ZJU configuration files
- `cfg_files/release/neuman/hugs_*.yaml`: Enhanced with cloth/dataset paths

### Utility Files:
- `hugs/tools/prepare_zjumocap.py`: Data preparation script (SMPL import fix)

---

## 6. Summary

This contribution extends HUGS from a human body-only rendering framework to a cloth-aware system that can:

1. **Model clothing as separate Gaussians** with independent optimization
2. **Integrate physics-based cloth simulation** via SNUG ground truth meshes
3. **Support multiple datasets** (ZJU-MoCap, NeuMan) with configurable paths
4. **Render multi-layer scenes** (body + cloth + scene) with proper depth compositing
5. **Optimize cloth appearance and deformation** using specialized losses (simulation, ARAP, mask, LBS)

The implementation maintains backward compatibility with the original HUGS codebase while adding comprehensive cloth modeling capabilities.

