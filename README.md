# ml-hugs

Setup notes for this repo, including the Python/CUDA environment, dependency install steps, and the Neuman checkpoint layout used by the current training runs.

## 1. Create the environment

This repo is set up around Python 3.8, PyTorch 1.13.1, and CUDA 11.7.

```bash
conda create -n hugs python=3.8 -y
conda activate hugs
git submodule update --init --recursive
```

## 2. Install PyTorch

```bash
conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

## 3. Install Python dependencies

`requirements.txt` contains the general Python dependencies for this repo. `pytorch3d`, `diff-gaussian-rasterization`, and `simple-knn` are installed separately because they are CUDA/PyTorch-build specific.

```bash
pip install -r requirements.txt
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

If you want the helper script instead of running the commands manually:

```bash
bash scripts/conda_setup.sh
```

## 4. Prepare data and pretrained assets

If the archives are already in the repo root, unpack them with:

```bash
unzip -qq neuman_data.zip
unzip -qq hugs_pretrained_models.zip
```

If you need to download them first, use:

```bash
bash scripts/prepare_data_models.sh
```

After unpacking, the main paths should exist:

```text
data/neuman/dataset/citron
data/neuman/dataset/seattle
data/neuman/dataset/bike
data/neuman/dataset/lab
data/neuman/dataset/jogging
data/neuman/dataset/parkinglot
hugs_pretrained_models/
```

## 5. Example Neuman training command

The `human_scene` Neuman config writes runs to:

```text
output/human_scene/neuman/<sequence>/hugs_trimlp/<exp_name>/<timestamp>/
```

Example:

```bash
python main.py --cfg_file cfg_files/release/neuman/hugs_human_scene.yaml dataset.seq=bike exp_name=demo
```

Final checkpoints for a completed run should be:

```text
output/human_scene/neuman/<sequence>/hugs_trimlp/<exp_name>/<timestamp>/ckpt/human_final.pth
output/human_scene/neuman/<sequence>/hugs_trimlp/<exp_name>/<timestamp>/ckpt/scene_final.pth
```

## 6. Check Neuman checkpoints

This repo now includes a simple audit script for the numbered `20k_1` to `20k_5` Neuman runs:

```bash
bash scripts/check_neuman_checkpoints.sh
```

It checks the six expected Neuman sequences:

```text
citron
seattle
bike
lab
jogging
parkinglot
```

and verifies whether each numbered run directory exists and whether both `human_final.pth` and `scene_final.pth` are present.
