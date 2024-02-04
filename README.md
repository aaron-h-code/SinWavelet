# A Single 3D Shape Wavelet-based Generative Model

## Primary dependencies
- Python 3.9+
- PyTorch 2.0+
- Cuda 11.8
- Trimesh 4.1
- PyWavelets 1.5

Or install dependencies with conda:
```bash
conda env create -f environment.yml
conda activate sinwavelet
```
```angular2html
# NOTE: The versions of the dependencies listed above are only for reference, 
and please check https://pytorch.org/ for pytorch installation command for your CUDA version.
```

## Pretrained models
We provide the pretrained models for Table 2 in our paper [here](https://drive.google.com/file/d/1FXk0_AF6J1BMGLtqvv6csDByqvt0XlqS/view?usp=sharing). 
Downloading all of them needs about 1.7G storage space:
```bash
bash download_models.sh
```

## Data preparation
### Download data
A list of the sources for all shapes used in our paper can be found here: [data/README.md](data/README.md), 
also as listed in Table 1 in our paper. Most of these shapes are free for download.

### Voxelize data
After downloading the shapes, make sure the variable [BINVOX_PATH]() in `voxelization/voxelize.py` is set to the path 
of excetuable binvox. Then run our script
```bash
bash scripts/voxelize.sh
```
Change the bash command arguments as needed, and the processed data will be saved in `.h5` format.

### Preprocessed data
We provide the preprocessed data used in our paper in [\data](\data) folder.

## Training
To train the model on the processed .h5 file, run
```bash
bash scripts/train.sh 
```
Modify the other argument values as needed. By default, the log and model will be saved in `checkpoints/{experiment-tag}`.

## Generating
Before evaluation, we need to generate new shapes by running
```bash
bash scripts/generate.sh
```

## Evaluation
To evaluate using metrics LP-IoU, LP-F-score, SSFID and Div, run
```bash
cd evaluation
bash eval.sh
```
As SSFID relies on a pretrained 3D shape classifiers, please download them 
from [here](https://drive.google.com/file/d/1iIWqq9pRVnVVIE75VKtUlGpc9lrTsEHu/view?usp=sharing) or 
from [here](https://drive.google.com/file/d/1HjnDudrXsNY4CYhIGhH4Q0r3-NBnBaiC/view?usp=sharing), and then put
`Clsshapenet_128.pth` under `evaluation` folder.

## Export meshes
To export the generated shapes as mesh .obj files for visualization, run
```bash
bash vis_export.sh
```

## Acknowledgement
Our code is built upon the repositories [SingleShapeGen](https://github.com/ChrisWu1997/SingleShapeGen), 
[DAG](https://github.com/sutd-visual-computing-group/dag-gans), and [coulomb_gan](https://github.com/bioinf-jku/coulomb_gan). 
We would appreciate their authors.

## Citation
```angular2html
TBD
```


<!---
Image example: wavelet-v6/test_pywt_2D_morph_torch_multiscale.py
Shape example: wavelet-v6/test_pywt_3D_morph_torch.py
--->