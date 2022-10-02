## RenyiCL: Contrastive Learning with Skew Renyi Divergence

### Introduction
This is an official PyTorch implementation of NeurIPS 2022 paper [RényiCL: Contrastive Learning with skew Rényi Divergence](https://arxiv.org/abs/2208.06270).

### Results
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">pretrain<br/>epochs</th>
<th valign="center">linear<br/>acc</th>
<th valign="center">pretrain<br/>files</th>
<th valign="center">linear<br/>files</th>
<th valign="center">eval<br/>logs</th>
<!-- TABLE BODY -->
<tr>
<td align="right">300</td>
<td align="center">76.2</td>
<td align="center"><a href="https://drive.google.com/file/d/1ifcHOPKW9Zyayvb7ZpjW_UOBG8DjZWIb/view?usp=sharing">ckpt</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1HrMf1mhmFrphickMDWmxKOHmzr306LSx/view?usp=sharing">ckpt</a> 
<td align="center"><a href="https://drive.google.com/file/d/1bqN8H4s0JMarVl3rN_MVVIA6Cu97SKYp/view?usp=sharing">txt</a> 
</tr>
</tbody></table>

### Usage: Preparation
Install
- pytorch>=1.9.0
- tensorboard, tensorboardx, pyyaml
- timm=0.4.9
and download the ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). 
Most of our code is based on [MoCo v3](https://github.com/facebookresearch/moco-v3).

The code has been tested with CUDA 11.3, PyTorch 1.11.0 and timm 0.4.9.

### Usage: Self-supervised Pre-Training

For 100 epoch without multi-crops:
```
python main_renyicl.py \
  --ema-cos \
  --crop-min=.2 \
  --dist-url tcp://localhost:10002 \
  --epochs 100 \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --data /data/ImgageNet/ \
  --outdir ../outdir/ \
  --trial renyicl_100ep
```

For 100 epoch with multi-crops:
```
python main_renyicl.py \
  --moco-m-cos --crop-min=.2 \
  --dist-url tcp://localhost:10002 \
  --epochs 100 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --data /data/ImgageNet/ \
  --n_crops 6 \
  --outdir /tmp/ \
  --trial renyicl_100ep_mc 
```

To reproduce our results in main paper:
```
python main_renyicl.py \
  --moco-m-cos --crop-min=.2 \
  --dist-url tcp://localhost:10002 \
  --epochs 300 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --data /data/ImgageNet/ \
  --outdir /tmp/ \
  --trial renyicl_300ep_mc \
  --n_crops 6
```
Then, it will results 76.2% in ImageNet linear evaluation protocol.

To run MoCo v3 with multi-crops:
```
python main_mocov3.py \
  --moco-m-cos --crop-min=.2 \
  --dist-url tcp://localhost:10002 \
  --epochs 100 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --data /data/ImgageNet/ \
  --outdir /tmp/ \
  --trial mocov3_100ep_mc \
  --n_crops 6
```
Then, it will results 73.5% in ImageNet linear evaluation protocol.


### Usage: Linear Classification

We use SGD with batch size 4096 for linear evaluation. 
```
python main_lincls.py \
  --dist-url 'tcp://localhost:10002' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained /tmp/trial/checkpoint_last.pth.tar \
  --data /data/ImageNet \
  --save_dir /tmp/trial/eval/
```

### Citation
```
@article{lee2022r,
  title={R$\backslash$'enyiCL: Contrastive Representation Learning with Skew R$\backslash$'enyi Divergence},
  author={Lee, Kyungmin and Shin, Jinwoo},
  journal={arXiv preprint arXiv:2208.06270},
  year={2022}
}
```