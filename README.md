# JGBA-pointcloud-attack
Official code of MM'20 paper &lt;Efficient Joint Gradient Based Attack Against SOR Defense for 3D Point Cloud Classification>

## Requirements
torch >= 1.0
<br>
numpy
<br>
scipy
<br>
sklearn
<br>
tqdm [optional]


## Dataset
We conduct experiments on a 1024-point downsampled version of ModelNet40 dataset, just as other point cloud adversarial attack papers do.

Here is the download link for the dataset:
<br>
[Google Drive](https://drive.google.com/file/d/1CDA67w5LDsjqaNgInNWdvH_efPMH0G90/view?usp=sharing)
<br>
[Baidu Drive](https://pan.baidu.com/s/1KJe2qIbTtbXbBB7VLVFSag) passwd: f9uy
<br>
Location: ./dataset/random1024/whole_data_and_whole_label.pkl

## Model
Four victim classifiers are tested with, including [PointNet](https://github.com/fxia22/pointnet.pytorch), [PointNet++ (SSG)](https://github.com/erikwijmans/Pointnet2_PyTorch), [PointNet++ (MSG)](https://github.com/erikwijmans/Pointnet2_PyTorch), and [DGCNN](https://github.com/WangYueFt/dgcnn).

Remember to build pointnet++ before attacking it: 
```
python setup.py build_ext --inplace
```

If there raise any error when you run the codes about the four models, please try to solve it by yourself before contacting us. Because we just fork the codes from their official repo :mask:

Here are the download links for the checkpoints:

PointNet
[Google Drive](https://drive.google.com/file/d/1wADG0GM7xsSXSAoV1pTPttUA8ZnxLWl8/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1322xEaB9tc2zB9_FzLtiOA) passwd: ouk3
<br>
Location: ./PointNet/pointnet/cls_model_201.pth
<br>

PointNet++ (SSG)
[Google Drive](https://drive.google.com/drive/folders/1wZ4BICRGvRJVUgLanApDidqiPzrhG1U0?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1kA0ZaENlWAfDhLMUdtiIJg) passwd: t2zq
<br>
Location: ./PointNet2_PyTorch/checkpoints_ssg/pointnet2_cls_best.pth.tar
<br>

PointNet++ (MSG)
[Google Drive](https://drive.google.com/drive/folders/1Uh8F8jLOIYFaq_3JQwdU80I_JiUn0nBl?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/19Ce-I09K6sYigtjfYwV14Q) passwd: cdfe
<br>
Location: ./PointNet2_PyTorch/checkpoints_msg/pointnet2_cls_best.pth.tar
<br>

DGCNN
[Google Drive](https://drive.google.com/file/d/1bBrvogBQnAWi-x-soMtAYgra2SA-JtK3/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1QoSAz6wHeaXdBohJE7LEMg) passwd: r0gc
<br>
Location: ./DGCNN/checkpoints/model.t7
<br>

## How to run
untargeted attack:
```
python craft_adv_examples-untargeted.py --adv JGBA --eps 0.1 --n 40 --eps_iter 0.01
```
targeted attack:
```
python craft_adv_examples-targeted.py --adv JGBA --eps 0.1 --n 40 --eps_iter 0.01
```

## Performance
The success rates of our JGBA attack on both untargeted attack and targeted attack are satisfying, because we propose to break the SOR defense directly.
Please refer to the final version of our paper for more experimental results.

![GitHub](https://github.com/machengcheng2016/JGBA-pointcloud-attack/blob/master/fig/untargeted.png "Untargeted Attack Success Rate")

![Github](https://github.com/machengcheng2016/JGBA-pointcloud-attack/blob/master/fig/targeted.png "Targeted Attack Success Rate")

## Citation
If you find this work useful, please consider citing our paper. We provide a BibTeX entry of our paper below:
```
@inproceedings{ma2020efficient,
  title={Efficient Joint Gradient Based Attack Against SOR Defense for 3D Point Cloud Classification},
  author={Ma, Chengcheng and Meng, Weiliang and Wu, Baoyuan and Xu, Shibiao and Zhang, Xiaopeng},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={1819--1827},
  year={2020}
}
```
