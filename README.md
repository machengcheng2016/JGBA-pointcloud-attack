# JGBA-pointcloud-attack
Official code of paper &lt;Efficient Joint Gradient Based Attack Against SOR Defense for 3D Point Cloud Classification>

## Dataset
We conduct experiments on a downsampled version of ModelNet40 dataset, just as other point cloud adversarial attack papers do.

Here is download link:

[Google Drive](https://drive.google.com/file/d/1CDA67w5LDsjqaNgInNWdvH_efPMH0G90/view?usp=sharing)

[Baidu Drive](https://pan.baidu.com/s/1KJe2qIbTtbXbBB7VLVFSag) passwd: f9uy

## Model
Four victim classifiers are tested with, including [PointNet](https://github.com/fxia22/pointnet.pytorch), [PointNet++ (SSG)](https://github.com/erikwijmans/Pointnet2_PyTorch), [PointNet++ (MSG)](https://github.com/erikwijmans/Pointnet2_PyTorch), and [DGCNN](https://github.com/WangYueFt/dgcnn).

If there raise errors when you run the codes about the four models, please try to solve them by yourself before contacting us. Because we just fork the codes from their official repo.

## Performance
The success rates of our JGBA attack on both untargeted attack and targeted attack are satisfying, because we are the first to break the SOR defense directly.

![GitHub](https://github.com/machengcheng2016/JGBA-pointcloud-attack/blob/master/fig/untargeted.png "Untargeted Attack Success Rate")

![Github](https://github.com/machengcheng2016/JGBA-pointcloud-attack/blob/master/fig/targeted.png "Targeted Attack Success Rate")

Please refer more experimental results to the final version of our paper.
