# VoloGAN

Paper: 

## Abstract: 

We present VoloGAN, an adversarial domain adaptation network that translates synthetic RGB-D
images of a high-quality 3D model of a person, into RGB-D images that could be generated with a
consumer depth sensor. This system is especially useful to generate high amount training data for
single-view 3D reconstruction algorithms replicating the real-world capture conditions, being able to
imitate the style of different sensor types, for the same high-end 3D model database. The network
uses a CycleGAN framework with a U-Net architecture for the generator and a discriminator inspired
by SIV-GAN. We use different optimizers and learning rate schedules to train the generator and the
discriminator. We further construct a loss function that considers image channels individually and,
among other metrics, evaluates the structural similarity. We demonstrate that CycleGANs can be used
to apply adversarial domain adaptation of synthetic 3D data to train a volumetric video generator
model having only few training samples.

## Architecture: 
<img src="https://github.com/sascha-kirch/VoloGAN/blob/master/imgs/vologan.png" width="800" />
Our framework consists of four models: two generators and two discriminators. The generators
translate an RGB-D image from one domain into the respective other domain. The discriminators predict, whether a generated
RGB-D is real or fake. We incorporate four loss terms: adversarial loss, a channel-wise cycle-consistency loss, a channel-wise
structural similarity loss of cycled image pairs and an identity loss.

<details><summary>Show Models</summary>
<p>

Generator             |  Discriminator
:-------------------------|:-------------------------
![](https://github.com/sascha-kirch/VoloGAN/blob/master/imgs/generator_model.png)  | ![](https://github.com/sascha-kirch/VoloGAN/blob/master/imgs/critic_model.png)
Our generator follows an encoder-decoder architecture with multiple connections between encoder and decoder. | The discriminator has three outputs to evaluate weather an input RGB-D image is real or fake: low level evaluation, layout evaluation and content evaluation. We explicitly encourage the disentanglement between layout and content by a two-branch architecture.

</p>
</details>




## Results:

### PCA - Principal Component Analysis
Principal component analysis of the five principal components from 50 samples of each domain. Orange: generated
images of the target domain. Blue: real images of the target domain.

Before Training             |  After Training
:-------------------------:|:-------------------------:
<img src="https://github.com/sascha-kirch/VoloGAN/blob/master/imgs/pca_before.PNG" width="200" />  |  <img src="https://github.com/sascha-kirch/VoloGAN/blob/master/imgs/pca_after.PNG" width="200" />


### 3D Point Clouds
Input RGBD             |  Generated RGBD
:-------------------------:|:-------------------------:
<img src="https://github.com/sascha-kirch/VoloGAN/blob/master/imgs/3d_pointcloud_input.png" width="500" />  |  <img src="https://github.com/sascha-kirch/VoloGAN/blob/master/imgs/3d_pointcloud_generated.png" width="500" />


### Generated RGBD
<img src="https://github.com/sascha-kirch/VoloGAN/blob/master/imgs/multiple_rgb.png" width="500" />
<img src="https://github.com/sascha-kirch/VoloGAN/blob/master/imgs/multiple_depth.png" width="500" />


