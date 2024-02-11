# AC-GAN
Tensorflow implementation of different Generative Adversarial Networks models

This repo consists of 3 implementations: original GAN, [Conditional GAN](https://arxiv.org/abs/1411.1784) and [AC-GAN](https://arxiv.org/abs/1610.09585).

### Usage

```
conda env create -f environment.yaml
conda activate gan_project
cd src/
python run.py --config ../config.yaml
```
config.yaml allows you to change model, as well as other hyperparameters.

### Models and Details 
<img src="model_schema.png" height="270px">

### Results
+ AC-GAN
<table align='center'>
<tr align='center'>
<td> g_lr=0.0003, d_lr=0.0002 </td>
<td> g_lr=0.0003, d_lr=0.0001 </td>
<td> g_lr=0.0002, d_lr=0.0001 </td>
</tr>
<tr>
<td><img src="generated_imgs/ac_gan/lblsmo00_glr0003_dlr0002_g13m_d390k.png" height="270px">
<td><img src="generated_imgs/ac_gan/lblsmo00_glr0003_dlr0001_g13m_d390k.png" height="270px">
<td><img src="generated_imgs/ac_gan/lblsmo00_glr0002_dlr0001_g13m_d390k.png" height="270px">
</tr>
</table>

+ Conditional GAN
<table align='center'>
<tr align='center'>
<td> g_lr=0.0001, d_lr=0.0001 </td>
<td> g_lr=0.0003, d_lr=0.0001 </td>
<td> g_lr=0.0001, d_lr=0.0001, lbl_smooth=0.2 </td>
</tr>
<tr>
<td><img src="generated_imgs/cond_gan/lblsmo00_glr0001_dlr0001_g13m_d390k.png" height="270px">
<td><img src="generated_imgs/cond_gan/lblsmo00_glr0003_dlr0001_g13m_d390k.png" height="270px">
<td><img src="generated_imgs/cond_gan/lblsmo02_glr0001_dlr0001_g13m_d390k.png" height="270px">
</tr>
</table>

+ Simple GAN
<table align='center'>
<tr align='center'>
<td> g_lr=0.0001, d_lr=0.0001 </td>
<td> g_lr=0.0002, d_lr=0.0001 </td>
<td> g_lr=0.0002, d_lr=0.0001, lbl_smooth=0.2 </td>
<td> g_lr=0.0004, d_lr=0.0003 </td>
</tr>
<tr>
<td><img src="generated_imgs/simple_gan/lblsmo00_glr0001_dlr0001_g13m_d390k.png" height="270px">
<td><img src="generated_imgs/simple_gan/lblsmo00_glr0002_dlr0001_g13m_d390k.png" height="270px">
<td><img src="generated_imgs/simple_gan/lblsmo02_glr0002_dlr0001_g13m_d390k.png" height="270px">
<td><img src="generated_imgs/simple_gan/lblsmo00_glr0004_dlr0003_g13m_d390k.png" height="270px">
</tr>
</table>

### Comments
GAN objective defines a two-player minimax game. If either side becomes obviously stronger than other, training fails. Thus, I carefully conducted experimentation on hyperparameters (especially on disc and gen learning rates).

Dataset is relatively simple MNIST, thus different architectures is not employed, but easily possible by changing model.py. Current generator has 1.3M, and discriminator has 390K parameters. Generating is harder than discriminating, hence this choice makes sense.

GAN training is further stabilized by using side information such as image labels. Conditional-GAN augments labels in both generator and discriminator, allowing the model to utilize extra information. This creates better image quality, and conditioned generation.

AC-GAN asks the discriminator predict class label, along with 'real or fake' prediction, which results in high dimensional quality image generation. For MNIST, I do not expect to sense any differences between Conditional-GAN and AC-GAN.

I implemented label augmentation as activation map concatenation. Following this [gan trick](https://github.com/soumith/ganhacks#16-discrete-variables-in-conditional-gans), resulting 7x7 map concatenated to first layer's 128 maps for generator, and 28x28 map concatenated to image as second dimension for discriminator. In addition, AC-GAN, and Conditional-GAN generators are identical.

### TODO
merge {simple|cond|ac}_train.py files into single train.py
