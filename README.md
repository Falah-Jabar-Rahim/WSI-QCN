# ColorNorm
![WSI-QA](./figs/fig.1.png)

A pipeline for running and comparing color normalization methods on histology (H&E) images. 
12 methods across two families:
- 7 classical methods — Macenko, Reinhard, Multi-Macenko, Vahadane, Ruifrok, histogram matching, and mean/std normalization. Each estimates its stain reference from the target image once per run (cached), then applies it to every source image.
- 5 neural network methods — StainGAN, StainNet, SAStainDiff (diffusion-based), CycleGAN, and DensePix2Pix. These are fully convolutional models with no cross-image statistics, so same-shape images are automatically grouped and pushed through the model together in batches instead of one at a time.
