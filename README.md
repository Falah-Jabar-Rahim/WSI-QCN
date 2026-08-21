# ColorNorm
A pipeline for running and comparing stain-normalization methods on histology (H&E) images. Point it at a folder of images and a reference/target image, and it runs every configured method over the whole folder — saving each method's output to its own subfolder so results are easy to compare side by side.

12 methods across two families:
- classical methods — Macenko, Reinhard, Multi-Macenko, Vahadane, Ruifrok, histogram matching, and mean/std normalization. Each estimates its stain reference from the target image once per run (cached), then applies it to every source image.
- 5 neural network methods — StainGAN, StainNet, SAStainDiff (diffusion-based), CycleGAN, and DensePix2Pix. These are fully convolutional models with no cross-image statistics, so same-shape images are automatically grouped and pushed through the model together in batches instead of one at a time.
