# WSI-QCN
![WSI-QA](./figs/fig.1.png)

<p align="justify">  The pipeline begins with an input whole-slide image (WSI), followed by a quality assessment (QA) stage that identifies and retains high-quality tissue tiles while excluding low-quality regions affected by artifacts, blur, background, or insufficient tissue content. The selected tiles are then processed using one of 12 color normalization methods, including traditional approaches (Ruifrok, Vahadane, Histogram Matching, Mean-Std, Macenko, Reinhard, and Multi-Macenko) and deep learning-based methods (StainGAN, StainNet, Sastaindiff, CycleGAN, and Pix2Pix), to reduce stain variability and improve color consistency. The resulting normalized images provide standardized inputs for a wide range of deep learning-based computational pathology tasks, such as nuclei detection, segmentation, classification, and quantitative analysis. </p>

This pipeline consists of two sequential steps designed to generate high-quality, standardized H&E image tiles for downstream deep learning applications.

### 1. Quality Assessment (QA)
The quality assessment model is first applied [WSI-SmartTiling] (https://github.com/Falah-Jabar-Rahim/Fully-Automatic-Content-Aware-Tiling-Pipeline-for-WSIs) to the whole-slide images (WSIs) to identify and remove low-quality regions (e.g., background, blur, artifacts, or out-of-focus areas). Only high-quality tissue tiles are retained and exported for the next preprocessing stage.

### 2. Color Normalization (ColorNorm)
The retained image tiles are then processed using the color normalization pipeline provided in this repository. This step reduces stain and color variability across H&E images, producing standardized image tiles that can be directly used for downstream tasks such as nuclei detection, segmentation, classification, and quantitative analysis.
