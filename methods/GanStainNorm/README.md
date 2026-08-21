# Introduction
The codebase was developed using python 3.7.3. All the dependencies can be installed using requirements.txt. This project involves two types of experiments: deep learning or GAN-based stain normalization and traditional stain normalization. GAN-based stain normalization has the following steps: tiling, training, inference, merge, and evaluation. Traditional normalization has two step, i.e., normalization and evaluation. 

GAN-based normalization steps are configuration driven and the config files already contain almost all the settings except for some directories and epoch information. Instructions to execute the stesp are provided in the sections below. 

**Note**: The main repository folder will be referred to as root in the instructions below. 
# Tiling:
Tiling is the process of splitting a full-resolution tissue image into smaller patches for training and inference. Tiling configs for both train and test sets can be found in the folder root/configs/tiling. To tile the full-resolution images use the following commands:

```
python execute.py tile --config root/configs/tiling/config_tiling_train_a.json --multiprocess
```

```
python execute.py tile --config root/configs/tiling/config_tiling_test_a.json --multiprocess
```

# Training:
Training config files could be found in the folder root/configs/train. Each training will generate an experiment folder like **Exp-Pix2Pix-02092022-124502** with the following folders and files:
**checkpoint**: contains epoch's model weights
**config.json**: config with which the training was started
**ds_wsi**: for downsampled whole slide images
**evaluation**: for evaluation file
**inference**: for epoch and sample-wise virtually stained tiles
**logs**: for training logs
**output**: for post-epoch validation sample visualization
**readme.txt**: contains git commit has
**wsi**: for full-resolution tissue whole slide images. 

Use the following python command to run the trainings with different training configs:
```
python execute.py train --exp_root "experiment-directory" --data_root "tiles-directory"/train/512/ --config root/configs/train/config_train_pix2pix_dense_a.json.json

python execute_tile.py train --exp_root output/train/cyclegan_resnet --data_root input/dataset/tissue_a_b_c_tiles/train/512 --config config/train/config_train_cyclegan_resnet_a_b_c.json


```
_experiments-directory_: where you want to save the experiments.
_tiles-directory_: where the tiles are saved. 
# Inference:
Similar to training, inference configuration files in the folder root/configs/inference. Add the epoch numbers, for which you want to run inference, to the "epochs" list in the configs. Inference will save epoch and sample-wise tiles under the inference folder in the specific experiment folder mentioned in the step above. Use the following python command to run the inference with different inference configs:
```
python execute.py inference --exp_path "experiment-directory"/"experiment-name" --data_root "tiles-directory"/test/2048/ --config root/configs/inference/config_inference_pix2pix_dense_a.json
```
_experiments-directory_: directory where experiments are saved.
_experiment-name_: Name of the specific experiment something like: Exp-Pix2Pix-06032022-201735
_tiles-directory_: where the tiles are saved.
# Merge:
The next step is merging the experiment specific tiles. Merge also has six config files in the folder root/configs/merge. Add the epoch numbers, for which you want to merge inference results, to the "epochs" list in the configs. Merge will generate full-resolution WSIs. Use the following python command to run merge tiles with different merge configs:
```
python execute.py merge_tiles --exp_path "experiment-directory"/"experiment-name" --inference_root "experiment-directory"/"experiment-name"/inference --multiprocess --config ./configs/merge/config_merge_pix2pix_dense_a.json
```

_experiments-directory_: directory where experiments are saved.
_experiment-name_: Name of the specific experiment something like: **Exp-Pix2Pix-06032022-201735**
# Evaluation:
There are three different types of evaluation. 
1) root/evaluation/dst_evaluate.py
```
python root/evaluation/dst_evaluate.py --root_img_dir "path-to-original-images" --ref_img_name "name-of-ref-image" --tgt_path "target-path-of-normalized-images" --result_path "path-for-saving-result-excels" --colorspace hsv
```
2) root/evaluation/fid_evaluate.py
```
python root/evaluation/dst_evaluate.py --root_img_dir "path-to-original-images" --ref_img_name "name-of-ref-image" --tgt_path "path-of-normalized-images" --result_path "path-for-saving-result-excels"
```
3) root/evaluation/ssim_evaluate.py
```
python root/evaluation/ssim_evaluate.py --root_img_dir "path-to-original-images" --ref_img_name "name-of-ref-image" --tgt_path "path-of-normalized-images" --result_path "path-for-saving-result-excels" --colorspace hsv
```
# Traditional normalization:
Four different traditional normalization techniques were used in this project.
1) Histogram matching
```
python root/traditional_methods/hist_normalize.py --root_img_dir "path-to-original-images" --ref_img_name "name-of-ref-image" --output_dir "path-for-saving-normalized-images"
```
2) Macenko normalization
```
python root/traditional_methods/macenko_normalize.py --root_img_dir "path-to-original-images" --ref_img_name "name-of-ref-image" --output_dir "path-for-saving-normalized-images"
```
3) Reinhard normalization
```
python root/traditional_methods/reinhard_normalize.py --root_img_dir "path-to-original-images" --ref_img_name "name-of-ref-image" --output_dir "path-for-saving-normalized-images"
```
4) Vahadane matching
```
python root/traditional_methods/vahadane_normalize.py --root_img_dir "path-to-original-images" --ref_img_name "name-of-ref-image" --output_dir "path-for-saving-normalized-images"
```
Evaluation for traditional normalization has the same execution procedure as mentioned in the Evaluation section.
