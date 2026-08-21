# Training models
# please refer to https://github.com/openai/guided-diffusion/

DATA_FLAGS="--data_dir <Your_data_dir> --stain_database_path <Your_database_dir> --random_crop True --random_flip True --nearest_neighbours 5 --sigma_perturb 0.1 --sigma1 0.7 --sigma2 0.7 --shift_value 25 --color_threshold 1000 --stain_threshold 1000000 --gaussian_blur True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 16 --is_train True --checkpoint_path ./checkpoint/temp/"
MODEL_FLAGS="--attention_resolutions 32,16,8 --image_size 256 --num_channels 128 --num_res_blocks 3 --use_fp16 False --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear" 

mpiexec -n 2 python train.py $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
