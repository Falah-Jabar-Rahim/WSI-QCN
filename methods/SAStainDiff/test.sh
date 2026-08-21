# input_path: your data path
# save_path: your sample result path
# model_path your checkpoint path
# timestep_step: the initial sampling point (start noise level)
# timestep_respacing: the sampling step size
# use_anysize: Restore Any-size Image sampling strategy
# use_ddim: DDIM sampling strategy

SAMPLE_FLAGS="--input_path ./data/test/ --save_path ./data/sample/ --model_path ./checkpoint/other_to_Aperio.pt --use_ddim True --timestep_step 900 --timestep_respacing ddim5 --use_anysize False"

python test.py $SAMPLE_FLAGS

