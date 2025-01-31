# 2nd-Place-Solution Kaggle-Learning-Agency-Lab-Automated-Essay-Scoring-2.0
This is the training code for the 2nd place solution of kaggle competition: https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2

The 2nd place submission notebook is available here: https://www.kaggle.com/code/syhens/aes2-voting?scriptVersionId=184408611

Full solution write up can be found here:
- 2nd Place Overview: https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516582
- 2nd Place Solution: https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516790
- 2nd Place Efficiency Solution: https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/517015

The experiments were run on 1x RTX4090, and its doesn't support multi-gpu for now.

# Note
1. When training, I will always turn on train.use_random_seed. So, it's not full reproducible.
2. The config of stage one and stage two are almost same except the `train.stage`. The both stage use random seed, and I lost my stage one's seed.
3. Always I will run my experiments in bash shell, the format of the shell is:
```shell
export PYTHONPATH=path-to-the-main-folder
python train_xxx.py --config xxxx.yaml train.fold=0
```

**To run the code, follow these steps:**
1. Add your kaggle key to the setup-server.sh, then run to clone this code.
2. Run setup.sh to setup running envs.
3. set environ variable `WANDB_TOKEN` if you want to use wandb.
4. Run `tools/mlm/prepare_inputs.py`, `tools/mlm/run_mlm_large.sh` to prepare `deberta-v3-large-10`
5. Add `model.path=xxx/deberta-v3-large-10` argument to the script and run.

*You can use my trained mlm10 deberta via command `kaggle datasets download -d syhens/aes2-deberta-v3-large-10`, 
It add `\n\n` and `\n` as new tokens, so don't forget to add the two tokens if you want to train your own models.

# Single model training
All the model training is under `training` folder, the run.sh will also be written into this folder.

## exp302 cope
`run.sh`

```shell
export PYTHONPATH=path/2nd-Place-Solution-Kaggle-Learning-Agency-Lab-Automated-Essay-Scoring-2.0

python train_two_stage.py --config "two_stage/config_exp302_cope.yaml" train.fold=0 train.stage=1 train.use_random_seed=false train.use_separated=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_two_stage.py --config "two_stage/config_exp302_cope.yaml" train.fold=0 train.stage=2 train.use_random_seed=false train.use_separated=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_two_stage.py --config "two_stage/config_exp302_cope.yaml" train.fold=1 train.stage=1 train.use_random_seed=false train.use_separated=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_two_stage.py --config "two_stage/config_exp302_cope.yaml" train.fold=1 train.stage=2 train.use_random_seed=false train.use_separated=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_two_stage.py --config "two_stage/config_exp302_cope.yaml" train.fold=2 train.stage=1 train.use_random_seed=false train.use_separated=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_two_stage.py --config "two_stage/config_exp302_cope.yaml" train.fold=2 train.stage=2 train.use_random_seed=false train.use_separated=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_two_stage.py --config "two_stage/config_exp302_cope.yaml" train.fold=3 train.stage=1 train.use_random_seed=false train.use_separated=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_two_stage.py --config "two_stage/config_exp302_cope.yaml" train.fold=3 train.stage=2 train.use_random_seed=false train.use_separated=false model.path="../my_datasets/aes2-deberta-v3-large-10"
```

## exp306b
```shell
export PYTHONPATH=path/2nd-Place-Solution-Kaggle-Learning-Agency-Lab-Automated-Essay-Scoring-2.0

python train_ordinal.py --config "ordinal/config_exp306b.yaml" train.fold=0 train.stage=1 train.use_random_seed=false train.use_cope=false train.hyper_loss_factor=0 model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal.py --config "ordinal/config_exp306b.yaml" train.fold=0 train.stage=2 train.use_random_seed=false train.use_cope=false train.hyper_loss_factor=0 model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal.py --config "ordinal/config_exp306b.yaml" train.fold=1 train.stage=1 train.use_random_seed=false train.use_cope=false train.hyper_loss_factor=0 model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal.py --config "ordinal/config_exp306b.yaml" train.fold=1 train.stage=2 train.use_random_seed=false train.use_cope=false train.hyper_loss_factor=0 model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal.py --config "ordinal/config_exp306b.yaml" train.fold=2 train.stage=1 train.use_random_seed=false train.use_cope=false train.hyper_loss_factor=0 model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal.py --config "ordinal/config_exp306b.yaml" train.fold=2 train.stage=2 train.use_random_seed=false train.use_cope=false train.hyper_loss_factor=0 model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal.py --config "ordinal/config_exp306b.yaml" train.fold=3 train.stage=1 train.use_random_seed=false train.use_cope=false train.hyper_loss_factor=0 model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal.py --config "ordinal/config_exp306b.yaml" train.fold=3 train.stage=2 train.use_random_seed=false train.use_cope=false train.hyper_loss_factor=0 model.path="../my_datasets/aes2-deberta-v3-large-10"
```

## exp306b clean
```shell
export PYTHONPATH=path/2nd-Place-Solution-Kaggle-Learning-Agency-Lab-Automated-Essay-Scoring-2.0

python train_ordinal.py --config "ordinal/config_exp306b_clean.yaml" train.fold=0 train.clean=true train.stage=1 train.use_random_seed=false train.use_cope=false train.hyper_loss_factor=0 model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal.py --config "ordinal/config_exp306b_clean.yaml" train.fold=0 train.clean=true train.stage=2 train.use_random_seed=false train.use_cope=false train.hyper_loss_factor=0 model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal.py --config "ordinal/config_exp306b_clean.yaml" train.fold=1 train.clean=true train.stage=1 train.use_random_seed=false train.use_cope=false train.hyper_loss_factor=0 model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal.py --config "ordinal/config_exp306b_clean.yaml" train.fold=1 train.clean=true train.stage=2 train.use_random_seed=false train.use_cope=false train.hyper_loss_factor=0 model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal.py --config "ordinal/config_exp306b_clean.yaml" train.fold=2 train.clean=true train.stage=1 train.use_random_seed=false train.use_cope=false train.hyper_loss_factor=0 model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal.py --config "ordinal/config_exp306b_clean.yaml" train.fold=2 train.clean=true train.stage=2 train.use_random_seed=false train.use_cope=false train.hyper_loss_factor=0 model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal.py --config "ordinal/config_exp306b_clean.yaml" train.fold=3 train.clean=true train.stage=1 train.use_random_seed=false train.use_cope=false train.hyper_loss_factor=0 model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal.py --config "ordinal/config_exp306b_clean.yaml" train.fold=3 train.clean=true train.stage=2 train.use_random_seed=false train.use_cope=false train.hyper_loss_factor=0 model.path="../my_datasets/aes2-deberta-v3-large-10"
```
The fullfit model can be trained via extra argument: `train.fullfit=true`

## exp320b
```shell
export PYTHONPATH=path/2nd-Place-Solution-Kaggle-Learning-Agency-Lab-Automated-Essay-Scoring-2.0

python train_pet.py --config "pet/config_exp320b.yaml" train.fold=0 train.stage=1 train.use_random_seed=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_pet.py --config "pet/config_exp320b.yaml" train.fold=0 train.stage=2 train.use_random_seed=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_pet.py --config "pet/config_exp320b.yaml" train.fold=1 train.stage=1 train.use_random_seed=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_pet.py --config "pet/config_exp320b.yaml" train.fold=1 train.stage=2 train.use_random_seed=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_pet.py --config "pet/config_exp320b.yaml" train.fold=2 train.stage=1 train.use_random_seed=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_pet.py --config "pet/config_exp320b.yaml" train.fold=2 train.stage=2 train.use_random_seed=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_pet.py --config "pet/config_exp320b.yaml" train.fold=3 train.stage=1 train.use_random_seed=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_pet.py --config "pet/config_exp320b.yaml" train.fold=3 train.stage=2 train.use_random_seed=false model.path="../my_datasets/aes2-deberta-v3-large-10"
```

## exp321
```shell
export PYTHONPATH=path/2nd-Place-Solution-Kaggle-Learning-Agency-Lab-Automated-Essay-Scoring-2.0

python train_ordinal_ms.py --config "ordinal_multi_scale/config_exp321.yaml" train.fold=0 train.stage=1 train.use_random_seed=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal_ms.py --config "ordinal_multi_scale/config_exp321.yaml" train.fold=0 train.stage=2 train.use_random_seed=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal_ms.py --config "ordinal_multi_scale/config_exp321.yaml" train.fold=1 train.stage=1 train.use_random_seed=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal_ms.py --config "ordinal_multi_scale/config_exp321.yaml" train.fold=1 train.stage=2 train.use_random_seed=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal_ms.py --config "ordinal_multi_scale/config_exp321.yaml" train.fold=2 train.stage=1 train.use_random_seed=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal_ms.py --config "ordinal_multi_scale/config_exp321.yaml" train.fold=2 train.stage=2 train.use_random_seed=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal_ms.py --config "ordinal_multi_scale/config_exp321.yaml" train.fold=3 train.stage=1 train.use_random_seed=false model.path="../my_datasets/aes2-deberta-v3-large-10"
python train_ordinal_ms.py --config "ordinal_multi_scale/config_exp321.yaml" train.fold=3 train.stage=2 train.use_random_seed=false model.path="../my_datasets/aes2-deberta-v3-large-10"
```

# Thresholds search and voting
```shell
cd ensemble2
python vote_blend.py
```
You can replace the oofs in the `ensemble2` folder

Note that the oof saved when training will not be used. I only use oof generated on Kaggle, to make inference correlate.
