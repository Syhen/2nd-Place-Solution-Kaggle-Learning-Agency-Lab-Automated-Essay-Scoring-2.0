"""
@created by: heyao
@created at: 2022-09-21 18:48:30
"""
import os

import torch
from omegaconf import OmegaConf
from safetensors import safe_open


def save_yaml(config, to_filename):
    s = OmegaConf.to_yaml(config)
    with open(to_filename, "w") as f:
        f.write(s)
    return True


def save_checkpoints(model, save_file_name, config, half_precision=True):
    if half_precision:
        state_dict = model.half().state_dict()
    else:
        state_dict = model.state_dict()
    if config.train.fullfit:
        save_file_name = f"model_full_seed{config.train.seed}.pth"
    print(f"<<< save state dict to: {os.path.join(config.model.save_path, save_file_name)}")
    torch.save(state_dict, os.path.join(config.model.save_path, save_file_name))
    if half_precision:
        model.float()


def load_backbone_state_dict(torch_file, model_state_dict):
    state_dict = torch.load(torch_file, map_location="cuda")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    new_state_dict = {}
    no_ins = []
    for k, v in state_dict.items():
        if k.startswith("backbone.") and k.split(".", 1)[-1] in model_state_dict:
            new_state_dict[k.split(".", 1)[-1]] = v
        else:
            no_ins.append(k.split(".", 1)[-1])
    if no_ins:
        print(f"<<< unexpected state dict {', '.join(no_ins)} when loading state dict.")
    else:
        print("<<< all keys load successfully.")
    return new_state_dict


def load_lm_head_state(torch_file):
    tensors = {}
    with safe_open(torch_file, framework="pt", device="cpu") as f:
        for k in f.keys():
            if not k.startswith("cls."):
                continue
            tensors[k.split(".", 1)[-1]] = f.get_tensor(k)
    return tensors
