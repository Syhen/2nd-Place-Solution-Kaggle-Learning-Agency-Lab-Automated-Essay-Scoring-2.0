"""
@created by: heyao
@created at: 2024-02-12 01:36:56
"""


def beautify_print(d, intent=0, ignore_keys=("ema", "mixout", "pl", "id2label", "label2id")):
    for key, val in d.items():
        if key in ignore_keys:
            continue
        if isinstance(val, dict):
            print(f"{' ' * intent}{key}:")
            beautify_print(val, intent=intent + 2)
        else:
            print(f"{' ' * intent}{key}: {val}")


if __name__ == '__main__':
    d = {'id2label': {0: 'O', 1: 'B-EMAIL', 2: 'B-ID_NUM', 3: 'B-NAME_STUDENT', 4: 'B-PHONE_NUM', 5: 'B-STREET_ADDRESS', 6: 'B-URL_PERSONAL', 7: 'B-USERNAME', 8: 'I-ID_NUM', 9: 'I-NAME_STUDENT', 10: 'I-PHONE_NUM', 11: 'I-STREET_ADDRESS', 12: 'I-URL_PERSONAL'}, 'label2id': {'O': 0, 'B-EMAIL': 1, 'B-ID_NUM': 2, 'B-NAME_STUDENT': 3, 'B-PHONE_NUM': 4, 'B-STREET_ADDRESS': 5, 'B-URL_PERSONAL': 6, 'B-USERNAME': 7, 'I-ID_NUM': 8, 'I-NAME_STUDENT': 9, 'I-PHONE_NUM': 10, 'I-STREET_ADDRESS': 11, 'I-URL_PERSONAL': 12}, 'dataset': {'train_file': '../my_datasets/pii-detection-removal-from-educational-data/train.json', 'fold_file': '../my_datasets/pii-detection-removal-from-educational-data/multi_folds4.csv', 'external': '../my_datasets/pii-dd-mistral-generated/mixtral-8x7b-v1.json', 'n_samples_in_train': 5105, 'remove_non_pii_in_train': True, 'remove_non_pii_in_eval': False, 'save_meta_pred': False}, 'model': {'save_path': '../models', 'path': 'microsoft/deberta-v3-large', 'head': '', 'pooling': 'identity', 'max_position_embeddings': 512, 'layer_start': 12, 'msd': False, 'num_reinit_layers': 1, 'differential_lr': {'enable': True, 'lr_factor': 2.6}, 'llrd': {'enable': True, 'value': 0.9}}, 'optim': {'optimizer': {'name': 'adamw8bit', 'lookahead': False, 'lr': 5e-05, 'head_lr': 5e-05, 'betas': [0.9, 0.999], 'eps': 1e-06, 'weight_decay': 0.01}, 'scheduler': {'name': 'linear', 'warmup_percent': 0, 'num_warmup_steps': 0, 'kwargs': {}}}, 'train': {'fold': 0, 'exp_name': 'exp302', 'device': 'cuda', 'num_folds': 4, 'fullfit': False, 'seed': 893, 'use_random_seed': True, 'ensure_data_order': False, 'reinit_weights': True, 'loss': 'ce', 'loss_factor': 4.0, 'accumulate_grad': 1, 'max_length': 2560, 'validate_max_length': 4090, 'batch_size': 8, 'log_steps': 25, 'epochs': 5, 'num_workers': 4, 'max_grad_norm': 1, 'validation_interval': 0.5, 'precision': 'bf16', 'gradient_checkpointing': True, 'freeze_embeddings': False, 'freeze_encoders': 0, 'save_last_checkpoint': False, 'use_wandb': False, 'use_process': False}, 'awp': {'enable': True, 'from_score': 0.9, 'adv_param': 'weight', 'adv_lr': 0.001, 'adv_eps': 0.001, 'start_epoch': 1, 'adv_step': 1}, 'ema': {'tag': 'not implement yet', 'enable': False, 'from_score': 100, 'decay': 0.999}, 'mixout': {'enable': False, 'p': 0.1}, 'pl': {'enable': False, 'path': '???', 'stage': 1, 'prev_model_path': '???'}}
    beautify_print(d)
