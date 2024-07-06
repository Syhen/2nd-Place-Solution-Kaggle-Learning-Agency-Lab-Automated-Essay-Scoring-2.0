"""
@created by: heyao
@created at: 2024-05-26 14:23:59
"""


def differential_learning_rate(model, encoder_lr, decoder_lr, ko_lr=1e-3, weight_decay=0.0, lr_factor=2.6):
    no_decay = [".bias", "LayerNorm.bias", "LayerNorm.weight"]
    name = "backbone"
    if hasattr(model.backbone.encoder, "layer"):
        num_layers = len(model.backbone.encoder.layer)
    elif hasattr(model.backbone.encoder, "layers"):
        num_layers = len(model.backbone.encoder.layers)
    elif hasattr(model.backbone.encoder, "blocks"):
        num_layers = len(model.backbone.encoder.blocks)
    else:
        print(model)
        raise ValueError("cant access layer modules")
    print(f"model {model.__class__.__name__} has {num_layers} encoder layers")
    print(f"model {model.__class__.__name__} has {len(list(model.named_parameters()))} trainable parameter groups.")
    sub_layers = int(num_layers / 3)
    shallow_groups, middle_groups, high_groups, head_groups = [], [], [], []
    shallow_layer_names = [f".{i}." for i in range(sub_layers)]
    middle_layer_names = [f".{i}." for i in range(sub_layers, sub_layers * 2)]
    high_layer_names = [f".{i}." for i in range(sub_layers * 2, sub_layers * 3)]

    for n, p in model.named_parameters():
        _weight_decay = weight_decay
        if any(decay in n for decay in no_decay):
            _weight_decay = 0
        if "embeddings" in n:
            shallow_groups.append({
                "params": p, "lr": encoder_lr / lr_factor / lr_factor, "weight_decay": _weight_decay
            })
        elif name in n and any(i in n for i in shallow_layer_names):
            shallow_groups.append({
                "params": p, "lr": encoder_lr / lr_factor / lr_factor, "weight_decay": _weight_decay
            })
        elif name in n and any(i in n for i in middle_layer_names):
            middle_groups.append({
                "params": p, "lr": encoder_lr / lr_factor, "weight_decay": _weight_decay
            })
        elif name in n and any(i in n for i in high_layer_names):
            high_groups.append({
                "params": p, "lr": encoder_lr, "weight_decay": _weight_decay
            })
        elif name not in n:
            if "head_ko" in name:
                head_groups.append({
                    "params": p, "lr": ko_lr, "weight_decay": _weight_decay
                })
            else:
                head_groups.append({
                    "params": p, "lr": decoder_lr, "weight_decay": _weight_decay
                })
        else:
            shallow_groups.append({
                "params": p, "lr": encoder_lr / lr_factor / lr_factor, "weight_decay": _weight_decay
            })

    optimizer_parameters = shallow_groups + middle_groups + high_groups + head_groups
    ensure_all_training = len(list(model.named_parameters())) == len(optimizer_parameters)
    assert ensure_all_training, "some param not training."
    return optimizer_parameters
