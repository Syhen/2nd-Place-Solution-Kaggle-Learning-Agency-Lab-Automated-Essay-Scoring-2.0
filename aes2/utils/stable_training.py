"""
@created by: heyao
@created at: 2022-08-25 00:48:52
"""
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ..nn.optim.lion import Lion

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


def init_weights(module, kaiming=False, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        if kaiming:
            nn.init.kaiming_normal_(module.weight.data)
        else:
            module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def differential_learning_rate(model, encoder_lr, decoder_lr, weight_decay=0.0, lr_factor=2.6):
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


# def differential_learning_rate(model, encoder_lr, decoder_lr, weight_decay=0.0, lr_factor=2.6):
#     """
#     :param model:
#     :param encoder_lr:
#     :param decoder_lr:
#     :param weight_decay:
#     :param lr_factor:
#     :return:
#     """
#     # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
#     no_decay = [".bias", "LayerNorm.bias", "LayerNorm.weight"]
#     name = "backbone"
#     if hasattr(model.backbone.encoder, "layer"):
#         num_layers = len(model.backbone.encoder.layer)
#     elif hasattr(model.backbone.encoder, "layers"):
#         num_layers = len(model.backbone.encoder.layers)
#     elif hasattr(model.backbone.encoder, "blocks"):
#         num_layers = len(model.backbone.encoder.blocks)
#     else:
#         print(model)
#         raise ValueError("")
#     print(f"model {model.__class__.__name__} has {num_layers} encoder layers")
#     print(f"model {model.__class__.__name__} has {len(list(model.named_parameters()))} trainable parameter groups.")
#     sub_layers = int(num_layers / 3)
#     special_terms = ["backbone.encoder.LayerNorm.weight", "backbone.encoder.LayerNorm.bias"]
#     bart_special = [
#         'backbone.shared.weight', 'backbone.encoder.layernorm_embedding.weight',
#         'backbone.encoder.layernorm_embedding.bias', 'backbone.encoder.embed_positions.weight',
#     ]
#     bart_special2 = [
#         'backbone.decoder.embed_positions.weight', 'backbone.decoder.layernorm_embedding.weight',
#         'backbone.decoder.layernorm_embedding.bias'
#     ]
#     optimizer_parameters = [
#         # {'params': [p for n, p in model.named_parameters() if
#         #             'embeddings' in n and not any(nd in n for nd in no_decay)],
#         #  'lr': encoder_lr / lr_factor / lr_factor, 'weight_decay': weight_decay},
#         # {'params': [p for n, p in model.named_parameters() if
#         #             'embeddings' in n and any(nd in n for nd in no_decay)],
#         #  'lr': encoder_lr / lr_factor / lr_factor, 'weight_decay': 0.0},
#         {'params': [p for n, p in model.named_parameters() if n in special_terms],
#          'lr': encoder_lr / lr_factor / lr_factor, 'weight_decay': 0.0},
#         {'params': [p for n, p in model.named_parameters() if name in n and
#                     not any(nd in n for nd in no_decay) and
#                     any(f".{i}." in n for i in range(sub_layers))],
#          'lr': encoder_lr / lr_factor / lr_factor, 'weight_decay': weight_decay},
#         {'params': [p for n, p in model.named_parameters() if name in n and
#                     not any(nd in n for nd in no_decay) and
#                     any(f".{i}." in n for i in range(sub_layers, sub_layers * 2))],
#          'lr': encoder_lr / lr_factor, 'weight_decay': weight_decay},
#         {'params': [p for n, p in model.named_parameters() if name in n and
#                     not any(nd in n for nd in no_decay) and
#                     any(f".{i}." in n for i in range(sub_layers * 2, sub_layers * 3))],
#          'lr': encoder_lr, 'weight_decay': weight_decay},
#
#         {'params': [p for n, p in model.named_parameters() if name in n and
#                     any(nd in n for nd in no_decay) and
#                     any(f".{i}." in n for i in range(sub_layers))],
#          'lr': encoder_lr / lr_factor / lr_factor, 'weight_decay': 0.0},
#         {'params': [p for n, p in model.named_parameters() if name in n and
#                     any(nd in n for nd in no_decay) and
#                     any(f".{i}." in n for i in range(sub_layers, sub_layers * 2))],
#          'lr': encoder_lr / lr_factor, 'weight_decay': 0.0},
#         {'params': [p for n, p in model.named_parameters() if name in n and
#                     any(nd in n for nd in no_decay) and
#                     any(f".{i}." in n for i in range(sub_layers * 2, sub_layers * 3))],
#          'lr': encoder_lr, 'weight_decay': 0.0},
#
#         {'params': [p for n, p in model.named_parameters() if name not in n],
#          'lr': decoder_lr, 'weight_decay': 0.0},
#
#         {'params': [p for n, p in model.named_parameters() if
#                     any(name == n for name in bart_special)],
#          'lr': encoder_lr / lr_factor / lr_factor, 'weight_decay': weight_decay},
#         {'params': [p for n, p in model.named_parameters() if
#                     any(name == n for name in bart_special2)],
#          'lr': 0, 'weight_decay': 0.0},
#     ]
#     ensure_all_training = len(list(model.named_parameters())) == sum(len(i['params']) for i in optimizer_parameters)
#     assert ensure_all_training, "some param not training."
#     return optimizer_parameters


def reinit_last_layers(model: PreTrainedModel, num_layers: int, kaiming=False):
    """Re-initialize the last-k transformer layers.
    Args:
        model: The target transformer model.
        num_layers: The number of layers to be re-initialized.
    """
    if num_layers <= 0:
        return
    if hasattr(model.encoder, "layer"):
        model.encoder.layer[-num_layers:].apply(lambda x: init_weights(x, kaiming=kaiming))
    elif hasattr(model.encoder, "layers"):
        model.encoder.layers[-num_layers:].apply(lambda x: init_weights(x, kaiming=kaiming))
    elif hasattr(model.encoder, "blocks"):
        model.encoder.blocks[-1][-num_layers:].apply(lambda x: init_weights(x, kaiming=kaiming))
    else:
        print(model)
        raise ValueError("can't re-init.")


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0, weight_decay_head=False):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    name = "backbone"
    #  and p.requires_grad
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if name in n and
                    not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if name in n and
                    any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if name not in n],
         'lr': decoder_lr, 'weight_decay': 0.0 if not weight_decay_head else weight_decay}
    ]
    return optimizer_parameters


def get_optimizer_grouped_parameters_with_llrd(model, encoder_lr, decoder_lr, weight_decay=0.0,
                                               weight_decay_head=False, llrd=0.9):
    """layerwise learning rate decay implementation
    from Raja
    """
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": decoder_lr,
            "weight_decay": weight_decay,
        },
    ]

    # initialize lrs for backbone layers
    layers = [model.backbone.embeddings] + list(model.backbone.encoder.layer)
    layers.reverse()
    lr = encoder_lr

    for layer in layers:
        lr *= llrd

        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },

            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]

    return optimizer_grouped_parameters


def get_optimizer(model_parameters, config):
    model_classes = {
        "adamw8bit": bnb.optim.AdamW8bit if bnb is not None else None,
        "adamw": torch.optim.AdamW,
        "lion": Lion
    }
    name = config.optim.optimizer.get("name", "adamw8bit")
    if name.lower() == "adamw8bit":
        if bnb is None:
            raise ModuleNotFoundError("you are using a 8bit AdamW, please install `bitsandbytes`")
    if name.lower() not in model_classes:
        raise ValueError(f"no optimizer named {name}, choice one from {', '.join(model_classes.keys())}")
    model_class = model_classes[name.lower()]
    print(f"using {name} optimizer.")
    return model_class(
        model_parameters, lr=config.optim.optimizer.lr, betas=config.optim.optimizer.get("betas", (0.9, 0.999)),
        eps=config.optim.optimizer.eps, weight_decay=config.optim.optimizer.weight_decay
    )
