import pytorch_lightning as pl
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import MultiheadAttention
from transformers.pytorch_utils import Conv1D as HFConv1D

SUPPORTED_LORA_TYPES = (
    nn.Linear,
    nn.Embedding,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    HFConv1D,
    MultiheadAttention,
)


def apply_lora(model, lora_cfg):
    """
    Applies LoRA adaptation to the model's network using PEFT.

    Args:
        model: A LightningModule object containing a 'network' attribute.
        lora_cfg: Configuration object with attributes for configuring LoRA

    Returns:
        The model with LoRA applied to its network.
    """
    assert isinstance(model, pl.LightningModule), "model must be a LightningModule"

    if lora_cfg.target_modules == "all":
        target_modules = [
            name
            for name, module in model.network.named_modules()
            if isinstance(module, SUPPORTED_LORA_TYPES)
        ]
    else:
        raise NotImplementedError(
            "Haven't added functionality for injecting LoRA to specific layers"
        )

    peft_config = LoraConfig(
        r=lora_cfg.rank, lora_dropout=lora_cfg.dropout, target_modules=target_modules
    )
    peft_network = get_peft_model(model.network, peft_config)
    model.network = peft_network
    return model
