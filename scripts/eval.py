import hydra
import lightning as L
import omegaconf
import torch
import torch.utils._pytree as pytree
import wandb

from python_ml_project_template.datasets.cifar10 import CIFAR10DataModule
from python_ml_project_template.metrics.classification import get_metrics
from python_ml_project_template.models.classifier import ClassifierInferenceModule
from python_ml_project_template.utils.script_utils import (
    PROJECT_ROOT,
    create_model,
    flatten_outputs,
    match_fn,
)


@torch.no_grad()
@hydra.main(config_path="../configs", config_name="eval", version_base="1.3")
def main(cfg):
    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("medium")

    # Global seed for reproducibility.
    L.seed_everything(42)

    ######################################################################
    # Create the datamodule.
    # Should be the same one as in training, but we're gonna use val+test
    # dataloaders.
    ######################################################################

    datamodule = CIFAR10DataModule(
        root=cfg.dataset.data_dir,
        batch_size=cfg.inference.batch_size,
        num_workers=cfg.resources.num_workers,
    )
    # Gotta call this in order to establish the dataloaders.
    datamodule.setup("predict")

    ######################################################################
    # Set up logging in WandB.
    # This is a different job type (eval), but we want it all grouped
    # together. Notice that we use our own logging here (not lightning).
    ######################################################################

    # Create a run.
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        dir=cfg.wandb.save_dir,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        job_type=cfg.job_type,
        save_code=True,  # This just has the main script.
        group=cfg.wandb.group,
    )

    # Log the code.
    wandb.run.log_code(
        root=PROJECT_ROOT,
        include_fn=match_fn(
            dirs=["configs", "scripts", "src"],
            extensions=[".py", ".yaml"],
        ),
    )

    ######################################################################
    # Create the network(s) which will be evaluated (same as training).
    # You might want to put this into a "create_network" function
    # somewhere so train and eval can be the same.
    #
    # We'll also load the weights.
    ######################################################################

    network = create_model(
        image_size=cfg.dataset.image_size,
        num_classes=cfg.dataset.num_classes,
        model_cfg=cfg.model,
    )

    # Get the checkpoint file. If it's a wandb reference, download.
    # Otherwise look to disk.
    checkpoint_reference = cfg.checkpoint.reference
    if checkpoint_reference.startswith(cfg.wandb.entity):
        # download checkpoint locally (if not already cached)
        artifact_dir = cfg.wandb.artifact_dir
        artifact = run.use_artifact(checkpoint_reference, type="model")
        ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
    else:
        ckpt_file = checkpoint_reference

    # Load the network weights.
    ckpt = torch.load(ckpt_file)
    network.load_state_dict(
        {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
    )

    ######################################################################
    # Create an inference module, which is basically just a bare-bones
    # class which runs the model. In this example, we only implement
    # the "predict_step" function, which may not be the blessed
    # way to do it vis a vis lightning, but whatever.
    #
    # If this is a downstream application or something, you might
    # want to implement a different interface (like with a "predict"
    # function), so you can pass in un-batched observations from an
    # environment, for instance.
    ######################################################################

    model = ClassifierInferenceModule(network)

    ######################################################################
    # Create the trainer.
    # Bit of a misnomer here, we're not doing training. But we are gonna
    # use it to set up the model appropriately and do all the batching
    # etc.
    #
    # If this is a different kind of downstream eval, chuck this block.
    ######################################################################

    trainer = L.Trainer(
        accelerator="gpu",
        devices=cfg.resources.gpus,
        precision="16-mixed",
        logger=False,
    )

    ######################################################################
    # Run the model on the train/val/test sets.
    # This outputs a list of dictionaries, one for each batch. This
    # is annoying to work with, so later we'll flatten.
    #
    # If a downstream eval, you can swap it out with whatever the eval
    # function is.
    ######################################################################

    train_outputs, val_outputs, test_outputs = trainer.predict(
        model,
        dataloaders=[
            *datamodule.val_dataloader(),  # There are two different loaders (train_val and val).
            datamodule.test_dataloader(),
        ],
    )

    for outputs_list, name in [
        (train_outputs, "train"),
        (val_outputs, "val"),
        (test_outputs, "test"),
    ]:
        # Put everything on CPU, and flatten a list of dicts into one dict.
        out_cpu = [pytree.tree_map(lambda x: x.cpu(), o) for o in outputs_list]
        outputs = flatten_outputs(out_cpu)

        # Compute the metrics.
        metrics = get_metrics(outputs["preds"], outputs["labels"])
        global_acc = metrics["global_acc"]
        macro_acc = metrics["macro_acc"]
        acc_df = metrics["acc_df"]

        # Log the metrics + table to wandb.
        run.summary[f"{name}_true_accuracy"] = global_acc
        run.summary[f"{name}_class_balanced_accuracy"] = macro_acc

        table = wandb.Table(dataframe=acc_df)
        run.log({f"{name}_accuracy_table": table})


if __name__ == "__main__":
    main()
