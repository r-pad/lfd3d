# W&B only uploads the model checkpoint after training completes successfully
# This script is for manually uploading a checkpoint during the run
# or if the run crashes in between, so that we can evaluate with the checkpoint
#
# Example usage: python script.py --run_id o0aiolj1 --checkpoint_path /home/sriram/Desktop/lfd3d/logs/train_hoi4d/2024-11-10/23-38-49/checkpoints/epoch=1289-step=167700-val/rmse=0.117.ckpt

import argparse

import wandb

# Set up argument parser
parser = argparse.ArgumentParser(description="Upload checkpoint to W&B")
parser.add_argument("--run_id", type=str, required=True, help="W&B run ID")
parser.add_argument(
    "--checkpoint_path", type=str, required=True, help="Path to checkpoint file"
)

# Parse arguments
args = parser.parse_args()

# Initialize wandb and upload artifact
wandb.init(entity="r-pad", project="lfd3d", id=args.run_id, resume="must")

artifact = wandb.Artifact(f"model-{args.run_id}", type="model")
artifact.add_file(local_path=args.checkpoint_path, name="model.ckpt")
wandb.log_artifact(artifact, aliases=["latest", "best"])
wandb.finish()
