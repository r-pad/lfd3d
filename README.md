# lfd3d

## Installation

First, we'll need to install platform-specific dependencies for Pytorch. See [here](https://pytorch.org/get-started/locally/) for more details. For example, if we want to use CUDA 11.8 with Pytorch 2.

```bash

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118/

```

Then, we can install the package itself:

```bash

pip install -e ".[develop,notebook]"

```

Then we install pre-commit hooks:

```bash

pre-commit install

```

## Training

To train a model:
```
python scripts/train.py model=df_cross dataset=hoi4d dataset.data_dir=<path/to/dataset/>
```

Optionally, also set `dataset.cache_dir` to save processed data for faster training.

The best checkpoint is saved to WandB at the end of training. If training is interrupted or if you want to save an intermdiate checkpoint stored in `logs/`:
```
cd scripts/
python upload_wandb.py --run_id <wandb-run-id> --checkpoint_path <path/to/checkpoint>
```

## Evaluation

Evaluation:
```
python scripts/eval.py checkpoint.run_id=<wandb-run-id> dataset.data_dir=<path/to/dataset/>
```

If the model was trained on the cluster, `dataset.cache_dir` needs to be overridden and set to null.

## Docker

Make sure to set the `$DOCKERHUB_USERNAME` shell variable. To build the docker image, run:

```bash
docker build -t $DOCKERHUB_USERNAME/lfd3d .
```

To run the training script locally, run:

```bash
WANDB_API_KEY=<API_KEY>
# Optional: mount current directory to run / test new code.
# Mount data directory to access data.
docker run \
    -v $(pwd)/data:/opt/rpad/data \
    -v $(pwd)/logs:/opt/rpad/logs \
    --gpus all \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_DOCKER_IMAGE=lfd3d \
    $DOCKERHUB_USERNAME/lfd3d python scripts/train.py \
        model=df_cross \
        dataset=hoi4d \
        dataset.data_dir=/root/data
```

To push this:

```bash
docker push $DOCKER_USERNAME/lfd3d:latest
```

## Acknowledgements

This codebase was adapted from [TAX3D](https://github.com/ey-cai/non-rigid/) and [python-ml-project-template](https://github.com/r-pad/python_ml_project_template/)
