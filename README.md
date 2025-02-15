# lfd3d

## Installation


```bash

conda create -n lfd3d python=3.10
pip install -r requirements.txt
pip install -e ".[develop]"

```

Install `flash-attn` separately:

```bash

pip install flash-attn --no-build-isolation

```

Then we install pre-commit hooks:

```bash

pre-commit install

```

Alternatively, use Docker (see below).

## Feature Generation

### HOI4D

- Tracks: Generated using `hoi4d_inference.py` from [sriramsk1999/SpaTracker](https://github.com/sriramsk1999/spatracker)
- RGB/text features: Generated using `src/lfd3d/datasets/hoi4d_processing/rgb_text_feature_gen.py`.

Test split generated using [sriramsk1999/general-flow](https://github.com/sriramsk1999/general-flow/)

Additionally, if `use_gflow_tracks=True`, generate event-wise tracks from `label_gen_event.py` in [sriramsk1999/general-flow](https://github.com/sriramsk1999/general-flow/). When training with these tracks, the projections of the tracks won't align perfectly, likely due to the errors in the object pose annotation when generating these tracks.

### RT-1

- Tracks: Generated using `rt1_inference.py` from [sriramsk1999/CoTracker](https://github.com/sriramsk1999/co-tracker)
- Depth: Generated using `rt1_inference.py` from [sriramsk1999/RollingDepth](https://github.com/sriramsk1999/RollingDepth)

After generating tracks and depth:

1. Preprocess captions with `src/lfd3d/datasets/rt1_processing/process_captions.py`.
2. RGB/text features: Generated using `src/lfd3d/datasets/rt1_processing/rgb_text_feature_gen.py`.
3. Chunking and filtering: Generated using `src/lfd3d/datasets/rt1_processing/save_event_rgb.py`.

Test split taken from [3D-VLA](https://github.com/UMass-Foundation-Model/3D-VLA)

## Training

To train a model:
```
python scripts/train.py model=df_cross dataset=hoi4d dataset.data_dir=<path/to/dataset/>
```

Optionally, also set `dataset.cache_dir` to save processed data for faster training.

Training with multiple datasets:
```
python scripts/train.py model=df_cross dataset=multi \
    dataset.datasets.rt1.data_dir=/data/sriram/rt1 \
    dataset.datasets.rt1.cache_dir=/home/sriram/Desktop/lfd3d/rt1_cache \
    dataset.datasets.hoi4d.data_dir=/data/sriram/hoi4d/hoi4d_data \
    dataset.datasets.hoi4d.cache_dir=/home/sriram/Desktop/lfd3d/hoi4d_cache
```


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
