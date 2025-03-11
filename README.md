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


Download the [MANO models](https://mano.is.tue.mpg.de/) and place them in `mano/`.

Alternatively, use Docker (see below).

## Feature Generation

### HOI4D

- GT Points (any one of the following):
  - Tracks from SpatialTracker: Generated using `hoi4d_inference.py` from [sriramsk1999/SpaTracker](https://github.com/sriramsk1999/spatracker)
  - Tracks from General Flow: Generate event-wise tracks from `label_gen_event.py` in [sriramsk1999/general-flow](https://github.com/sriramsk1999/general-flow/). When training with these tracks, the projections of the tracks won't align perfectly, due to the errors in the object pose annotations in the dataset
  - Tracks from MANO hand pose: Load hand pose tracks provided by HOI4D dataset. When training with these tracks, the projections of the tracks won't align perfectly, due to the errors in the hand pose annotations in the dataset
- RGB/text features: Generated with:
`python rgb_text_feature_gen.py --dataset hoi4d --input_dir </path/to/hoi4d/data>`

Test split generated using [sriramsk1999/general-flow](https://github.com/sriramsk1999/general-flow/)

### DROID

- GT Gripper Trajectory: Rendered using Mujoco in `src/lfd3d/datasets/droid/render_robotiq.py`.
- Subgoal generation: Using Gemini in `src/lfd3d/datasets/droid/chunk_droid_with_gemini.py`.
- RGB/text features: Generated with:
`python rgb_text_feature_gen.py --dataset droid --input_dir </path/to/droid/data>`

### RT-1

**NOTE:** This section needs to be updated, rgb features should come from dinov2, set up gripper centric preds

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

Build the docker image:

```bash
docker build -t $DOCKERHUB_USERNAME/lfd3d .
```

To run the training script:

```bash
docker run \
    -v </path/to/dataset>:/opt/rpad/data \
    -v $(pwd)/logs:/opt/rpad/logs \
    --gpus all \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_DOCKER_IMAGE=lfd3d \
    $DOCKERHUB_USERNAME/lfd3d python scripts/train.py \
        model=df_cross \
        dataset=<dataset-name> \
        dataset.data_dir=/opt/rpad/data
```

## Acknowledgements

This codebase was adapted from [TAX3D](https://github.com/ey-cai/non-rigid/) and [python-ml-project-template](https://github.com/r-pad/python_ml_project_template/)
