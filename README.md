# lfd3d

## Installation

This project uses [pixi](https://pixi.sh/latest/) for dependency management.

``` bash
# avoid LeRobot install errors
export GIT_LFS_SKIP_SMUDGE=1
export CPPFLAGS="-I/usr/include"
export CFLAGS="-DHAVE_LINUX_INPUT_H"

pixi install
pixi run install-deps
pixi run setup-pre-commit
pixi shell
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
- Disparity: Using [sriramsk1999/FoundationStereo](https://github.com/sriramsk1999/FoundationStereo/)

### RPAD-Foxglove

- Download and process recordings from Foxglove using [https://github.com/r-pad/lfd3d-system/](`lfd3d-system`).
- Generata GT for human demonstrations with modified version of [https://github.com/sriramsk1999/wilor/](`wilor`).
- Generata GT for robot demonstrations with `src/lfd3d/datasets/rpad_foxglove/render_aloha.py`.
- Annotate events with `src/lfd3d/datasets/rpad_foxglove/annotate_events.py`.

### RPAD-Lerobot
- With your collected lerobot dataset, run upgrade_dataset.py from the lerobot repo to generate a [repo_id]_goal repo
- Run a training job like this:
```python scripts/train.py model=articubot dataset=rpadLerobot dataset.repo_id=beisner/aloha_plate_placement_goal dataset.augment_train=True dataset.data_sources="[aloha]" resources.num_workers=16 dataset.cache_dir=/data/lfd3d_dataloading_cache
```

### RT-1

**NOTE:** This section needs to be updated, rgb features should come from dinov2, set up gripper centric preds

- Tracks: Generated using `rt1_inference.py` from [sriramsk1999/CoTracker](https://github.com/sriramsk1999/co-tracker)
- Depth: Generated using `rt1_inference.py` from [sriramsk1999/RollingDepth](https://github.com/sriramsk1999/RollingDepth)

After generating tracks and depth:

1. Preprocess captions with `src/lfd3d/datasets/rt1/process_captions.py`.
2. RGB/text features: Generated using `src/lfd3d/datasets/rt1/rgb_text_feature_gen.py`.
3. Chunking and filtering: Generated using `src/lfd3d/datasets/rt1/save_event_rgb.py`.

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
    -v $(pwd)/mano:/opt/rpad/code/mano \
    --gpus all \
    --shm-size=8G \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_DOCKER_IMAGE=lfd3d \
    $DOCKERHUB_USERNAME/lfd3d pixi run python scripts/train.py \
        model=df_cross \
        dataset=<dataset-name> \
        dataset.data_dir=/opt/rpad/data
```

## Acknowledgements

This codebase was adapted from [TAX3D](https://github.com/ey-cai/non-rigid/) and [python-ml-project-template](https://github.com/r-pad/python_ml_project_template/)
