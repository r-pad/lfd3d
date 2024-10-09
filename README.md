# lfd3d

## Installation

Install TAX3D following the instructions [here](https://github.com/ey-cai/non-rigid/tree/articulated)

## Training

To train a model:
```
cd scripts/
python train.py model=df_cross dataset=hoi4d dataset.data_dir=<path/to/dataset/>
```

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

## Using the CI.

Set up pushing to docker:

Put the following secrets in the Github repository:
* `DOCKERHUB_USERNAME`: Your Dockerhub username
* `DOCKERHUB_TOKEN`: Your Dockerhub token

You'll also need to Ctrl-F replace instances of beisner and baeisner with appropriate usernames.

## Running on Clusters

* [Autobot](autobot.md)
