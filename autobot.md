# Instructions for running this thing on Autobot.


0. Before you do anything, make sure you've built your docker image and pushed it to dockerhub!!!

1. ssh into autobot:

    ```
    ssh <SCS_username>@autobot.vision.cs.cmu.edu
    ```

    a. *YOU ONLY NEED TO DO THIS ONCE*: Add your wandb API key to your bashrc:

        ```bash
        echo 'export WANDB_API_KEY="your_api_key_here"' >> ~/.bashrc
        source ~/.bashrc
        ```

2. Find a node on http://autobot.vision.cs.cmu.edu/mtcmon/ which has open GPUs.

3. SSH into that node:

    ```
    ssh autobot-0-33
    ```

    a. *YOU ONLY NEED TO DO THIS ONCE*: Create some scratch directories for your data and logs.

        ```bash
        mkdir -p /scratch/$(whoami)/data
        mkdir -p /scratch/$(whoami)/logs
        ```
4. Run a training job like so. Don't worry about building or installing. You can modify the files here to map to whatever you want. In future iterations of this, we'll make this easier to do (aka by using a hydra singularity condfig file or something so you don't have to explictly map as arguments).

    You can also change which GPU you want access to using CUDA_VISIBLE_DEVICES below.

    ```bash
    SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 \
    SINGULARITYENV_WANDB_DOCKER_IMAGE=python-ml-project-template \
    singularity exec \
    --nv \
    --pwd /opt/$(whoami)/code \
    -B /scratch/$(whoami)/data:/opt/data \
    -B /scratch/$(whoami)/logs:/opt/logs \
    docker://beisner/python-ml-project-template \
    python scripts/train.py \
        dataset.data_dir=/opt/data \
        log_dir=/opt/logs
    ```
