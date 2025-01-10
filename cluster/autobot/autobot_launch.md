Base command:

```
NCCL_P2P_LEVEL=NVL OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 scripts/train.py model=diptv3_cross dataset=hoi4d dataset.data_dir=/project_data/held/sskrishn/hoi4d/hoi4d_data dataset.cache_dir=/scratch/sskrishn/hoi4d_cache resources.gpus=4 training.epochs=500 training.batch_size=4 training.val_batch_size=4
```

TODO: Wrap into a cron job that queries for gpus using autobot.py and then launches. Refer "https://github.com/r-pad/python_ml_project_template/blob/main/cluster/launch_autobot.sh"
