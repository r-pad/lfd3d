#!/bin/bash

# This should take in 4 arguments:
# 1. the index of which GPU to use
# 2. model type
# 3. WANDB mode
# 4. the rest of the arguments for the train.py script

# Example usage:
# ./multi_cloth_train.sh 0 cross_point_relative online
# ./multi_cloth_train.sh 1 scene_flow disabled dataset.multi_cloth.hole=single dataset.multi_cloth.size=100

GPU_INDEX=$1
MODEL_TYPE=$2
WANDB_MODE=$3
shift
shift
shift
COMMAND=$@


# scene flow model - no object centric processing
if [ $MODEL_TYPE == "scene_flow" ]; then
  echo "Training scene flow model with command: $COMMAND."

  MODEL_PARAMS="model=df_base model.type=flow"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=flow dataset.scene=True dataset.world_frame=True"
# scene point model - no object centric processing
elif [ $MODEL_TYPE == "scene_point" ]; then
  echo "Training scene point model with command: $COMMAND."

  MODEL_PARAMS="model=df_base model.type=point"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=point dataset.scene=True dataset.world_frame=True"
# world frame cross flow
elif [ $MODEL_TYPE == "cross_flow_absolute" ]; then
  echo "Training absolute flow model with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=flow"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=flow dataset.scene=False dataset.world_frame=True"
# relative frame cross flow
elif [ $MODEL_TYPE == "cross_flow_relative" ]; then
  echo "Training relative flow model with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=flow"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=flow dataset.scene=False dataset.world_frame=False"
# world frame cross point
elif [ $MODEL_TYPE == "cross_point_absolute" ]; then
  echo "Training absolute point model with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=point"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=point dataset.scene=False dataset.world_frame=True"
# relative frame cross point
elif [ $MODEL_TYPE == "cross_point_relative" ]; then
  echo "Training relative point model with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=point"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=point dataset.scene=False dataset.world_frame=False"
# flow regression baseline
elif [ $MODEL_TYPE == "regression_flow" ]; then
  echo "Training flow regression model with command: $COMMAND."

  MODEL_PARAMS="model=regression model.type=flow"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=flow dataset.scene=False dataset.world_frame=False"
# point regression baseline
elif [ $MODEL_TYPE == "regression_point" ]; then
  echo "Training point regression model with command: $COMMAND."

  MODEL_PARAMS="model=regression model.type=point"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=point dataset.scene=False dataset.world_frame=False"
# relative frame cross point - HOI4D
elif [ $MODEL_TYPE == "cross_point_relative_hoi4d" ]; then
  echo "Training relative point model with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=point"
  DATASET_PARAMS="dataset=hoi4d dataset.type=point dataset.scene=False dataset.world_frame=False"
else
  echo "Invalid model type."
fi

WANDB_MODE=$WANDB_MODE python train_diff.py \
  $MODEL_PARAMS \
  $DATASET_PARAMS \
  wandb.group=proc_cloth \
  resources.gpus=[${GPU_INDEX}] \
  $COMMAND