#!/bin/bash
# Build a docker image, convert it to as singularity image, and push it to the seuss cluster.
# Right now, this is a total hack since we can't actually build the docker image on the cluster,
# nor can we build the singularity image on the cluster. So we build the docker image locally,
# convert it to a singularity image locally, and then push it to the cluster.

# Whole script fails if any command fails.

set -e

# Set some variables.
dockerhub_username=$DOCKERHUB_USERNAME
project_name=lfd3d

# Get paths.
script_path=$(realpath $0)
script_dir=$(dirname $script_path)
root_dir=$(realpath ${script_dir}/..)

# Compute a good tag for the image, which will be <dockerhub_username>/<project_name>:<branch-name>-scratch.
sanitized_branch_name=$(${script_dir}/sanitize_branch_name.bash)

# Build the docker image.
docker build -t ${dockerhub_username}/${project_name}:${sanitized_branch_name}-scratch .

# Convert the docker image to a singularity image, and save it in the .singularity_images directory.
mkdir -p ${root_dir}/.singularity_images
sif_name=${root_dir}/.singularity_images/${project_name}_${sanitized_branch_name}-scratch.sif
singularity build ${sif_name} docker-daemon://${dockerhub_username}/${project_name}:${sanitized_branch_name}-scratch

# Rsync the singularity image to the seuss cluster.
rsync -avz --progress ${sif_name} $CMU_SCS_USERNAME@seuss.ri.cmu.edu:/home/$CMU_SCS_USERNAME/singularity_images/
