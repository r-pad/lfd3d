#!/bin/bash

# Sanitize a branch name for use in a docker image tag.

branch_name=$(git branch | grep \* | cut -d ' ' -f2)

# Sanitize by replacing all slashes with underscores.
sanitized_branch_name=$(echo $branch_name | sed 's/\//_/g')

echo $sanitized_branch_name
