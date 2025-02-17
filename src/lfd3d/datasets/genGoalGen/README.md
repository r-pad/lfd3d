# GenGoalGen Evaluation

A qualitative evaluation of goal generation on some real world images.

### Preprocessing the images

- Get segmentation masks of active object with SAM2.
- Generate RGB/test features with SIGLIP and ConceptFusion

### Evaluation

``` sh
python scripts/eval_genGoalGen.py checkpoint.run_id=<checkpoint-id> dataset.name=genGoalGen dataset.data_dir=/path/to/dataset/ dataset.cache_dir=null
```
