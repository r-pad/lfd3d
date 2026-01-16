python eval_lerobot_episode.py checkpoint.run_id=xa8xefu6 dataset=rpadLerobot dataset.repo_id="sriramsk/fold_onesie_MV_20251025_ss_hg_mini" model=dino_3dgp checkpoint.type=rmse resources.num_workers=32 dataset.val_episode_ratio=1 inference.n_eval_episode=10

python eval_lerobot_episode.py checkpoint.run_id=xa8xefu6 dataset=rpadLerobot dataset.repo_id="sriramsk/fold_onesie_MV_20251210_ss_hg" model=dino_3dgp checkpoint.type=rmse resources.num_workers=32 dataset.val_episode_ratio=1 inference.n_eval_episode=10

python eval_lerobot_episode.py checkpoint.run_id=xa8xefu6 dataset=rpadLerobot dataset.repo_id="sriramsk/fold_onesie_MVHuman_20251210_ss_hg" model=dino_3dgp checkpoint.type=rmse resources.num_workers=32 dataset.val_episode_ratio=1 inference.n_eval_episode=10

python eval_lerobot_episode.py checkpoint.run_id=xa8xefu6 dataset=rpadLerobot dataset.repo_id="sriramsk/fold_shirt_MV_20251210_ss_hg" model=dino_3dgp checkpoint.type=rmse resources.num_workers=32 dataset.val_episode_ratio=1 inference.n_eval_episode=10

python eval_lerobot_episode.py checkpoint.run_id=xa8xefu6 dataset=rpadLerobot dataset.repo_id="sriramsk/fold_towel_forEval_MV_20251210_ss_hg" model=dino_3dgp checkpoint.type=rmse resources.num_workers=32 dataset.val_episode_ratio=1 inference.n_eval_episode=10

python eval_lerobot_episode.py checkpoint.run_id=xa8xefu6 dataset=rpadLerobot dataset.repo_id="sriramsk/fold_towel_MVHuman_20251210_ss_hg" model=dino_3dgp checkpoint.type=rmse resources.num_workers=32 dataset.val_episode_ratio=1 inference.n_eval_episode=10


python analyze_latents.py --dset1_path logs/xa8xefu6_sriramsk/fold_onesie_MV_20251210_ss_hg --dset2_path logs/xa8xefu6_sriramsk/fold_onesie_MVHuman_20251210_ss_hg --output robotOnesie_humanOnesie_withOTv2.png

python analyze_latents.py --dset1_path logs/xa8xefu6_sriramsk/fold_onesie_MV_20251210_ss_hg --dset2_path logs/xa8xefu6_sriramsk/fold_onesie_MV_20251025_ss_hg_mini --output robotOnesie_robotOnesie_withOTv2.png

python analyze_latents.py --dset1_path logs/xa8xefu6_sriramsk/fold_onesie_MV_20251210_ss_hg --dset2_path logs/xa8xefu6_sriramsk/fold_shirt_MV_20251210_ss_hg --output robotOnesie_robotShirt_withOTv2.png

python analyze_latents.py --dset1_path logs/xa8xefu6_sriramsk/fold_towel_forEval_MV_20251210_ss_hg --dset2_path logs/xa8xefu6_sriramsk/fold_towel_MVHuman_20251210_ss_hg --output humanTowel_robotTowel_withOTv2.png
