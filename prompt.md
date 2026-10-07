You are physical AI engineer and research scientist, roboticist and robotics control expert.
Look at the script below to run the each experiment:

0. Environment
tmux new -s perstep
conda activate dp
cd ~/robot-learning/diffusion_policy

1. Labels (done; rerun only if you change something)
python make_perstep_labels.py --budget 2 --overwrite
python make_perstep_labels.py --budget 4 --overwrite
python make_perstep_labels.py --budget 6 --overwrite
Each prints two checks that must be ~0 and the replay reward from the ta

2. Configs (done)
python make_perstep_configs.py

3. Smoke test (done; optional rerun, about 5 min)
python train.py --config-dir=. --config-name=image_pusht_perstep_b6.yamlmode=disabled \
  task.env_runner.n_train=1 task.env_runner.n_test=2 task.env_runner.n_envs=3 task.env_runner.max_steps=48 \
  policy.num_inference_steps=10 hydra.run.dir=data/outputs/smoke_perstep_b6
rm -rf data/outputs/smoke_perstep_b6      # ~12 GB of checkpoints

4. Train, one budget at a time on the 12 GB GPU
python train.py --config-dir=. --config-name=image_pusht_perstep_b2.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/dp-perstep-b2_seed42'
python train.py --config-dir=. --config-name=image_pusht_perstep_b4.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/dp-perstep-b4_seed42'
python train.py --config-dir=. --config-name=image_pusht_perstep_b6.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/dp-perstep-b6_seed42'
- Cost: Run only 1000 epochs for each experiment
- If run Extra seeds: also override logging.id=dp-perstep-b2-s43 logging.name=dandb resumes the seed-42 run.

5. Validation during training (automatic)
Every 50 epochs, the runner plays 50 test episodes (seeds 100000–100049) logs test/mean_score to wandb. The 5 best checkpoints are kept. To check
progress against dp-04 at any time:
python compare_runs.py data/outputs/dp-04_pusht_image_seed42 data/outputepoch 1000
Ignore train_action_mse_error: it now mixes px and px/s, so it isn't com

6. Final validation on fresh seeds
The best checkpoints were picked using seeds 100000–100049, so score theuded:
python eval_checkpoint.py -c "data/outputs/dp-04_pusht_image_seed42/checkpoints/epoch=0250-test_mean_score=0.907.ckpt" -o data/eval/dp-04 --n_test 500 --test_start_seed 200000
python eval_checkpoint.py -c "<best ckpt of dp-perstep-b2_seed42>" -o dat_start_seed 200000
# same for b4 and b6
Each prints the mean score, its standard error and the success rate (coverage ≥ 95%), and writes eval_summary.json.

7. Read the results
Record per experiment:
- best in-training score and its epoch,
- mean of the last 10 evaluations up to epoch 1000,
- the 500-episode score ± standard error,
- the label replay reward.

dp-04 so far: best 0.907 at epoch 250, last-10 mean 0.872.

In-training scores swing by ±0.03–0.05 between evaluations. With one seed per budget, treat gaps under about 0.03 as a tie, and add seeds 43 and 44 only to budgets that look
promising.

You need to monitor each experiment 