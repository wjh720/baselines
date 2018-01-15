# PPO_CAG

- Original paper: https://arxiv.org/abs/1707.06347
- Baselines blog post: https://blog.openai.com/openai-baselines-ppo/
- You should follow the https://github.com/wjh720/baselines to install this package.
- `mpirun -np 10 python3 -m baselines.ppo_CAG.run_atari` runs the algorithm.
- You should change the all `data_path` which is the absolute path of save_data in `pposgd_simple.py` and `pposgd_simple_test.py`.
- If you want to test the performance, you should change `timesteps_per_actorbatch` which is the number of iteration of save_data in `test_atari.py`.
- If you want to load save_data to continue training, please change `flag_load` which is the number of of iteration of save_data in `train_atari.py`.
