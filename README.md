# Revisiting the Minimalist Approach to Offline Reinforcement Learning

<!-- [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789) -->

<img src="figures/showcase.png" alt="Method and Results Summary" title="Method and Results Summary">

## Dependencies & Docker setup
To set up a python environment (with dev-tools of your taste, in our workflow, we use conda and python 3.8), just install all the requirements:

```commandline
python3 install -r requirements.txt
```

However, in this setup, you must install mujoco210 binaries by hand. Sometimes this is not super straightforward, but this recipe can help:
```commandline
mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
```
You may also need to install additional dependencies for mujoco_py. 
We recommend following the official guide from [mujoco_py](https://github.com/openai/mujoco-py).

### Docker

We also provide a more straightforward way with a dockerfile that is already set up to work. All you have to do is build and run it :)
```commandline
docker build -t rebrac .
```
To run, mount current directory:
```commandline
docker run -it \
    --gpus=all \
    --rm \
    --volume "<PATH_TO_THE_REPO>:/workspace/" \
    --name rebrac \
    rebrac bash
```

### V-D4RL
To reproduce V-D4RL, you need to download the corresponding datasets. The easiest way is probably to run the `download_vd4rl.sh` script we provide. 

You can also do it manually with the following links to the datasets archives: 

* [walker_walk](https://drive.google.com/file/d/1F4LIH_khOFw1asVvXo82OMa2tZ0Ax5Op/view?usp=sharing)
* [cheetah_run](https://drive.google.com/file/d/1WR2LfK0y94C_1r2e1ps1dg6zSMHlVY_e/view?usp=sharing)
* [humanoid_walk](https://drive.google.com/file/d/1zTBL8KWR3o07BQ62jJR7CeatN7vb-vjd/view?usp=sharing)

Note that provided links contain only datasets reported in the paper without distraction and multitasking.

After downloading the datasets, you must put the data into the `vd4rl` directory.

## How to reproduce experiments

### Training

Configs for the main experiments are stored in the `configs/rebrac/<task_type>` and `configs/rebrac-vis/<task_type>`. 
All available hyperparameters are listed in the `rebrac/algorithms/rebrac.py` for D4RL and `rebrac/algorithms/rebrac_torch_vis.py` for V-D4RL.

For example, to start ReBRAC training process with D4RL `halfcheetah-medium-v2` dataset, run the following:
```commandline
PYTHONPATH=. python3 src/algorithms/rebrac.py --config_path="configs/rebrac/halfcheetah/halfcheetah_medium.yaml"
```

For V-D4RL `walker_walk-expert-v2` dataset, run the following:
```commandline
PYTHONPATH=. python3 src/algorithms/rebrac_torch_vis.py --config_path="configs/rebrac-vis/walker_walk/expert.yaml"
```

### Targeted Reproduction
For better transparency and replication, we release all the experiments (5k+) in the form of [Weights & Biases reports](https://wandb.ai/tlab/ReBRAC/reportlist).

If you want to replicate results from our work, you can use the configs for [Weights & Biases Sweeps](https://docs.wandb.ai/guides/sweeps/quickstart) provided in the `configs/sweeps`. Note, we do not supply a codebase for both IQL and SAC-RND. However, in our work, we relied upon these implementations: [IQL (CORL)](https://github.com/tinkoff-ai/CORL), [SAC-RND (original implementation)](https://github.com/tinkoff-ai/sac-rnd).

| Paper element          | Sweeps to run from `configs/sweeps/`                         |
|------------------------|--------------------------------------------------------------|
| Tables 2, 3, 4         | `eval/rebrac_d4rl_sweep.yaml`, `eval/td3_bc_d4rl_sweep.yaml` |
| Table 5                | `eval/rebrac_visual_sweep.yaml`                              |
| Table 6                | All sweeps from `ablations`                                  |
| Figure 2               | All sweeps from `network_sizes`                              |
| Hyperparameters tuning | All sweeps from `tuning`                                     |


### Reliable Reports

We also provide scripts for reconstructing the graphs in our paper: `eop/ReBRAC_ploting.ipynb`, including performance profiles, probability of improvement, and expected online performance. For your convenience, we repacked the results into .pickle files, so you can re-use them for further research and head-to-head comparisons. 

<!-- # Citing
If you use this code for your research, please consider the following bibtex:
```

``` -->
