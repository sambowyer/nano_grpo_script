# Nano GRPO Implementation

Using [Maxime's Code](https://gist.github.com/MaximeRobeyns/baed5ec2945cade48b6d705e3d6a214d) mainly, based on the `nano_r1_script.py` file in [this repo](https://github.com/McGill-NLP/nano-aha-moment/tree/main).


## Conda environment with environment.yml

```bash
conda env create -f environment.yml
```

#### Bluepebble

N.B.: To get it working on bluepebble, I had to run

```
conda install -c nvidia cuda-compiler
```

after the regular env setup.

## Run the script

```bash
python train.py --algo grpo
```

Note: DAPO doesn't seem to be working at the moment with the currently chosen Qwen model.

# Structure

- `train.py`: Main script to train the model.
- `utils.py`: Utility functions, mostly for loading models and tokenizing.
- `rewards.py`: Reward functions.
- `episodes.py`: Episode processing functions (takes in group rollouts, gets rewards, returns metrics about rollouts -- some of which are used for calculating the loss).
- `loss.py`: Loss functions (for GRPO, Dr. GRPO, and DAPO).
