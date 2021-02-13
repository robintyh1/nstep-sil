
# Code for NeurIPS 2020 paper: Self-imitation Learning via Generalized Lower Bound Q-learning

## Dependencies
This implementation depends on the following libraries as well as dependencies that support these libraries.
* [OpenAI baselines](https://github.com/openai/baselines)
* [SpinningUp](https://github.com/openai/spinningup)

To run experiments with simulated environments, you will also need to install 
* [OpenAI gym](https://github.com/openai/gym)
* [Bullet physics](https://github.com/bulletphysics/bullet3)
* [DM control](https://github.com/deepmind/dm_control)

## Run the code
Hyper-parameters are specified in the python code. After running experiments, performance curves will be saved in a sub-directory in the current working directory for further processing.

For example, to run the nstep SIL algorithm with delayed environments, run the following
```bash
python td3_nstep_sil.py --env HalfCheetah-v3 --seed 100 --delay 3 --nstep 5 --sil-weights 0.1
```

To run without SIL, set the proper hyper-parameter
```bash
python td3_nstep_sil.py --env HalfCheetah-v3 --seed 100 --delay 3 --nstep 5 --sil-weights 0.0
```

To run the return based SIL algorithm with delayed environments, run the following
```bash
python td3_return_sil.py --env HalfCheetah-v3 --seed 100 --delay 3 --nstep 5 --sil-weights 0.1
```
## Citations
If you find this code base useful, you are encouraged to cite the following
* Yunhao Tang, "Self-imitation Learning via Generalized Lower Bound Q-learning". arXiv:2006.07442 [cs.LG], 2020.
