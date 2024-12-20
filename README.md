# MFPO
This work "A Framework for Federated Reinforcement Learning with Interaction and Communication Efficiency" has been submitted in INFOCOM 2024.
## :page_facing_up: Description
Momentum-assisted Federated Policy Optimization (MFPO), capable of jointly optimizing both interaction and communication complexities. Specifically, we introduce a new FRL framework that utilizes momentum, importance sampling, and extra server-side updating to control the variates of stochastic policy gradients and improve the efficiency of data utilization.
## :wrench: Dependencies
- Python == 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch == 1.8.1](https://pytorch.org/)
- [MuJoCo == 2.3.6](http://www.mujoco.org) 
- NVIDIA GPU (RTX A6000) + [CUDA 11.1](https://developer.nvidia.com/cuda-downloads)
### Installation
1. Clone repo
    ```bash
    git clone [https://github.com/HansenHua/MFPO-INFOCOM24.git](https://github.com/HansenHua/MFPO-INFOCOM24.git)
    cd MFPO-INFOCOM24
    ```
2. Install dependent packages
    ```
    pip install -r requirements.txt
    ```
## :zap: Quick Inference

Get the usage information of the project
```bash
cd code
python main.py -h
```
Then the usage information will be shown as following
```
usage: main.py [-h] [--env_name ENV_NAME] [--method METHOD] [--gamma GAMMA] [--batch_size BATCH_SIZE]
               [--local_update LOCAL_UPDATE] [--num_worker NUM_WORKER] [--average_type AVERAGE_TYPE] [--c C]
               [--seed SEED] [--lr_a LR_A] [--lr_c LR_C]
               mode max_iteration

positional arguments:
  mode                  train or test
  max_iteration         maximum training iteration

optional arguments:
  -h, --help            show this help message and exit
  --env_name ENV_NAME   the name of environment
  --method METHOD       method name
  --gamma GAMMA         gamma
  --batch_size BATCH_SIZE
                        batch_size
  --local_update LOCAL_UPDATE
                        frequency of local update
  --num_worker NUM_WORKER
                        number of federated agents
  --average_type AVERAGE_TYPE
                        average type (target/network/critic)
  --c C                 momentum parameter
  --seed SEED           random seed
  --lr_a LR_A           learning rate of actor
  --lr_c LR_C           learning rate of critic
```
Test the trained models provided in [MFPO-Momentum-assisted Federated Policy Optimization](https://github.com/HansenHua/MFPO-INFOCOM24/tree/main/log).
```
python main.py CartPole-v1 MFPO test
```
## :computer: Training

We provide complete training codes for MFPO.<br>
You could adapt it to your own needs.

	```
    python main.py CartPole-v1 MFPO train
	```
	The log files will be stored in [MFPO-INFOCOM24/code/log](https://github.com/HansenHua/MFPO-INFOCOM24/tree/main/code/log).
## :checkered_flag: Testing
1. Testing
	```
	python main.py CartPole-v1 MFPO test
	```
2. Illustration

We alse provide the performance of our model. The illustration videos are stored in [MFPO-INFOCOM24/performance](https://github.com/HansenHua/MFPO-INFOCOM24/tree/main/performance).

## :e-mail: Contact

If you have any question, please email `huaxy24@mails.tsinghua.edu.cn`.
