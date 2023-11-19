# CS5446 Project: Team POGS

# Quickstart
```python
conda env create -f requirements_conda.yml
```

# Repo Structure
## Saved Models

All trained models are stored in the `saved` folder.

##  Notebooks for Hyperparameter Tuning & Training of Agent

| Model | Notebook Name                     | Description       |
|---    |---                                |---                |
| DQN   | DQN-Optuna<br />DQN-Optuna-EnvD   | Optuna Tuning     | 
| DQN   | DQN<br />DQN-EnvD                 | Training Script   |
| A2C   | A2C-Optuna-EnvA-mini<br />A2C-Optuna-EnvA<br />A2C-Optuna-EnvB-mini<br />A2C-Optuna-EnvB<br />A2C-Optuna-EnvC-mini<br />A2C-Optuna-EnvC<br />A2C-Optuna-EnvD-v2<br />A2C-Optuna-EnvD| Optuna Tuning + Training Script  |
| PPO   | PPO-Optuna<br />PPO-Optuna-EnvD  | Optuna Tuning  |
| PPO   | PPO<br />PPO-EnvD  | Training Script  |
| RPPO  | RecurrentPPO_Optuna  | Optuna Tuning  |
| RPPO  | RecurrentPPO_EnvA<br />RecurrentPPO_EnvB<br />RecurrentPPO_EnvC<br />RecurrentPPO_EnvD<br />  | Training Script   |