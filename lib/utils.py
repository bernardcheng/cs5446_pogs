import yaml
from os.path import isfile
from pogema.animation import AnimationMonitor, AnimationConfig

def evaluate_metrics(model, env, num_episodes=10, num_trials=100, verbose=False, save_animation=False) -> dict:
    """Evaluate trained agent (model) on environment based on set number of trials.

    Args:
        model: Stable Baseline3 model
        env: Gym env
        num_episodes: Number of episodes in each trial
        num_trials: Number of trials used for evaluation
        verbose: Set to True to print env output and action taken for each step of trial 
        save_animation: Set to True to save successful trial as svg

    Returns:
        Dict containing calculated metrics (success_rate, step_array, ave_steps).
    """
    if save_animation:
        env = AnimationMonitor(env)

    success_count = 0
    step_array = []
    for trial in range(num_trials):
        obs, info = env.reset()

        max_step = num_episodes
        steps_taken = 0
        done = truncated = False
        while not done and max_step > 0:
            action, _ = model.predict(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            if verbose:
                print(f'Step {steps_taken} of Trial {trial} - Action: {action}, Steps Remaining: {max_step}, Done: {done}')
            max_step -= 1
            steps_taken += 1
            obs = next_obs

            # Check if agent was successful in that episode.
            if done:
                success_count += 1
                step_array.append(steps_taken)
                if save_animation:
                    env.save_animation(f"renders/render.svg", AnimationConfig(egocentric_idx=0))
                break
    
    # store evaluation metrics to be returned
    metrics = {}
    metrics['success_rate'] = success_count / num_trials
    metrics['step_array'] = step_array

    ave_steps = lambda x: sum(step_array)/len(step_array) if len(step_array) > 0 else None
    metrics['ave_steps'] = ave_steps(step_array)

    return metrics

def save_model_params(trial, model_name:str, save_path:str, default_params:dict=None) -> None:
    """
    """
    if isfile(save_path):
        with open(save_path) as file:
            # The FullLoader parameter handles the conversion from YAML scalar values to Python the dictionary format
            all_model_hyperparams = yaml.load(file, Loader=yaml.FullLoader)

    else:
        all_model_hyperparams = {}

    model_hyperparams = {**default_params}
    for key, value in trial.params.items():
        model_hyperparams[key] = value

    for key, value in trial.user_attrs.items():
        model_hyperparams[key] = value

    all_model_hyperparams[model_name] = model_hyperparams

    with open(save_path, 'w') as file:
        yaml.dump(all_model_hyperparams, file, default_flow_style=False)

def get_model_params(model_name:str, save_path:str) -> dict:
    """Reads all saved model hyperparameters and returns hyperparameters of specified model.

    Args:
      model_name: Model name used as reference key in saved hyperparameters file
      save_path: Yaml File path containing saved hyperparameters

    Returns:
      Dict containing saved model hyperparameters.

    Raises:
      AssertionError: If no model hyperparameters is not in found saved file.
    """
    if isfile(save_path):
        with open(save_path) as file:
            # FullLoader parameter handles the conversion from YAML scalar values to Python the dictionary format
            all_model_hyperparams = yaml.load(file, Loader=yaml.FullLoader)

    else:
        all_model_hyperparams = {}

    assert model_name in list(all_model_hyperparams.keys()), f"model_name not found in save file. Save filepath: {save_path}"

    return all_model_hyperparams[model_name]

def load_model_param(sb3_model, params:dict):
    """Load tuned hyperparameters into Stable Baseline3 (sb3) model

    Args:
        sb3_model: Stable Baseline3 model
        params: Gym env

    Returns:
        sb3 model with loaded hyperparameters
    """   
    kwargs = params.copy()
    model = sb3_model(**kwargs)
    return model