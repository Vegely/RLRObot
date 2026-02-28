import os
import gymnasium as gym
import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

# ==========================================
# 1. OPTUNA PRUNING CALLBACK
# ==========================================
class TrialEvalCallback(EvalCallback):
    """
    Evaluates the model periodically and stops the trial early 
    if the hyperparameters are underperforming.
    """
    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=10000):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=True,
            verbose=0
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            # Report intermediate reward to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Check if Optuna wants to cancel this trial
            if self.trial.should_prune():
                self.is_pruned = True
                return False # Returning False instantly stops model.learn()
        return True

# ==========================================
# 2. OBJECTIVE FUNCTION
# ==========================================
def objective(trial):
    # OPTIMIZED SEARCH SPACE: Removed bloated/harmful values
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512]) 
    net_arch_width = trial.suggest_categorical("net_arch_width", [128, 256, 300, 400])
    tau = trial.suggest_float("tau", 0.001, 0.05, log=True)
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.995, 0.999]) 

    policy_kwargs = dict(net_arch=[net_arch_width, net_arch_width])

    # Log directories
    trial_name = f"datatraining_trial_{trial.number}"
    log_dir = f"./{trial_name}/"
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "hyperparameters.txt"), "w") as f:
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")

    # Environments
    env = gym.make("BipedalWalker-v3")
    eval_env = gym.make("BipedalWalker-v3") # Dedicated env for the callback

    model = SAC(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        policy_kwargs=policy_kwargs,
        verbose=0 
    )

    new_logger = configure(log_dir, ["csv"])
    model.set_logger(new_logger)

    # Initialize the pruner callback (evaluates every 10,000 steps)
    eval_callback = TrialEvalCallback(eval_env, trial, n_eval_episodes=5, eval_freq=10_000)

    try:
        model.learn(
            total_timesteps=50_000, 
            callback=eval_callback, 
            progress_bar=True
        )
    except Exception as e:
        return -1000.0

    # If the callback flipped this switch to True, tell Optuna it was pruned!
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    env.close()
    eval_env.close()

    # Return the last evaluation score
    return eval_callback.last_mean_reward

# ==========================================
# 3. RUN THE STUDY
# ==========================================
def tune_hyperparameters():
    print("Starting Optuna search with Median Pruning...")
    
    # We specify a Pruner here! The MedianPruner is the standard choice.
    # n_startup_trials=5 means it waits to gather data from 5 trials before it starts killing bad ones.
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    )
    
    # Dropped to 100 trials - with pruning and optimized batch sizes, this should take just a few hours.
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    summary_text = (
        "\n==================================\n"
        "Optimization Finished!\n"
        f"Best Value: {study.best_trial.value}\n"
        f"Best Params: {study.best_trial.params}\n"
        "==================================\n"
    )

    print(summary_text)

    with open("best_hyperparameters_summary.txt", "w") as file:
        file.write(summary_text)

if __name__ == "__main__":
    tune_hyperparameters()