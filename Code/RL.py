import gymnasium as gym
import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

def train_combined_walker():
    # 1. Setup Relative Paths
    # This gets the folder where this specific .py script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create subfolders inside the script's directory
    log_dir = os.path.join(script_dir, "logs_and_csvs")
    model_dir = os.path.join(script_dir, "saved_models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = Monitor(gym.make("BipedalWalker-v3"), filename=os.path.join(log_dir, "train_monitor.csv"))
    eval_env = Monitor(gym.make("BipedalWalker-v3"))

    policy_kwargs = dict(net_arch=[256, 256]) 

    model = SAC(
        "MlpPolicy", 
        env, 
        learning_rate=0.0001965450924118268, 
        buffer_size=300000, 
        batch_size=512,
        ent_coef='auto',          
        gamma=0.995,
        tau=0.038853303295044855,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    new_logger = configure(log_dir, ["stdout", "csv"])
    model.set_logger(new_logger)

    # 2. Setup Eval Callback (Saves the absolute BEST model)
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=os.path.join(model_dir, 'best_model'), 
        log_path=log_dir,
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=True, 
        render=False 
    )

    # 3. Setup Checkpoint Callback (Saves the model every 50,000 steps so you can see the progress)
    checkpoint_callback = CheckpointCallback(
        save_freq=5_000,
        save_path=os.path.join(model_dir, 'checkpoints'),
        name_prefix='sac_bipedal'
    )

    # Combine callbacks
    callback_list = [eval_callback, checkpoint_callback]

    print(f"Starting Training. Saving everything relative to: {script_dir}")
    model.learn(
        total_timesteps=500_000, 
        callback=callback_list, 
        progress_bar=True
    )

    env.close()
    eval_env.close()

if __name__ == "__main__":
    train_combined_walker()