import os
import re
import glob
import subprocess
import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecEnvWrapper

# 1. THE CUSTOM TEXT WRAPPER
# This intercepts the video frames and adds a padded text box at the bottom
class TextOverlayVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv, text_to_display=""):
        super().__init__(venv)
        self.text_to_display = text_to_display

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        return self.venv.step_wait()

    def render(self, mode="rgb_array"):
        frame = self.venv.render(mode=mode)
        # If we are recording a video frame, intercept it and add the text box!
        if mode == "rgb_array" and frame is not None:
            h, w, c = frame.shape
            pad_h = 60 # Height of the text box in pixels
            
            # Create a new, taller image frame
            new_frame = np.zeros((h + pad_h, w, c), dtype=np.uint8)
            new_frame[:h, :, :] = frame # Put the original video on top
            new_frame[h:, :, :] = (30, 30, 30) # Dark gray box on the bottom
            
            # Draw the text inside the dark gray box
            cv2.putText(
                new_frame, 
                self.text_to_display, 
                (20, h + 40), # X, Y coordinates for the text
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, # Font size
                (255, 255, 255), # White text (B, G, R)
                2, # Line thickness
                cv2.LINE_AA
            )
            return new_frame
        return frame


# 2. THE SIMULATION & RECORDING FUNCTION
# 2. THE SIMULATION & RECORDING FUNCTION
def simulate_and_record(model_path, video_folder, output_name, step_count, num_envs=4, video_length=1000):
    print(f"\nSimulating model: {step_count} steps...")
    model = SAC.load(model_path)

    # CHANGED: Using DummyVecEnv instead of SubprocVecEnv to prevent Windows from freezing!
    vec_env = make_vec_env(
        "BipedalWalker-v3", 
        n_envs=num_envs, 
        vec_env_cls=DummyVecEnv, 
        env_kwargs={"render_mode": "rgb_array"} 
    )
    
    # Wrap the environment to add our custom text box
    display_text = f"BipedalWalker | Training Progress: {step_count:,} Steps"
    vec_env = TextOverlayVecEnvWrapper(vec_env, text_to_display=display_text)
    
    # Wrap it again to record the video
    os.makedirs(video_folder, exist_ok=True)
    vec_env = VecVideoRecorder(
        vec_env, 
        video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=output_name
    )
    
    obs = vec_env.reset()
    for i in range(video_length + 1):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        
        # ADDED: Print progress every 100 frames so you know it is actually working
        if i % 100 == 0 and i > 0:
            print(f"  ... recorded {i}/{video_length} frames")

    vec_env.close()

# 3. THE STITCHING FUNCTION
def stitch_videos_lossless(video_dir, output_filename="full_learning_timelapse.mp4"):
    print("\nStitching all 60 videos together...")
    search_pattern = os.path.join(video_dir, "progression_*.mp4")
    video_files = glob.glob(search_pattern)
    
    def extract_step(filepath):
        match = re.search(r'progression_(\d+)_steps', filepath)
        return int(match.group(1)) if match else 0
        
    video_files.sort(key=extract_step)
    
    list_file_path = os.path.join(video_dir, "ffmpeg_concat_list.txt")
    with open(list_file_path, "w") as f:
        for video in video_files:
            f.write(f"file '{video.replace('\\', '/')}'\n")

    output_path = os.path.join(video_dir, output_filename)
    command = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
        "-i", list_file_path, "-c", "copy", output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"\nSUCCESS! Master video ready at: {output_path}")
    except subprocess.CalledProcessError:
        print("\nError running FFmpeg.")
        
    if os.path.exists(list_file_path):
        os.remove(list_file_path)

# 4. THE MAIN EXECUTION BLOCK
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, "saved_models", "checkpoints")
    video_output_dir = os.path.join(script_dir, "simulation_videos")
    
    # Find all 60 checkpoints
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    
    def get_step(filepath):
        match = re.search(r'_(\d+)_steps', filepath)
        return int(match.group(1)) if match else 0
        
    checkpoint_files.sort(key=get_step)
    
    # Generate an MP4 for each checkpoint
    for model_path in checkpoint_files:
        step_count = get_step(model_path)
        output_filename = f"progression_{step_count}_steps"
        
        simulate_and_record(
            model_path=model_path, 
            video_folder=video_output_dir, 
            output_name=output_filename,
            step_count=step_count, # Pass the step count to our function!
            num_envs=4, 
            video_length=500 # Kept short so processing 60 models doesn't take hours
        )
        
    # Once all 60 are recorded, stitch them immediately
    stitch_videos_lossless(video_output_dir)