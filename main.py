import os
import shutil
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import osm_env
import matplotlib.pyplot as plt


def train():
    """Train PPO on the OSM environment and save the final learning curve."""
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    env = Monitor(gym.make("osm_map"), log_dir)                                         # no render during training for speed
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir
    )
    
    Timesteps = 10000

    for iters in range(1, 51):                                                         # train for 50 iterations with 10000 timesteps each
        model.learn(total_timesteps=Timesteps, reset_num_timesteps=False)
        model.save(os.path.join(model_dir, f"ppo_walking_loop-{iters}"))
    
    print(f"\nTraining complete")
    print(f"  Final model saved to: {model_dir}/ppo_walking_loop-50.zip")
    
    results = load_results(log_dir)
    x, y = ts2xy(results, "timesteps")
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Return")
    plt.title("Learning Curve")
    plt.grid(True)
    plt.savefig("learning_curve.png")
    plt.show()



def test(duration=None):
    """Load the trained model and generate closed walkable path."""
    env = gym.make("osm_map", render_mode="human", duration=duration)
    model = PPO.load('models/ppo_walking_loop-50', env=env)
    
    obs = env.reset()[0]
    
    while True:
        action, _ = model.predict(obs, deterministic=True)                  
        obs, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            print(f"\n--- Episode finished ---")
            print(f"  Target duration: {env.unwrapped.target_duration:.1f} min")
            print(f"  Actual duration: {env.unwrapped.actual_duration:.1f} min")
            env.unwrapped.render()                                                                  # display the loop on the map
            break
        


        
if __name__ == "__main__":
    # train()                                                                                         # uncomment this line to train a model
    duration = float(input("Enter desired walk duration in minutes (5-50): "))
    if duration < 5 or duration > 50:
        print("Invalid duration. Please enter a value between 5 and 50.")
        exit()
    test(duration)
