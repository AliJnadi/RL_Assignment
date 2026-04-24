import argparse
import os
import numpy as np
import imageio
import re
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env():
    def _init():
        return gym.make("FetchPickAndPlace-v4", render_mode="rgb_array")
    return _init

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO/SAC models on FetchPickAndPlace.")
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC"],
                        help="Algorithm used for training (must match saved models).")
    parser.add_argument("--mode", type=str, default="sparse", choices=["sparse", "dense"],
                        help="Reward type used during training (must match saved models).")
    parser.add_argument("--folder_name", type=str, default="trained_models",
                        help="Root directory where models are saved (e.g., trained_models).")
    parser.add_argument("--model_name", type=str, default="PickPlace",
                        help="Base name of the saved model (seed will be appended automatically).")
    parser.add_argument("--specific_seed", type=int, default=None,
                        help="If provided, evaluate only that specific seed. Otherwise, evaluate all seeds found.")
    args = parser.parse_args()

    # Build path to model directory: trained_models/PPO/sparse/
    model_dir = os.path.join(args.folder_name, args.algo, args.mode)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Find all model files (ending with .zip) that match the pattern
    pattern = re.compile(rf"{args.model_name}_seed_(\d+)\.zip")
    model_files = []
    for f in os.listdir(model_dir):
        m = pattern.match(f)
        if m:
            seed = int(m.group(1))
            if args.specific_seed is None or seed == args.specific_seed:
                model_files.append((seed, os.path.join(model_dir, f)))

    if not model_files:
        raise FileNotFoundError(f"No model files matching pattern in {model_dir}")

    gym.register_envs(gymnasium_robotics)

    print(f"\n===== Evaluating {args.algo} ({args.mode} reward) =====")
    deterministic_success_rates = []
    nondet_success_rates = []

    # GIF output folder
    gif_folder = os.path.join("gifs", args.algo, args.mode)
    os.makedirs(gif_folder, exist_ok=True)

    for seed, model_path in sorted(model_files):
        print(f"\n--- Evaluating seed {seed} ---")

        # Create environment
        vec_env = DummyVecEnv([make_env()])

        # Load VecNormalize if exists
        norm_path = os.path.join(model_dir, f"vecnorm_seed_{seed}.pkl")
        if os.path.exists(norm_path):
            vec_env = VecNormalize.load(norm_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
        else:
            print(f"Warning: No VecNormalize file found for seed {seed}. Continuing without normalisation.")

        # Load model
        if args.algo == "PPO":
            from stable_baselines3 import PPO
            model = PPO.load(model_path, env=vec_env)
        else:  # SAC
            from stable_baselines3 import SAC
            model = SAC.load(model_path, env=vec_env)

        # Deterministic evaluation (100 episodes)
        successes = []
        for _ in range(100):
            obs = vec_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, info = vec_env.step(action)
                if done:
                    successes.append(info[0]["is_success"])
        det_rate = np.mean(successes)
        deterministic_success_rates.append(det_rate)
        print(f" Deterministic success rate: {det_rate:.3f}")

        # Non-deterministic evaluation (100 episodes)
        successes = []
        for _ in range(100):
            obs = vec_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, _, done, info = vec_env.step(action)
                if done:
                    successes.append(info[0]["is_success"])
        nondet_rate = np.mean(successes)
        nondet_success_rates.append(nondet_rate)
        print(f" Non-deterministic success rate: {nondet_rate:.3f}")

        # Record a GIF (5 episodes, max 100 steps each)
        frames = []
        for ep in range(5):
            obs = vec_env.reset()
            for step in range(100):
                frame = vec_env.render()
                frames.append(frame)
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = vec_env.step(action)
                if done:
                    break
        gif_path = os.path.join(gif_folder, f"seed_{seed}.gif")
        imageio.mimsave(gif_path, frames, fps=20)
        print(f" GIF saved: {gif_path}")

        vec_env.close()

    print("\n------------------------------------------")
    print(f"Overall deterministic success rate: {np.mean(deterministic_success_rates):.2f} ± {np.std(deterministic_success_rates):.2f}")
    print(f"Overall non-deterministic success rate: {np.mean(nondet_success_rates):.2f} ± {np.std(nondet_success_rates):.2f}")
    print("Evaluation finished.")

if __name__ == "__main__":
    main()

# import argparse
# import os
# import random
# import numpy as np
# import imageio
# import re

# import gymnasium as gym
# import gymnasium_robotics
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# def make_env():
#     def _init():
#         env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array")
#         return env
#     return _init

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--algo", type=str, default="PPO")
#     parser.add_argument("--folder_name", type=str, default="PPO_models")
#     parser.add_argument("--mode", type=str, default="sparse", choices=["sparse", "dense"])
#     parser.add_argument("--model_name", type=str, default=None)
#     parser.add_argument("--n_envs", type=int, default=1)
#     args = parser.parse_args()

#     model_dir = f"{args.folder_name}/{args.mode}"
#     model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".zip")])

#     if args.model_name:
#         model_files = [f for f in model_files if args.model_name in f]
#         if not model_files:
#             raise FileNotFoundError(f"No model file containing {args.model_name} found")

#     gym.register_envs(gymnasium_robotics)

#     print("\n===== EVALUATING EACH SEED =====")
#     deterministic = []
#     non_deterministic = []

#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     gif_folder = os.path.join(script_dir, "gifs", model_dir)
#     os.makedirs(gif_folder, exist_ok=True)

#     for model_file in model_files:
#         model_path = os.path.join(model_dir, model_file)
#         match = re.search(r'\_(\d+)\.', model_path)
#         seed = int(match.group(1)) if match else 0

#         # Create environment
#         vec_env = DummyVecEnv([make_env()])

#         # Load VecNormalize if exists
#         vecnorm_path = f"{model_dir}/vecnorm_seed_{seed}.pkl"
#         if os.path.exists(vecnorm_path):
#             vec_env = VecNormalize.load(vecnorm_path, vec_env)
#             vec_env.training = False
#             vec_env.norm_reward = False
#         else:
#             print(f"Warning: {vecnorm_path} not found, using raw env.")

#         # Load model
#         if args.algo == "PPO":
#             from stable_baselines3 import PPO
#             model = PPO.load(model_path, env=vec_env)
#         else:
#             from stable_baselines3 import SAC
#             model = SAC.load(model_path, env=vec_env)

#         # Deterministic evaluation
#         success = []
#         for _ in range(100):
#             obs = vec_env.reset()
#             done = False
#             while not done:
#                 action, _ = model.predict(obs, deterministic=True)
#                 obs, _, done, info = vec_env.step(action)
#                 if done:
#                     success.append(info[0]["is_success"])
#         print(f"{model_file} deterministic success rate: {np.mean(success):.3f}")
#         deterministic.append(success)

#         # Non‑deterministic evaluation
#         success = []
#         for _ in range(100):
#             obs = vec_env.reset()
#             done = False
#             while not done:
#                 action, _ = model.predict(obs, deterministic=False)
#                 obs, _, done, info = vec_env.step(action)
#                 if done:
#                     success.append(info[0]["is_success"])
#         print(f"{model_file} non‑deterministic success rate: {np.mean(success):.3f}")
#         non_deterministic.append(success)

#         # Record GIF (5 episodes, 50 steps max each)
#         frames = []
#         for _ in range(5):
#             obs = vec_env.reset()
#             frame = vec_env.render()
#             frames.append(frame)
#             for _ in range(50):
#                 action, _ = model.predict(obs, deterministic=True)
#                 obs, _, done, _ = vec_env.step(action)
#                 frame = vec_env.render()
#                 frames.append(frame)
#                 if done:
#                     break
#         gif_path = os.path.join(gif_folder, f"seed_{seed}.gif")
#         imageio.mimsave(gif_path, frames, fps=20)
#         vec_env.close()

#     print("------------------------------------------")
#     print(f"Overall deterministic success: {np.mean(deterministic):.2f}")
#     print(f"Overall non‑deterministic success: {np.mean(non_deterministic):.2f}")
#     print("Evaluation done")

# if __name__ == "__main__":
#     main()