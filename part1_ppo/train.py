import argparse
import os
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Reward wrappers (assumed to exist in RewardWrapper.py)
def make_env(mode='sparse', seed=None):
    def _init():
        env = gym.make("FetchPickAndPlace-v4")
        if seed is not None:
            env.reset(seed=seed)
        if mode == 'sparse':
            from RewardWrapper import RewardWrapperSparse
            env = RewardWrapperSparse(env)
        elif mode == 'dense':
            from RewardWrapper import RewardWrapperDense
            env = RewardWrapperDense(env)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return env
    return _init

def create_model(algo, env, seed, tensorboard_log):
    if algo == "PPO":
        return PPO(
            "MultiInputPolicy", env, verbose=1, seed=seed, tensorboard_log=tensorboard_log,
            learning_rate=1e-4, n_steps=2048, batch_size=512, n_epochs=10,
            gamma=0.98, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256, 256])
        )
    elif algo == "SAC":
        return SAC(
            "MultiInputPolicy", env, verbose=1, seed=seed, tensorboard_log=tensorboard_log,
            learning_rate=3e-4, buffer_size=1_000_000, learning_starts=10000,
            batch_size=256, tau=0.005, gamma=0.98, train_freq=1, gradient_steps=1,
            ent_coef='auto',
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

def main():
    parser = argparse.ArgumentParser(description="Train PPO or SAC on FetchPickAndPlace task.")
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC"],
                        help="Reinforcement learning algorithm to use (PPO or SAC).")
    parser.add_argument("--mode", type=str, default="sparse", choices=["sparse", "dense"],
                        help="Reward type: 'sparse' (binary success/failure) or 'dense' (shaped reward).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="List of random seeds for training (each seed runs independently).")
    parser.add_argument("--timesteps", type=int, default=6_000_000,
                        help="Total number of environment timesteps to train for each seed.")
    parser.add_argument("--n_envs", type=int, default=2,
                        help="Number of parallel environments (vectorised). Use 1 for SAC usually.")
    parser.add_argument("--model_name", type=str, default="PickPlace",
                        help="Base name for saved model files (seed will be appended).")
    parser.add_argument("--folder_name", type=str, default="trained_models",
                        help="Root directory to save trained models and VecNormalize files.")
    args = parser.parse_args()

    gym.register_envs(gymnasium_robotics)

    # Output directory: trained_models/PPO/sparse/  or  trained_models/SAC/dense/
    model_dir = os.path.join(args.folder_name, args.algo, args.mode)
    os.makedirs(model_dir, exist_ok=True)

    # TensorBoard and checkpoint dirs
    tb_log_dir = f"./tensorboard/{args.algo}/{args.mode}"   # ← keep nested structure
    checkpoint_dir = f"./checkpoints/{args.algo}_{args.mode}"
    os.makedirs(checkpoint_dir, exist_ok=True)


    print(f"===== Training {args.algo} with {args.mode} reward =====")
    print(f"Models saved to: {model_dir}")
    print(f"TensorBoard logs: {tb_log_dir}")
    print(f"Checkpoints: {checkpoint_dir}")

    for idx, seed in enumerate(args.seeds):
        print(f"\n--- Run {idx+1}/{len(args.seeds)} | Seed {seed} | {args.algo} ---")

        # Create vectorised environment (one process with n_envs copies)
        vec_env = make_vec_env(
            make_env(mode=args.mode, seed=seed),
            n_envs=args.n_envs,
            seed=seed,
            vec_env_cls=DummyVecEnv
        )

        # Normalise observations (always). Reward norm only for on‑policy (PPO)
        norm_reward = (args.algo == "PPO")
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=norm_reward,
            clip_obs=10.0,
            clip_reward=10.0 if norm_reward else None
        )

        # Checkpoint callback (different save frequency for each algo)
        if args.algo == "PPO":
            save_freq = max(500_000 // args.n_envs, 1)
        else:  # SAC
            save_freq = max(200_000 // args.n_envs, 1)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=checkpoint_dir,
            name_prefix=f"{args.algo}_seed_{seed}",
            save_vecnormalize=True
        )

        # Create and train model
        model = create_model(args.algo, vec_env, seed, tb_log_dir)
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            tb_log_name=f"seed_{seed}"
        )

        # Save final model + VecNormalize
        model.save(os.path.join(model_dir, f"{args.model_name}_seed_{seed}"))
        vec_env.save(os.path.join(model_dir, f"vecnorm_seed_{seed}.pkl"))

        vec_env.close()

    print(f"\nAll {len(args.seeds)} runs finished successfully.")

if __name__ == "__main__":
    main()

# import argparse
# import os
# import gymnasium as gym
# import gymnasium_robotics
# from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
# from stable_baselines3.common.callbacks import CheckpointCallback


# # Reward wrappers
# def make_env(mode='sparse', seed=None):
#     """Environment factory with reward wrapper."""
#     def _init():
#         env = gym.make("FetchPickAndPlace-v4")
#         if seed is not None:
#             env.reset(seed=seed)
#         if mode == 'sparse':
#             from RewardWrapper import RewardWrapperSparse
#             env = RewardWrapperSparse(env)
#         elif mode == 'dense':
#             from RewardWrapper import RewardWrapperDense
#             env = RewardWrapperDense(env)
#         else:
#             raise ValueError(f"Unknown mode: {mode}")
#         return env
#     return _init


# # Algorithm‑specific model creation
# def create_model(algo, env, seed, tensorboard_log):
#     """Create and return a PPO or SAC model with tuned hyperparameters."""
#     if algo == "PPO":
#         return PPO(
#             "MultiInputPolicy",
#             env,
#             verbose=1,
#             seed=seed,
#             tensorboard_log=tensorboard_log,
#             # PPO‑specific settings (already tuned)
#             learning_rate=1e-4,
#             n_steps=2048,
#             batch_size=512,
#             n_epochs=10,
#             gamma=0.98,
#             gae_lambda=0.95,
#             clip_range=0.2,
#             ent_coef=0.01,
#             vf_coef=0.5,
#             max_grad_norm=0.5,
#             policy_kwargs=dict(net_arch=[256, 256, 256])
#         )
#     elif algo == "SAC":
#         return SAC(
#             "MultiInputPolicy",
#             env,
#             verbose=1,
#             seed=seed,
#             tensorboard_log=tensorboard_log,
#             # SAC settings (off‑policy, dense reward recommended)
#             learning_rate=3e-4,
#             buffer_size=1_000_000,
#             learning_starts=10000,
#             batch_size=256,
#             tau=0.005,
#             gamma=0.98,
#             train_freq=1,
#             gradient_steps=1,
#             ent_coef='auto',
#             policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
#         )
#     else:
#         raise ValueError(f"Unsupported algorithm: {algo}")

# # Main training routine
# def main():
#     parser = argparse.ArgumentParser(description="Train PPO or SAC on FetchPickAndPlace")
#     parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC"],
#                         help="Reinforcement learning algorithm")
#     parser.add_argument("--mode", type=str, default="sparse", choices=["sparse", "dense"],
#                         help="Reward type: 'sparse' (binary success) or 'dense' (shaped)")
#     parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
#                         help="List of random seeds to train")
#     parser.add_argument("--timesteps", type=int, default=6_000_000,
#                         help="Total training timesteps per seed")
#     parser.add_argument("--n_envs", type=int, default=2,
#                         help="Number of parallel environments")
#     parser.add_argument("--model_name", type=str, default="PickPlace",
#                         help="Base name for saved models")
#     parser.add_argument("--folder_name", type=str, default="trained_models",
#                         help="Root folder to save models and normalizers")
#     args = parser.parse_args()

#     # Register gymnasium robotics environments
#     gym.register_envs(gymnasium_robotics)

#     # Build dynamic output paths: e.g., trained_models/PPO/sparse/
#     model_dir = os.path.join(args.folder_name, args.algo, args.mode)
#     os.makedirs(model_dir, exist_ok=True)

#     # Log directories (TensorBoard and checkpoints)
#     tb_log_dir = f"./tensorboard/{args.algo.lower()}_{args.mode}"
#     checkpoint_dir = f"./checkpoints/{args.algo}_{args.mode}"
#     os.makedirs(checkpoint_dir, exist_ok=True)

#     print(f"===== Training {args.algo} with {args.mode} reward =====")
#     print(f"Models will be saved in: {model_dir}")
#     print(f"TensorBoard logs: {tb_log_dir}")
#     print(f"Checkpoints: {checkpoint_dir}")

#     for idx, seed in enumerate(args.seeds):
#         print(f"\n--- Run {idx+1}/{len(args.seeds)} | Seed {seed} | {args.algo} ---")

#         # 1. Create vectorised environment (with wrapper, fixed seed for reproducibility)
#         vec_env = make_vec_env(
#             make_env(mode=args.mode, seed=seed),
#             n_envs=args.n_envs,
#             seed=seed,
#             vec_env_cls=DummyVecEnv
#         )

#         # 2. Observation normalisation (always on), reward normalisation only for on‑policy
#         norm_reward = (args.algo == "PPO")   # PPO benefits, SAC usually keeps raw reward
#         vec_env = VecNormalize(
#             vec_env,
#             norm_obs=True,
#             norm_reward=norm_reward,
#             clip_obs=10.0,
#             clip_reward=10.0 if norm_reward else None
#         )

#         # 3. Checkpoint callback (saves model + VecNormalize stats)
#         save_freq = max(500_000 // args.n_envs, 1) if args.algo == "PPO" else 200_000 // args.n_envs
#         checkpoint_callback = CheckpointCallback(
#             save_freq=save_freq,
#             save_path=checkpoint_dir,
#             name_prefix=f"{args.algo}_seed_{seed}",
#             save_vecnormalize=True
#         )

#         # 4. Create model
#         model = create_model(
#             algo=args.algo,
#             env=vec_env,
#             seed=seed,
#             tensorboard_log=tb_log_dir,
#             mode=args.mode
#         )

#         # 5. Train
#         model.learn(
#             total_timesteps=args.timesteps,
#             callback=checkpoint_callback,
#             tb_log_name=f"seed_{seed}"
#         )

#         # 6. Save final model and VecNormalize
#         model.save(os.path.join(model_dir, f"{args.model_name}_seed_{seed}"))
#         vec_env.save(os.path.join(model_dir, f"vecnorm_seed_{seed}.pkl"))

#         # Cleanup
#         vec_env.close()

#     print(f"\nAll {len(args.seeds)} runs finished successfully.")

# if __name__ == "__main__":
#     main()