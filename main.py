import gym
import sys
import torch
import numpy as np
import highway_env
import warnings
import time

from arguments import get_args
from ppo import PPO
from network import FeedForwardNN
from eval_policy import eval_policy

warnings.filterwarnings('ignore')


def test_ori():
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["y"],
            # "features": ["presence", "x", "y", "vx", "vy", "cos_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted",
            "longitudinal": False
        },
        "action": {
            'type': 'DiscreteMetaAction'
            # "longitudinal": False
        },
        # "simulation_frequency": 20,  ##  越大仿真速度越慢
        # "policy_frequency": 10,  # 越大，采样越快
        'lanes_count': 4
    }
    env = gym.make('highway-v1')
    # print(env.controlled_vehicles[0].beilv)
    env.configure(config)
    #
    env.reset()
    for i in range(20):
        action = env.action_type.actions_indexes["IDLE"]
        if i == 5:
            action = env.action_type.actions_indexes["LANE_RIGHT"]
        if i == 6:
            action = env.action_type.actions_indexes["LANE_LEFT"]
        if i == 7:
            action = env.action_type.actions_indexes["LANE_RIGHT"]
        if i == 8:
            action = env.action_type.actions_indexes["LANE_LEFT"]
        if i == 9:
            action = env.action_type.actions_indexes["LANE_RIGHT"]
        obs, reward, done, info = env.step(action)
        env.render()
    # continue


# 	action = env.action_type.actions_indexes["IDLE"]
# 	env.controlled_vehicles[0].beilv = 0.8
# 	print("i=", i)
# 	if i == 1:
# 		# print("env.controlled_vehicles[0].KP_LATERAL ",env.controlled_vehicles[0].KP_LATERAL)
# 		action = env.action_type.actions_indexes["LANE_LEFT"]
# 	if i == 4:
# 		action = env.action_type.actions_indexes["LANE_RIGHT"]
# 	obs, reward, done, info = env.step(action)
# 	print("obs", obs[0])
# 	env.render()

def train(env, hyperparameters, actor_model, critic_model):
    """
		Trains the model.

		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training

		Return:
			None
	"""
    print(f"Training", flush=True)

    # Create a model for PPO.
    model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '':  # Don't train from scratch if user accidentally forgets actor/critic model
        print(
            f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    # Train the PPO model with a specified total timesteps
    # NOTE: You can change the total timesteps here, I put a big number just because
    # you can kill the process whenever you feel like PPO is converging
    model.learn(total_timesteps=800_000)


def test(env, actor_model):
    """
		Tests the model.

		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in

		Return:
			None
	"""
    print(f"Testing {actor_model}", flush=True)

    # If the actor model is not specified, then exit
    if actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = 30
    act_dim = 1

    # Build our policy the same way we build our actor model in PPO
    policy = FeedForwardNN(obs_dim, act_dim)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))

    # Evaluate our policy with a separate module, eval_policy, to demonstrate
    # that once we are done training the model/policy with ppo.py, we no longer need
    # ppo.py since it only contains the training algorithm. The model/policy itself exists
    # independently as a binary file that can be loaded in with torch.
    eval_policy(policy=policy, env=env, render=True)


def main(args):
    """
		The main function to run.

		Parameters:
			args - the arguments parsed from command line

		Return:
			None
	"""
    # NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
    # ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
    # To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
    hyperparameters = {
        # 'timesteps_per_batch': 2048,
        # 'max_timesteps_per_episode': 200,
        'timesteps_per_batch': 120,

        'max_timesteps_per_episode': 60,
        'gamma': 0.99,
        'n_updates_per_iteration': 10,
        'lr': 3e-4,
        'clip': 0.2,
        'render': True,
        'render_every_i': 10
    }
    env = gym.make("racetrack-v1")
    env.configure(
        {
            "observation": {
                "type": "OccupancyGrid",
                "features": ['presence', 'on_road'],
                "grid_size": [[-7.5, 7.5], [-7.5, 22.5]],  # 前车左右，后车前后
                "grid_step": [5, 5],  # 每个网格的大小
                "as_image": False,
                "align_to_vehicle_axes": True
            },

            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-10, 10],
                "vy": [-10, 10]},
            # "action": {
            #     "type": "ContinuousAction",
            #     "longitudinal": True,
            #     "lateral": True
            # },
            "simulation_frequency": 15,
            "policy_frequency": 3,
            "duration": 500,
            "collision_reward": -10,
            "lane_centering_cost": 6,
            "action_reward": -0.3,
            "controlled_vehicles": 1,
            "other_vehicles": 5,
            # "screen_width": 600,
            # "screen_height": 600,
            # "centering_position": [0.5, 0.5],
            # "scaling": 7,
            "show_trajectories": False,
            # "render_agent": True,
            # "offscreen_rendering": False
        })
    env.reset()

    # env = gym.make("CartPole-v1")
    # env = gym.make("highway-v0")
    # env = gym.make("Pendulum-v1")

    # Train or test, depending on the mode specified
    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    else:
        test(env=env, actor_model=args.actor_model)


if __name__ == '__main__':
    args = get_args()  # Parse arguments from command line
    # test_ori()
main(args)
