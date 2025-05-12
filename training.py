import gymnasium as gym
from agents import BlackjackMC, MountainCarQ, BlackjackDQ
import os

models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)
history_dir = 'history'
os.makedirs(history_dir, exist_ok=True)

# Blackjack - Monte Carlo Q Learning
env = gym.make("Blackjack-v1", sab=True)
agent = BlackjackMC(env, epsilon=0.1, discount_factor=0.9)
agent.train(100000)
agent.save_model(os.path.join(models_dir, 'mc_blackjack.pkl'))
agent.save_reward_history(os.path.join(history_dir, 'mc_blackjack_history.pkl'))
env.close()

# Mountain Car Continous - Q Learning (gamma=1.0)
env = gym.make("MountainCarContinuous-v0")
agent = MountainCarQ(env, learning_rate=0.01, discount_factor=1.0, position_bins=10, velocity_bins=5, action_bins=10, initial_epsilon=1.0, final_epsilon=0.1, epsilon_decay=0.0001)
agent.train(10000)
agent.save_model(os.path.join(models_dir, 'q_mountain_car.pkl'))
agent.save_reward_history(os.path.join(history_dir, 'q_mountain_car_history.pkl'))
env.close()

# Mountain Car Continous - Q Learning (gamma=0.7)
env = gym.make("MountainCarContinuous-v0")
agent = MountainCarQ(env, learning_rate=0.01, discount_factor=0.7, position_bins=10, velocity_bins=5, action_bins=10, initial_epsilon=1.0, final_epsilon=0.1, epsilon_decay=0.0001)
agent.train(10000)
agent.save_model(os.path.join(models_dir, 'q_mountain_car_2.pkl'))
agent.save_reward_history(os.path.join(history_dir, 'q_mountain_car_2_history.pkl'))
env.close()

# Mountain Car Continous - Q Learning (gamma=0.4)
env = gym.make("MountainCarContinuous-v0")
agent = MountainCarQ(env, learning_rate=0.01, discount_factor=0.4, position_bins=10, velocity_bins=5, action_bins=10, initial_epsilon=1.0, final_epsilon=0.1, epsilon_decay=0.0001)
agent.train(10000)
agent.save_model(os.path.join(models_dir, 'q_mountain_car_3.pkl'))
agent.save_reward_history(os.path.join(history_dir, 'q_mountain_car_3_history.pkl'))
env.close()

# Blackjack - Deep Q Learning
env = gym.make("Blackjack-v1", sab=True)
agent = BlackjackDQ(env, learning_rate=0.01, discount_factor=0.9, initial_epsilon=1.0, final_epsilon=0.1, epsilon_decay=0.00001)
agent.train(100000)
agent.save_model(os.path.join(models_dir, 'deep_q_blackjack.pt'))
agent.save_reward_history(os.path.join(history_dir, 'deep_q_blackjack_history.pkl'))
env.close()
