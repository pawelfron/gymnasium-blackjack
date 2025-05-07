from collections import defaultdict
import numpy as np
import gymnasium as gym
from tqdm import tqdm

class MountainCarQ:
    """
    Classical Q-Learning agent for the Mountain Car Continuous environment
    """
    def __init__(self, env: gym.Env, learning_rate: float = 0.01, discount_factor: float = 0.9, initial_epsilon: float = 1.0, final_epsilon: float = 0.1, epsilon_decay: float = 0.001, position_bins: int = 1000, velocity_bins: int = 100, action_bins: int = 40):
        """
        :param env: The gymnasium Mountain Car Continuous environment.
        :param learning_rate: The learning rate of the agent.
        :param discount_factor: The discount factor of the agent.
        :param initial_epsilon: The inital probability to choose a random action, instead of the recommended one.
        :param final_epsilon: The final value of the epsilon parameter.
        :param epsilon_decay: The value with which the epsilon parameter is decreased during training.
        :param position_bins: The amount of bins the position of the car is discretized into.
        :param velocity_bins: The amount of bins the velocity of the car is discretized into.
        :param action_bins: The amount of bins the force (action) is discretized into.
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        ACTION_LEFT_BOUND = -1
        ACTION_RIGHT_BOUND = 1
        POSITION_LEFT_BOUND = -1.2
        POSITION_RIGHT_BOUND = 0.6
        VELOCITY_LEFT_BOUND = -0.07
        VELOCITY_RIGHT_BOUND = 0.07

        self.possible_actions = np.linspace(ACTION_LEFT_BOUND, ACTION_RIGHT_BOUND, action_bins)
        self.possible_positions = np.linspace(POSITION_LEFT_BOUND, POSITION_RIGHT_BOUND, position_bins)
        self.possible_velocities = np.linspace(VELOCITY_LEFT_BOUND, VELOCITY_RIGHT_BOUND, velocity_bins)

        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay

        self.Q = defaultdict(lambda: np.zeros(action_discretization, dtype=np.float64))

        self.trained = False

        self.reward_history = []

    def _decay_epsilon(self):
        """
        Decreases the epsilon value by epsilon_decay parameter.
        """
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_decay

    def _action_selection(self, observation):
        """
        With the probability of epsilon, selects a random action, otherwise selects the recommended one.

        :param observation: The current state of the game.
        :returns: The selected action.
        """
        if np.random.random() > self.epsilon:
            return self.get_action(observation)
        else:
            return self.possible_actions[np.digitize(self.env.action_space.sample(), self.possible_actions)]

    def _update_q(self, observation, action, next_observation, reward):
        """
        Updates the Q table.
        """
        action_arg = np.digitize(action, self.possible_actions, right=True)
        action_arg = np.clip(action_arg, 0, len(self.possible_actions) - 1)

        obs_arg = (
            np.clip(np.digitize(observation[0], self.possible_positions, right=True), 0, len(self.possible_positions) - 1),
            np.clip(np.digitize(observation[1], self.possible_velocities, right=True), 0, len(self.possible_velocities) - 1)
        )

        next_obs_arg = (
            np.clip(np.digitize(next_observation[0], self.possible_positions, right=True), 0, len(self.possible_positions) - 1),
            np.clip(np.digitize(next_observation[1], self.possible_velocities, right=True), 0, len(self.possible_velocities) - 1)
        )

        self.Q[obs_arg][action_arg] = (
            self.Q[obs_arg][action_arg] + self.learning_rate * (
                reward + self.discount_factor * np.max(self.Q[next_obs_arg]) - self.Q[obs_arg][action_arg]
            )
        )

    def train(self, num_episodes: int):
        """
        Trains the agent.

        :param num_episodes: The number of episodes for training.
        """
        # Reset the trained parameters, if the agent was already trained
        if self.trained:
            self.Q = defaultdict(lambda: np.zeros(self.n))
            self.reward_history = []

        for _ in tqdm(range(num_episodes)):
            observation, _ = self.env.reset()

            total_reward = 0
            done = False
            while not done:
                action = self._action_selection(observation)

                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                self._update_q(observation, action, next_observation, reward)

                done = terminated or truncated
                total_reward += reward
                observation = next_observation

            self.reward_history.append(total_reward)
            self._decay_epsilon()

        self.trained = True

    def get_action(self, observation):
        """
        Get the recommended action in the provided state.

        :param observation: The state of the game, in the format (position, velocity).
        :returns: The force with which the car should be moved.
        """
        obs_arg = (
            np.clip(np.digitize(observation[0], self.possible_positions, right=True), 0, len(self.possible_positions) - 1),
            np.clip(np.digitize(observation[1], self.possible_velocities, right=True), 0, len(self.possible_velocities) - 1)
        )
        return [self.possible_actions[np.argmax(self.Q[obs_arg])]]
