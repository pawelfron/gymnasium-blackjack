import numpy as np
import torch
from torch import nn
import gymnasium as gym
from tqdm import tqdm
import random
from collections import deque
import pickle

Observation = tuple[int, int, int]
Experience = tuple[Observation, int, Observation, int, bool]

class BlackjackDQ:
    """
    Blackjack Deep Q Learning agent, with an epsilon-greedy policy.
    """

    class BlackjackNetwork(nn.Module):
        """
        Deep neural network, that approximates the values of Q.
        """

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, 10),
                nn.ReLU(),
                nn.Linear(10, 5),
                nn.ReLU(),
                nn.Linear(5, 2)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class BlackjackMemory:
        """
        Memory of past experiences of the agent.
        """

        def __init__(self, max_length: int = 1000):
            self.memory: deque[Experience] = deque([], maxlen=max_length)

        def append(self, transition: Experience):
            self.memory.append(transition)

        def get_sample(self, sample_size: int = 10) -> list[Experience]:
            return random.sample(self.memory, sample_size)

        def __len__(self):
            return len(self.memory)

    def __init__(self, env: gym.Env, learning_rate: float = 0.01, discount_factor: float = 1.0, initial_epsilon: float = 0.0, final_epsilon: float = 0.0, epsilon_decay: float = 0.0, batch_size: int = 32, sync_rate: int = 10):
        """
        :param env: The gymnasium Mountain Car Continuous environment.
        :param learning_rate: The learning rate of the agent.
        :param discount_factor: The discount factor of the agent.
        :param initial_epsilon: The inital probability to choose a random action, instead of the recommended one.
        :param final_epsilon: The final value of the epsilon parameter.
        :param epsilon_decay: The value with which the epsilon parameter is decreased during training.
        :param batch_size: The number of samples from memory that are passed to the optimizer each episode.
        :param sync_rate: The number of episodes until the policy network is synced with the target network.
        """
        self.env = env
        self.observation_space_n = 3
        self.action_space_n = 2

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.sync_rate = sync_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = self.BlackjackNetwork()
        self.policy_net.to(self.device)
        self.target_net = self.BlackjackNetwork()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.to(self.device)
 
        self.memory = self.BlackjackMemory(max_length=1000)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.reward_history: list[int] = []

        self.trained = False

    def train(self, num_episodes: int = 1000):
        """
        Trains the agent.

        :param num_episodes: The number of training episodes.
        """
        # Reset the trained parameters, if the agent was already trained
        if self.trained:
            self.policy_net = self.BlackjackNetwork()
            self.target_net = self.BlackjackNetwork()
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
            self.experience = self.BlackjackMemory(max_length=1000)
            self.reward_history = []

        for i in tqdm(range(num_episodes)):
            # Run the episode
            observation, _ = self.env.reset()
            done = False
            while not done:
                action = self._policy(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                self.memory.append((observation, action, next_observation, reward, terminated))

                observation = next_observation
                done = terminated or truncated

            self.reward_history.append(reward)
            self._decay_epsilon()

            if len(self.memory) < self.batch_size:
                continue

            # Evaluate the model on randomly selected memories
            batch = self.memory.get_sample(self.batch_size)
            self._optimize(batch)

            # Sync the target and policy networks
            if i % self.sync_rate == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_action(self, observation: Observation) -> int:
        """
        Choose the recommended action, given the state.

        :param observation: The state of the game.
        :returns: The recommended action: 0 if 'STICK', 1 if 'HIT'.
        """
        with torch.no_grad():
            value = torch.tensor(observation, dtype=torch.float, device=self.device)
            return self.policy_net(value).argmax().item()

    def _policy(self, observation: Observation) -> int:
        """
        With the probability of epsilon, choose a random action, otherwise choose the recommended one.

        :param observation: The state of the game.
        :returns: The recommended action: 0 if 'STICK', 1 if 'HIT'.
        """
        if np.random.random() > self.epsilon:
            return self.get_action(observation)
        else:
            return self.env.action_space.sample()

    def _optimize(self, batch: list[Experience]):
        """
        Optimize the policy and target networks.

        :param batch: Samples from memory the optimizer works on.
        """
        # Move most of the stuff to the gpu
        observation_batch = torch.tensor([obs for obs, *_ in batch], dtype=torch.float, device=self.device)
        action_batch = [act for _, act, *_ in batch]
        next_observation_batch = torch.tensor([next_obs for _, _, next_obs, *_ in batch], dtype=torch.float, device=self.device)
        reward_batch = torch.tensor([reward for *_, reward, _ in batch], device=self.device)
        terminated_batch = torch.tensor([ter for *_, ter in batch], device=self.device)

        policy_values = self.policy_net(observation_batch)
        target_values = self.policy_net(observation_batch)

        # This is:
        #   reward if terminated else reward + discount_factor * max(target_net(next_observation))
        # but vectorized
        with torch.no_grad():
            next_target_values, _ = self.target_net(next_observation_batch).max(axis=1)
        target = terminated_batch * reward_batch + (~terminated_batch) * (reward_batch + self.discount_factor * next_target_values)

        # Substitute the predicted values for the selected action
        for i, action in enumerate(action_batch):
            target_values[i, action] = target[i]

        # Optimize
        self.optimizer.zero_grad()
        loss = self.loss_fn(policy_values, target_values)
        loss.backward()
        self.optimizer.step()

    def _decay_epsilon(self):
        """
        Descrease the value of epsilon, by epsilon_decay.
        """
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_decay

    def save_model(self, filename: str):
        """
        Save the policy network.

        :param filename: File to save the model weights.
        """
        torch.save(self.policy_net.state_dict(), filename)
    
    def load_model(self, filename: str):
        """
        Load the policy network.

        :param filename: File with the model weights.
        """
        self.policy_net.load_state_dict(torch.load(filename, weights_only=True))
        self.trained = True
    
    def save_reward_history(self, filename: str):
        """
        Save the reward history as a numpy array.

        :param filename: File to save the reward history.
        """
        with open(filename, 'wb') as file:
            pickle.dump(np.array(self.reward_history), file)
