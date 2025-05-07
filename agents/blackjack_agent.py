from collections import defaultdict
import numpy as np
import gymnasium as gym
from tqdm import tqdm

class BlackjackMC:
    """
    Monte Carlo Blackjack agent, with an epsilon-greedy policy.
    """

    def __init__(self, env: gym.Env, epsilon: float = 0.1, discount_factor: float = 0.9):
        """
        :param env: The gymnasium Blackjack environment.
        :param epsilon: Parameter influencing the probability to choose a random action during training, instead of the recommend one.
        :param discount_factor: The discount factor of the agent.
        """
        self.env = env
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.n = env.action_space.n

        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)
        self.Q = defaultdict(lambda: np.zeros(self.n))

        self.trained = False

        self.reward_history = []

    def _policy(self, state):
        A = np.ones(self.n) * self.epsilon / self.n
        best_action = np.argmax(self.Q[state])
        A[best_action] += (1.0 - self.epsilon)
        return A

    def train(self, num_episodes: int):
        """
        Train the agent.

        :param num_episodes: The number of training episodes.
        """
        # Reset the trained parameters, if the agent was already trained
        if self.trained:
            self.returns_sum = defaultdict(float)
            self.returns_count = defaultdict(float)
            self.Q = defaultdict(lambda: np.zeros(self.n))
            self.reward_history = []

        for _ in tqdm(range(num_episodes)):
            # Run the episode
            episode_history = []
            state, _ = self.env.reset()
            done = False
            while not done:
                probs = self._policy(state)
                action = np.random.choice(np.arange(len(probs)), p=probs)
                next_state, reward, done, _, _ = self.env.step(action)
                episode_history.append((state, action, reward))
                state = next_state
            self.reward_history.append(reward)

            # Update Q
            visited = set()
            G = 0
            for step in reversed(episode_history):
                state, action, reward = step
                G = self.discount_factor * G + reward

                if (state, action) not in visited:
                    self.returns_sum[(state, action)] += G
                    self.returns_count[(state, action)] += 1
                    self.Q[state][action] = self.returns_sum[(state, action)] / self.returns_count[(state, action)]

                    visited.add((state, action))
        
        self.trained = True

    def get_action(self, state):
        """
        Get the recommended action in the provided state.

        :param state: The state of the game, in the format (player' sum, dealer's upcard, has usable ace).
        :returns: 0, if the action is STICK, 1 if it's HIT.
        """
        return np.argmax(self.Q[state]) 
                    
