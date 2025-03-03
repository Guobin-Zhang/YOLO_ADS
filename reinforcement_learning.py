import numpy as np

class ReinforcementLearning:
    def __init__(self, num_states, num_actions, learning_rate=0.05, discount_factor=0.95, epsilon=0.05):
        """
        Initialize the reinforcement learning module with explicit state and action value functions.
        Args:
            num_states: The number of possible states.
            num_actions: The number of possible actions.
            learning_rate: The learning rate for Q-learning.
            discount_factor: The discount factor for future rewards.
            epsilon: The exploration rate for epsilon-greedy policy.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Initialize Q-table (action value function)
        self.q_table = np.zeros((num_states, num_actions))
        
        # Initialize V-table (state value function)
        self.v_table = np.zeros(num_states)

    def update_value_functions(self, state, action, reward, next_state):
        """
        Update both state value function (V) and action value function (Q).
        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
        """
        # Update action value function (Q-learning)
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state, action]
        new_q = (1 - self.learning_rate) * current_q + \
                self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[state, action] = new_q
        
        # Update state value function (V = max_a Q(s, a))
        self.v_table[state] = np.max(self.q_table[state])

    def get_state_value(self, state):
        """
        Get the state value for a given state.
        Args:
            state: The current state.
        Returns:
            The state value.
        """
        return self.v_table[state]

    def get_action_value(self, state, action):
        """
        Get the action value for a given state and action.
        Args:
            state: The current state.
            action: The action taken.
        Returns:
            The action value.
        """
        return self.q_table[state, action]

    def choose_action(self, state):
        """
        Choose an action based on epsilon-greedy policy.
        Args:
            state: The current state.
        Returns:
            The chosen action.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        """
        Update the Q-table using Q-learning.
        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
        """
        self.q_table[state, action] += self.learning_rate * (
            reward + self.discount_factor * np.max(self.q_table[next_state]) - self.q_table[state, action]
        )