import numpy as np

class DecisionMaker:
    def __init__(self):
        """
        Initialize the decision-making module for autonomous driving.
        The system uses YOLO model outputs to make real-time decisions.
        """
        # State dimensions: [class_id, distance_bin, speed_bin]
        self.state_space = (5, 10, 5)  # 5 classes, 10 distance bins, 5 speed bins
        self.action_space = 4  # Brake, Turn Left, Turn Right, Continue
        
        # Initialize Q-table (action value function)
        self.q_table = np.random.normal(
            loc=0, scale=0.1, 
            size=self.state_space + (self.action_space,)
        )
        
        # Initialize V-table (state value function)
        self.v_table = np.zeros(self.state_space)
        
        # Initialize state transition model
        self.transition_counts = {}  # Count transitions to estimate probabilities
        self.transition_probs = {}  # Estimated transition probabilities
        
        # Initialize reward function parameters
        self.collision_penalty = -100
        self.safe_driving_reward = 20
        self.default_penalty = -2
        
        # Reinforcement learning parameters
        self.learning_rate = 0.05
        self.discount_factor = 0.95
        self.epsilon = 0.05  # Exploration rate

    def _initialize_transition_probs(self):
        """
        Initialize state transition probabilities P(s'|s,a).
        Returns:
            A dictionary mapping (state, action) pairs to probability distributions over next states.
        """
        # Placeholder for transition probabilities
        # In practice, these should be learned from data or estimated from a simulator
        probs = {}
        for s in np.ndindex(*self.state_space):
            for a in range(self.action_space):
                # Default: deterministic transitions
                probs[(s, a)] = {s: 1.0}  # Stay in the same state by default
        return probs

    def update_transition(self, state, action, next_state):
        """
        Update transition counts based on observed transitions.
        Args:
            state: The current state.
            action: The action taken.
            next_state: The next state.
        """
        key = (state, action)
        if key not in self.transition_counts:
            self.transition_counts[key] = {}
        if next_state not in self.transition_counts[key]:
            self.transition_counts[key][next_state] = 0
        self.transition_counts[key][next_state] += 1

    def estimate_transition_probs(self):
        """
        Estimate transition probabilities from observed counts.
        """
        for (state, action), counts in self.transition_counts.items():
            total = sum(counts.values())
            self.transition_probs[(state, action)] = {s: c/total for s, c in counts.items()}

    def get_transition_prob(self, state, action, next_state):
        """
        Get the probability of transitioning to next_state from state with action.
        Args:
            state: The current state.
            action: The action taken.
            next_state: The next state.
        Returns:
            The transition probability.
        """
        return self.transition_probs.get((state, action), {}).get(next_state, 0.0)

    def calculate_reward(self, state, action):
        """
        Calculate the reward based on the state and action.
        Args:
            state: The current state.
            action: The action taken.
        Returns:
            The calculated reward.
        """
        class_id, distance_bin, speed_bin = state
        if distance_bin == 0:  # Collision
            return self.collision_penalty
        elif action == "Continue" and distance_bin > 5:  # Safe driving
            return self.safe_driving_reward
        else:
            return self.default_penalty

    def get_state(self, detection):
        """
        Convert YOLO detection to state representation.
        Args:
            detection: A tuple containing (class_id, x_center, y_center, speed).
        Returns:
            A tuple representing the state (class_id, distance_bin, speed_bin).
        """
        class_id, x_center, y_center, speed = detection
        distance = np.sqrt(x_center**2 + y_center**2)
        
        # Discretize inputs
        distance_bin = min(int(distance // 10), 9)  # 10 bins @ 10m each
        speed_bin = min(int(speed // 5), 4)         # 5 bins @ 5m/s each
        
        return (class_id, distance_bin, speed_bin)
    
    def choose_action(self, state):
        """
        Choose an action based on epsilon-greedy policy.
        Args:
            state: The current state of the system.
        Returns:
            The chosen action (Brake, Turn Left, Turn Right, Continue).
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(["Brake", "Turn Left", "Turn Right", "Continue"])
        else:
            return ["Brake", "Turn Left", "Turn Right", "Continue"][np.argmax(self.q_table[state])]
    
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
        current_q = self.q_table[state][action]
        new_q = (1 - self.learning_rate) * current_q + \
                self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[state][action] = new_q
        
        # Update state value function (V = max_a Q(s, a))
        self.v_table[state] = np.max(self.q_table[state])

    def get_next_state(self, state, action):
        """
        Simulate the next state based on the current state and action.
        Args:
            state: The current state.
            action: The action taken.
        Returns:
            The next state.
        """
        # Sample next state from transition probabilities
        next_state_probs = self.transition_probs.get((state, action), {state: 1.0})
        next_states = list(next_state_probs.keys())
        probs = list(next_state_probs.values())
        return next_states[np.random.choice(len(next_states), p=probs)]

    def make_decision(self, detections):
        """
        Make decisions based on YOLO detections.
        Args:
            detections: A list of detections from the YOLO model.
        Returns:
            A list of decisions for each detection.
        """
        decisions = []
        for detection in detections:
            state = self.get_state(detection)
            action = self.choose_action(state)
            reward = self.calculate_reward(state, action)
            next_state = self.get_next_state(state, action)
            self.update_transition(state, action, next_state)
            self.update_value_functions(state, action, reward, next_state)
            decisions.append(action)
        return decisions