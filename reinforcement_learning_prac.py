import numpy as np
from yolov5_model import YOLOv5_GPU, YOLOv5_FPGA
from dataset_preprocessing import get_datasets, get_data_loaders
import torch

class ReinforcementLearning:
    def __init__(self, use_fpga=False):
        """
        Initialize the reinforcement learning module with YOLO model integration.
        """
        self.use_fpga = use_fpga
        self.device = torch.device("cpu" if use_fpga else "cuda")
        
        # Load YOLO model
        if use_fpga:
            self.model = YOLOv5_FPGA(num_classes=5)
        else:
            self.model = YOLOv5_GPU(num_classes=5)
            self.model.load_state_dict(torch.load("yolov5_gpu.pth"))
        
        self.model.to(self.device)
        self.model.eval()

        # State dimensions: [class_id, distance_bin, speed_bin]
        self.state_space = (5, 10, 5)  # 5 classes, 10 distance bins, 5 speed bins
        self.num_actions = 4  # Brake, Turn Left, Turn Right, Continue
        
        # Initialize Q-table (action value function)
        self.q_table = np.zeros(self.state_space + (self.num_actions,))
        
        # Reinforcement learning parameters
        self.learning_rate = 0.05
        self.discount_factor = 0.95
        self.epsilon = 0.05  # Exploration rate

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
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state])

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
            return -100
        elif action == 3 and distance_bin > 5:  # Safe driving (action 3 = Continue)
            return 20
        else:
            return -2

    def update_q_table(self, state, action, reward, next_state):
        """
        Update the Q-table using Q-learning.
        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
        """
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        new_q = (1 - self.learning_rate) * current_q + \
                self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[state][action] = new_q

    def train_with_y