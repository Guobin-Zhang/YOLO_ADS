import numpy as np
import cv2
import torch
from yolov9_model import YOLOv9_GPU, YOLOv9_FPGA
from dataset_preprocessing import get_datasets, get_data_loaders

class ReinforcementLearning:
    def __init__(self, use_fpga=False):
        """
        Initialize the reinforcement learning module with explicit state and action value functions.
        """
        self.use_fpga = use_fpga
        self.device = torch.device("cuda" if torch.cuda.is_available() and not use_fpga else "cpu")
        
        # Load YOLO model
        if use_fpga:
            self.model = YOLOv9_FPGA(num_classes=5)
        else:
            self.model = YOLOv9_GPU(num_classes=5)
            self.model.load_state_dict(torch.load("yolov9_gpu.pth"))
        self.model.to(self.device)
        self.model.eval()

        # State dimensions: [class_id, distance_bin, speed_bin]
        self.num_states = np.prod((5, 10, 5))  # 5 classes, 10 distance bins, 5 speed bins
        self.num_actions = 4  # Brake, Turn Left, Turn Right, Continue
        
        # Initialize Q-table (action value function)
        self.q_table = np.zeros((self.num_states, self.num_actions))
        
        # Initialize V-table (state value function)
        self.v_table = np.zeros(self.num_states)
        
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
            The chosen action.
        """
        state_index = np.ravel_multi_index(state, (5, 10, 5))
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state_index])

    def update_q_table(self, state, action, reward, next_state):
        """
        Update the Q-table using Q-learning.
        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
       