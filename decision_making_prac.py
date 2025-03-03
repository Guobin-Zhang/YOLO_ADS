import numpy as np
from yolov5_model import YOLOv5_GPU, YOLOv5_FPGA
from dataset_preprocessing import get_datasets, get_data_loaders
import torch

class DecisionMaker:
    def __init__(self, use_fpga=False):
        """
        Initialize the decision-making module for autonomous driving.
        The system uses YOLO model outputs to make real-time decisions.
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
        self.action_space = 4  # Brake, Turn Left, Turn Right, Continue
        
        # Initialize Q-table (action value function)
        self.q_table = np.random.normal(loc=0, scale=0.1, size=self.state_space + (self.action_space,))
        
        # Initialize V-table (state value function)
        self.v_table = np.zeros(self.state_space)
        
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
            return np.random.choice(["Brake", "Turn Left", "Turn Right", "Continue"])
        else:
            return ["Brake", "Turn Left", "Turn Right", "Continue"][np.argmax(self.q_table[state])]

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
        elif action == "Continue" and distance_bin > 5:  # Safe driving
            return 20
        else:
            return -2

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

    def make_decision(self, image):
        """
        Make decisions based on YOLO detections.
        Args:
            image: Input image tensor.
        Returns:
            A list of decisions for each detection.
        """
        with torch.no_grad():
            image = image.to(self.device)
            detections = self.model(image)
            detections = detections.cpu().numpy()
        
        decisions = []
        for detection in detections:
            state = self.get_state(detection)
            action = self.choose_action(state)
            reward = self.calculate_reward(state, action)
            next_state = state  # Simplified: assume next state is the same
            self.update_value_functions(state, action, reward, next_state)
            decisions.append(action)
        
        return decisions