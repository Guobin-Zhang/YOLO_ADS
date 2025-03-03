import numpy as np
import cv2
import torch
from yolov9_model import YOLOv9_GPU, YOLOv9_FPGA
from dataset_preprocessing import get_datasets, get_data_loaders

class DecisionMaker:
    def __init__(self, use_fpga=False):
        """
        Initialize the decision-making module for autonomous driving.
        The system uses YOLO model outputs to make real-time decisions.
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
        self.state_space = (5, 10, 5)  # 5 classes, 10 distance bins, 5 speed bins
        self.action_space = 4  # Brake, Turn Left, Turn Right, Continue
        
        # Initialize Q-table (action value function)
        self.q_table = np.random.normal(loc=0, scale=0.1, size=self.state_space + (self.action_space,))
        
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
            return self.collision_penalty
        elif action == "Continue" and distance_bin > 5:  # Safe driving
            return self.safe_driving_reward
        else:
            return self.default_penalty

    def make_decision(self, image_path):
        """
        Make decisions based on YOLO detections from an image.
        Args:
            image_path: Path to the input image.
        """
        # Load image and preprocess
        image = cv2.imread(image_path)
        image = cv2.resize(image, (640, 640))
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image)
            detections = outputs[0].cpu().numpy()
        
        # Process detections
        decisions = []
        for detection in detections:
            class_id, x_center, y_center, speed = detection
            state = self.get_state((class_id, x_center, y_center, speed))
            action = self.choose_action(state)
            reward = self.calculate_reward(state, action)
            decisions.append((state, action, reward))
        
        return decisions

if __name__ == "__main__":
    decision_maker = DecisionMaker(use_fpga=False)
    decisions = decision_maker.make_decision("path/to/image.jpg")
    for state, action, reward in decisions:
        print(f"State: {state}, Action: {action}, Reward: {reward}")