import numpy as np
import matplotlib.pyplot as plt
from CSTR_model_plus import CSTRRLEnv
import time
from tqdm import tqdm

class PSOParticle:
    """
    Represents a single particle in the PSO algorithm.
    Each particle corresponds to a policy that outputs PID gains.
    """
    def __init__(self, dim, bounds):
        """
        Initialize a particle with random position and zero velocity.
        
        Args:
            dim: Dimensionality of the search space (number of policy parameters)
            bounds: List of tuples (min, max) for each dimension
        """
        # Initialize position randomly within bounds
        self.position = np.array([np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)])
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_score = -np.inf  # We maximize reward, so start with negative infinity

    def update_velocity(self, global_best_position, w=0.7, c1=1.5, c2=1.5):
        """
        Update the particle's velocity based on PSO equations.
        
        Args:
            global_best_position: The global best position across all particles
            w: Inertia weight
            c1: Cognitive parameter (personal best influence)
            c2: Social parameter (global best influence)
        """
        r1 = np.random.random(len(self.position))
        r2 = np.random.random(len(self.position))
        
        cognitive_component = c1 * r1 * (self.best_position - self.position)
        social_component = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive_component + social_component

    def update_position(self, bounds):
        """
        Update the particle's position based on its velocity and enforce bounds.
        
        Args:
            bounds: List of tuples (min, max) for each dimension
        """
        self.position = self.position + self.velocity
        
        # Enforce bounds
        for i in range(len(self.position)):
            self.position[i] = np.clip(self.position[i], bounds[i][0], bounds[i][1])

    def evaluate(self, env, setpoints_Cb, setpoints_V, setpoint_durations):
        """
        Evaluate the particle by running an episode in the environment.
        The policy represented by this particle outputs PID gains.
        
        Args:
            env: The environment to evaluate on
            setpoints_Cb, setpoints_V, setpoint_durations: Setpoint tracking parameters
            
        Returns:
            total_reward: The total reward accumulated over the episode
        """
        # Reset environment with specific setpoint schedule
        options = {
            'setpoints_Cb': setpoints_Cb,
            'setpoints_V': setpoints_V,
            'setpoint_durations': setpoint_durations
        }
        obs, _ = env.reset(options=options)
        
        total_reward = 0
        done = False
        info = {}
        
        while not done:
            # The position of the particle represents the policy parameters
            # In PSO, we directly optimize the policy parameters (gains)
            action = self.position
            
            # Normalize action to [-1, 1] for the environment
            min_bounds = np.array([-5, 0, 0.02, 0, 0, 0.01])
            max_bounds = np.array([25, 20, 10, 1, 2, 1])
            normalized_action = 2 * ((action - min_bounds) / (max_bounds - min_bounds)) - 1
            
            # Clip to ensure within action bounds of environment
            normalized_action = np.clip(normalized_action, -1, 1)
            
            # Take a step in the environment
            obs, reward, done, truncated, info = env.step(normalized_action)
            total_reward += reward
            
            # Optional: Break early if the controller is performing extremely poorly
            if total_reward < -1000:
                return total_reward
            
        return total_reward