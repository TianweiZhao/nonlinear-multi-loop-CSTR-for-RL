"""
gen_sim_data.py - Generate simulation data for offline RL training

This module provides functions to generate historical data from the CSTR system
for training RL algorithms offline. It includes methods for:
1. Generating diverse setpoint schedules
2. Collecting data with various exploration strategies
3. Processing and formatting data for offline learning
"""

import numpy as np
import os
import torch
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict


class DataGenerator:
    """
    Class for generating and managing simulation data for offline RL training.
    """
    def __init__(self, env, save_dir="./data"):
        """
        Initialize the data generator.
        
        Args:
            env: The CSTR environment instance
            save_dir (str): Directory to save generated data
        """
        self.env = env
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Default setpoint schedules for training
        self.default_setpoint_schedules = [
            [0.65, 0.75, 0.85],  # Increasing steps
            [0.85, 0.75, 0.65],  # Decreasing steps
            [0.70, 0.90, 0.70],  # Peak
            [0.80, 0.60, 0.80],  # Valley
            [0.75, 0.75, 0.75]   # Constant
        ]
    
    def generate_random_setpoint_schedule(self, n_setpoints=3, min_val=0.60, max_val=0.90):
        """
        Generate a random setpoint schedule.
        
        Args:
            n_setpoints (int): Number of setpoints in the schedule
            min_val (float): Minimum setpoint value
            max_val (float): Maximum setpoint value
            
        Returns:
            list: Random setpoint schedule
        """
        return list(min_val + (max_val - min_val) * np.random.rand(n_setpoints))
    
    def generate_exploration_action(self, step, total_steps, strategy="random"):
        """
        Generate an action for exploration based on chosen strategy.
        
        Args:
            step (int): Current step
            total_steps (int): Total number of steps
            strategy (str): Exploration strategy to use
                - "random": Pure random actions
                - "static_pid": Fixed PID gains with noise
                - "decaying": Decreasing randomness over time
                - "mixed": Combination of strategies
                
        Returns:
            numpy.ndarray: Action vector in range [-1, 1]
        """
        if strategy == "random":
            # Pure random actions
            return self.env.action_space.sample()
        
        elif strategy == "static_pid":
            # Static PID gains with noise
            # These values are normalized PID gains that work reasonably well
            base_action = np.array([0.3, 0.2, 0.1, 0.2, 0.1, 0.05])
            noise = 0.2 * np.random.randn(6)  # Add some exploration noise
            return np.clip(base_action + noise, -1, 1)
        
        elif strategy == "decaying":
            # Decaying randomness (more exploration early, more exploitation later)
            progress = step / total_steps
            exploration_scale = max(0.1, 1.0 - progress)  # Decay from 1.0 to 0.1
            
            # Baseline PID gains
            base_action = np.array([0.3, 0.2, 0.1, 0.2, 0.1, 0.05])
            noise = exploration_scale * np.random.randn(6)
            
            return np.clip(base_action + noise, -1, 1)
        
        elif strategy == "mixed":
            # Mixed strategy: sometimes random, sometimes PID-based
            if np.random.rand() < 0.3:  # 30% chance of pure random
                return self.env.action_space.sample()
            else:
                # PID-based with noise
                base_action = np.array([0.3, 0.2, 0.1, 0.2, 0.1, 0.05])
                noise = 0.15 * np.random.randn(6)
                return np.clip(base_action + noise, -1, 1)
        
        else:
            # Default to random
            return self.env.action_space.sample()
    
    def generate_dataset(self, n_episodes=20, steps_per_setpoint=25, 
                         exploration_strategy="mixed", custom_schedules=None,
                         verbose=True):
        """
        Generate a dataset by running episodes with exploration.
        
        Args:
            n_episodes (int): Number of episodes to run
            steps_per_setpoint (int): Number of steps for each setpoint
            exploration_strategy (str): Strategy for exploration
            custom_schedules (list): Custom setpoint schedules to use (optional)
            verbose (bool): Whether to show progress bar
            
        Returns:
            dict: Dataset with states, actions, rewards, next_states, dones
        """
        # Use custom schedules if provided, otherwise use default
        if custom_schedules is None:
            setpoint_schedules = self.default_setpoint_schedules.copy()
            
            # Add some random schedules for diversity
            for _ in range(3):
                setpoint_schedules.append(self.generate_random_setpoint_schedule())
        else:
            setpoint_schedules = custom_schedules
        
        # Initialize dataset dictionary
        dataset = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'pid_gains': [],  # Store the actual PID gains (unscaled)
            'setpoints': []   # Store the setpoints for reference
        }
        
        # Run episodes to collect data
        episode_range = tqdm(range(n_episodes)) if verbose else range(n_episodes)
        
        for ep in episode_range:
            # Select a schedule for this episode
            schedule_idx = ep % len(setpoint_schedules)
            setpoint_schedule = setpoint_schedules[schedule_idx]
            
            # Configure environment with setpoint schedule
            setpoints_Cb = setpoint_schedule
            setpoints_V = [100.0] * len(setpoint_schedule)  # Volume setpoint constant
            setpoint_durations = [steps_per_setpoint] * len(setpoint_schedule)
            
            # Reset environment with this schedule
            state, _ = self.env.reset(seed=ep, options={
                'setpoints_Cb': setpoints_Cb,
                'setpoints_V': setpoints_V,
                'setpoint_durations': setpoint_durations
            })
            
            # Calculate total steps for this episode
            total_steps = len(setpoint_schedule) * steps_per_setpoint
            
            # Run the episode
            done = False
            step = 0
            
            while not done and step < total_steps:
                # Generate action using the chosen exploration strategy
                action = self.generate_exploration_action(step, total_steps, exploration_strategy)
                
                # Take a step in the environment
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # Calculate actual PID gains (unscaled)
                pid_gains = ((action + 1) / 2) * (self.env.pid_upper - self.env.pid_lower) + self.env.pid_lower
                
                # Get current setpoints
                current_setpoint_Cb = next_state[9]  # Index where setpoint Cb is stored
                current_setpoint_V = next_state[10]  # Index where setpoint V is stored
                
                # Store transition
                dataset['states'].append(state)
                dataset['actions'].append(action)
                dataset['rewards'].append(reward)
                dataset['next_states'].append(next_state)
                dataset['dones'].append(done or truncated)
                dataset['pid_gains'].append(pid_gains)
                dataset['setpoints'].append([current_setpoint_Cb, current_setpoint_V])
                
                # Move to next step
                state = next_state
                step += 1
                
                if done or truncated:
                    break
        
        # Convert lists to numpy arrays
        for key in dataset:
            dataset[key] = np.array(dataset[key])
        
        if verbose:
            print(f"Generated dataset with {len(dataset['states'])} transitions from {n_episodes} episodes")
        
        return dataset
    
    def save_dataset(self, dataset, filename="cstr_dataset.pkl"):
        """
        Save dataset to a file.
        
        Args:
            dataset (dict): Dataset to save
            filename (str): Filename to save to
        """
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filename="cstr_dataset.pkl"):
        """
        Load dataset from a file.
        
        Args:
            filename (str): Filename to load from
            
        Returns:
            dict: Loaded dataset
        """
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Loaded dataset from {filepath} with {len(dataset['states'])} transitions")
        return dataset
    
    def analyze_dataset(self, dataset, save_plots=True):
        """
        Analyze dataset and generate diagnostic plots.
        
        Args:
            dataset (dict): Dataset to analyze
            save_plots (bool): Whether to save plots to disk
            
        Returns:
            dict: Statistics about the dataset
        """
        # Basic statistics
        n_transitions = len(dataset['states'])
        n_episodes = sum(dataset['dones'])
        
        print(f"Dataset contains {n_transitions} transitions from approximately {n_episodes} episodes")
        
        # Reward statistics
        rewards = dataset['rewards']
        print(f"Reward statistics:")
        print(f"  Mean: {np.mean(rewards):.4f}")
        print(f"  Std: {np.std(rewards):.4f}")
        print(f"  Min: {np.min(rewards):.4f}")
        print(f"  Max: {np.max(rewards):.4f}")
        
        # Action statistics
        actions = dataset['actions']
        print(f"Action statistics:")
        for i in range(actions.shape[1]):
            print(f"  Action {i} - Mean: {np.mean(actions[:, i]):.4f}, Std: {np.std(actions[:, i]):.4f}")
        
        # PID gains statistics
        pid_gains = dataset['pid_gains']
        pid_names = ["Kp_Cb", "Ki_Cb", "Kd_Cb", "Kp_V", "Ki_V", "Kd_V"]
        print(f"PID gains statistics:")
        for i, name in enumerate(pid_names):
            print(f"  {name} - Mean: {np.mean(pid_gains[:, i]):.4f}, Std: {np.std(pid_gains[:, i]):.4f}")
        
        # Setpoint distribution
        setpoints = dataset['setpoints']
        unique_setpoints = np.unique(setpoints[:, 0])
        print(f"Unique Cb setpoints: {unique_setpoints}")
        
        # Plot action distributions
        if save_plots:
            os.makedirs(os.path.join(self.save_dir, "plots"), exist_ok=True)
            
            # Action distribution plots
            plt.figure(figsize=(15, 10))
            for i in range(actions.shape[1]):
                plt.subplot(2, 3, i+1)
                plt.hist(actions[:, i], bins=30, alpha=0.7)
                plt.title(f"Action {i} Distribution")
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, "plots", "action_distributions.png"))
            plt.close()
            
            # PID gains distribution plots
            plt.figure(figsize=(15, 10))
            for i, name in enumerate(pid_names):
                plt.subplot(2, 3, i+1)
                plt.hist(pid_gains[:, i], bins=30, alpha=0.7)
                plt.title(f"{name} Distribution")
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, "plots", "pid_distributions.png"))
            plt.close()
            
            # Reward distribution plot
            plt.figure(figsize=(10, 6))
            plt.hist(rewards, bins=50, alpha=0.7)
            plt.title("Reward Distribution")
            plt.xlabel("Reward")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, "plots", "reward_distribution.png"))
            plt.close()
        
        # Return statistics dictionary
        stats = {
            'n_transitions': n_transitions,
            'n_episodes': n_episodes,
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'action_means': np.mean(actions, axis=0),
            'action_stds': np.std(actions, axis=0),
            'pid_means': np.mean(pid_gains, axis=0),
            'pid_stds': np.std(pid_gains, axis=0),
            'unique_setpoints': unique_setpoints
        }
        
        return stats
    
    def combine_datasets(self, datasets):
        """
        Combine multiple datasets into one.
        
        Args:
            datasets (list): List of datasets to combine
            
        Returns:
            dict: Combined dataset
        """
        if not datasets:
            return None
        
        # Initialize combined dataset with the same keys
        combined = {key: [] for key in datasets[0].keys()}
        
        # Combine all datasets
        for dataset in datasets:
            for key in combined:
                combined[key].append(dataset[key])
        
        # Concatenate arrays
        for key in combined:
            combined[key] = np.concatenate(combined[key], axis=0)
        
        print(f"Combined {len(datasets)} datasets with a total of {len(combined['states'])} transitions")
        
        return combined


def generate_diverse_dataset(env, n_episodes=50, save_dir="./data"):
    """
    Convenience function to generate a diverse dataset with multiple strategies.
    
    Args:
        env: CSTR environment instance
        n_episodes (int): Total number of episodes to simulate
        save_dir (str): Directory to save data
        
    Returns:
        dict: Combined dataset
    """
    # Create data generator
    gen = DataGenerator(env, save_dir=save_dir)
    
    # Generate data with different exploration strategies
    datasets = []
    
    # Random exploration
    dataset_random = gen.generate_dataset(
        n_episodes=n_episodes // 4,
        exploration_strategy="random",
        verbose=True
    )
    datasets.append(dataset_random)
    
    # Static PID exploration
    dataset_static = gen.generate_dataset(
        n_episodes=n_episodes // 4,
        exploration_strategy="static_pid",
        verbose=True
    )
    datasets.append(dataset_static)
    
    # Decaying exploration
    dataset_decay = gen.generate_dataset(
        n_episodes=n_episodes // 4,
        exploration_strategy="decaying",
        verbose=True
    )
    datasets.append(dataset_decay)
    
    # Mixed exploration
    dataset_mixed = gen.generate_dataset(
        n_episodes=n_episodes // 4,
        exploration_strategy="mixed",
        verbose=True
    )
    datasets.append(dataset_mixed)
    
    # Combine all datasets
    combined_dataset = gen.combine_datasets(datasets)
    
    # Analyze and save the combined dataset
    gen.analyze_dataset(combined_dataset, save_plots=True)
    gen.save_dataset(combined_dataset, filename="cstr_diverse_dataset.pkl")
    
    return combined_dataset


if __name__ == "__main__":
    # Example usage
    from CSTR_model_plus import CSTRRLEnv
    
    # Create environment
    env = CSTRRLEnv(
        simulation_steps=100,  # Should be enough for a full episode
        dt=1.0,
        uncertainty_level=0.05,  # Add some uncertainty for realism
        noise_level=0.02,        # Add some noise for realism
        actuator_delay_steps=1,  # Add realistic delays
        transport_delay_steps=1,
        enable_disturbances=True  # Enable disturbances for robustness
    )
    
    # Generate dataset
    dataset = generate_diverse_dataset(env, n_episodes=50)
    
    print("Dataset generation complete.")