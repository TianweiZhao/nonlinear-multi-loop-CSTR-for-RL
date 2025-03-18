"""
gen_sim_data.py - Generate simulation data for offline RL training

This module provides functions to generate diverse historical data from the CSTR system
for training RL algorithms offline. It includes methods for:
1. Generating varied setpoint schedules (increasing, decreasing, peak, valley, constant)
2. Collecting data with multiple exploration strategies
3. Processing and formatting data for offline learning
4. Analyzing and visualizing the generated data
"""

import numpy as np
import os
import torch
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict


class DataGenerator:
    """
    Class for generating and managing simulation data for offline RL training.
    """
    def __init__(self, env, save_dir="./offline_data", seed=None):
        """
        Initialize the data generator.
        
        Args:
            env: The CSTR environment instance
            save_dir (str): Directory to save generated data
            seed (int): Random seed for reproducibility (optional)
        """
        self.env = env
        self.save_dir = save_dir
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Default setpoint schedules for training
        self.default_setpoint_schedules = [
            [0.15, 0.30, 0.45, 0.60, 0.75],  # Increasing steps
            [0.85, 0.70, 0.55, 0.40, 0.25],  # Decreasing steps
            [0.35, 0.65, 0.90, 0.65, 0.35],  # Peak
            [0.88, 0.68, 0.48, 0.68, 0.88],  # Valley
            [0.75, 0.75, 0.75, 0.75, 0.75]   # Constant
        ]
    
    def generate_random_setpoint_schedule(self, n_setpoints=5, min_val=0.20, max_val=0.90):
        """
        Generate a random setpoint schedule.
        
        Args:
            n_setpoints (int): Number of setpoints in the schedule
            min_val (float): Minimum setpoint value
            max_val (float): Maximum setpoint value
            
        Returns:
            list: Random setpoint schedule
        """
        # Generate random setpoints within the specified range
        schedule = list(min_val + (max_val - min_val) * np.random.rand(n_setpoints))
        
        # Ensure there's at least some variation in the schedule
        if n_setpoints > 1 and max(schedule) - min(schedule) < 0.1:
            # If variation is too small, create a more diverse schedule
            schedule[0] = min_val + 0.1   # Near minimum
            schedule[-1] = max_val - 0.1  # Near maximum
        
        return schedule
    
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
            base_action = np.array([0.95, 0.01, 0.2, 0.2, 0.5, 0.05])
            noise = 0.2 * np.random.randn(6)  # Add some exploration noise
            return np.clip(base_action + noise, -1, 1)
        
        elif strategy == "decaying":
            # Decaying randomness (more exploration early, more exploitation later)
            progress = step / total_steps
            exploration_scale = max(0.1, 1.0 - progress)  # Decay from 1.0 to 0.1
            
            # Baseline PID gains
            base_action = np.array([0.95, 0.01, 0.2, 0.2, 0.5, 0.05])
            noise = exploration_scale * np.random.randn(6)
            
            return np.clip(base_action + noise, -1, 1)
        
        elif strategy == "mixed":
            # Mixed strategy: sometimes random, sometimes PID-based
            if np.random.rand() < 0.3:  # 30% chance of pure random
                return self.env.action_space.sample()
            else:
                # PID-based with noise
                base_action = np.array([0.95, 0.01, 0.2, 0.2, 0.5, 0.05])
                noise = 0.15 * np.random.randn(6)
                return np.clip(base_action + noise, -1, 1)
        
        else:
            # Default to random
            print(f"Unknown strategy '{strategy}', defaulting to random")
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
            'setpoints': [],  # Store the setpoints for reference
            'episode_ids': [] # Store which episode each transition belongs to
        }
        
        # Metadata for episodes
        episode_metadata = []
        
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
            
            # Store episode metadata
            episode_info = {
                'id': ep,
                'setpoint_schedule': setpoint_schedule,
                'exploration_strategy': exploration_strategy,
                'rewards': []
            }
            
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
                dataset['episode_ids'].append(ep)
                
                # Update episode info
                episode_info['rewards'].append(reward)
                
                # Move to next step
                state = next_state
                step += 1
                
                if done or truncated:
                    break
            
            # Update episode metadata
            episode_info['total_reward'] = sum(episode_info['rewards'])
            episode_info['steps'] = step
            episode_metadata.append(episode_info)
        
        # Convert lists to numpy arrays
        for key in dataset:
            dataset[key] = np.array(dataset[key])
        
        # Add episode metadata to dataset
        dataset['episode_metadata'] = episode_metadata
        
        if verbose:
            print(f"Generated dataset with {len(dataset['states'])} transitions from {n_episodes} episodes")
        
        return dataset
    
    def save_dataset(self, dataset, filename="offline_train_dataset.pkl"):
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
    
    def load_dataset(self, filename="offline_train_dataset.pkl"):
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
        # Create plots directory
        plots_dir = os.path.join(self.save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Basic statistics
        n_transitions = len(dataset['states'])
        episode_metadata = dataset.get('episode_metadata', [])
        n_episodes = len(episode_metadata) if episode_metadata else dataset['dones'].sum()
        
        print(f"Dataset contains {n_transitions} transitions from {n_episodes} episodes")
        
        # Extract episode rewards if available in metadata
        if episode_metadata:
            episode_rewards = [ep['total_reward'] for ep in episode_metadata]
            episode_ids = [ep['id'] for ep in episode_metadata]
        else:
            # Calculate rewards per episode if metadata not available
            episode_rewards = []
            episode_ids = []
            
            current_reward = 0
            current_id = 0
            
            for i, (reward, done) in enumerate(zip(dataset['rewards'], dataset['dones'])):
                current_reward += reward
                
                if done or i == len(dataset['rewards']) - 1:
                    episode_rewards.append(current_reward)
                    episode_ids.append(current_id)
                    current_reward = 0
                    current_id += 1
        
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
        if 'pid_gains' in dataset:
            pid_gains = dataset['pid_gains']
            pid_names = ["Kp_Cb", "Ki_Cb", "Kd_Cb", "Kp_V", "Ki_V", "Kd_V"]
            print(f"PID gains statistics:")
            for i, name in enumerate(pid_names):
                print(f"  {name} - Mean: {np.mean(pid_gains[:, i]):.4f}, Std: {np.std(pid_gains[:, i]):.4f}")
        
        # Setpoint distribution
        if 'setpoints' in dataset:
            setpoints = dataset['setpoints']
            unique_setpoints = np.unique(setpoints[:, 0])
            print(f"Unique Cb setpoints: {unique_setpoints}")
        
        # Create plots
        if save_plots:
            # 1. Episode reward plot
            plt.figure(figsize=(10, 6))
            plt.bar(episode_ids, episode_rewards, alpha=0.7)
            plt.title("Episode Rewards")
            plt.xlabel("Episode ID")
            plt.ylabel("Total Reward")
            plt.grid(True, axis='y')
            plt.savefig(os.path.join(plots_dir, "episode_rewards.png"))
            plt.close()
            
            # 2. Action distribution (overall distribution of all actions)
            plt.figure(figsize=(12, 8))
            for i in range(actions.shape[1]):
                plt.subplot(2, 3, i+1)
                plt.hist(actions[:, i], bins=30, alpha=0.7)
                plt.title(f"Action {i} Distribution")
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "action_distributions.png"))
            plt.close()
            
            # 3. State trajectories vs setpoints (sample from an episode)
            if episode_metadata and len(episode_metadata) > 0:
                # Select a random episode to visualize
                sample_ep_id = np.random.choice(episode_ids)
                sample_indices = np.where(dataset['episode_ids'] == sample_ep_id)[0]
                
                if len(sample_indices) > 0:
                    # Extract data for this episode
                    ep_states = dataset['states'][sample_indices]
                    ep_next_states = dataset['next_states'][sample_indices]
                    ep_setpoints = dataset['setpoints'][sample_indices]
                    
                    # Plot Cb trajectory
                    plt.figure(figsize=(12, 8))
                    
                    # Plot Cb vs setpoint
                    plt.subplot(2, 2, 1)
                    plt.plot(range(len(sample_indices)), ep_next_states[:, 0], 'b-', label='Cb (Measured)')
                    plt.plot(range(len(sample_indices)), ep_setpoints[:, 0], 'r--', label='Cb (Setpoint)')
                    plt.title(f"Cb Trajectory (Episode {sample_ep_id})")
                    plt.xlabel("Step")
                    plt.ylabel("Concentration B")
                    plt.legend()
                    plt.grid(True)
                    
                    # Plot V vs setpoint
                    plt.subplot(2, 2, 2)
                    plt.plot(range(len(sample_indices)), ep_next_states[:, 2], 'g-', label='V (Measured)')
                    plt.plot(range(len(sample_indices)), ep_setpoints[:, 1], 'r--', label='V (Setpoint)')
                    plt.title(f"Volume Trajectory (Episode {sample_ep_id})")
                    plt.xlabel("Step")
                    plt.ylabel("Volume")
                    plt.legend()
                    plt.grid(True)
                    
                    # Plot Temperature
                    plt.subplot(2, 2, 3)
                    plt.plot(range(len(sample_indices)), ep_next_states[:, 3], 'm-')
                    plt.title(f"Temperature Trajectory (Episode {sample_ep_id})")
                    plt.xlabel("Step")
                    plt.ylabel("Temperature")
                    plt.grid(True)
                    
                    # Plot rewards
                    plt.subplot(2, 2, 4)
                    plt.plot(range(len(sample_indices)), dataset['rewards'][sample_indices], 'c-')
                    plt.title(f"Rewards (Episode {sample_ep_id})")
                    plt.xlabel("Step")
                    plt.ylabel("Reward")
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f"sample_trajectory_ep{sample_ep_id}.png"))
                    plt.close()
            
            # 4. Reward distribution
            plt.figure(figsize=(10, 6))
            plt.hist(rewards, bins=50, alpha=0.7)
            plt.title("Reward Distribution")
            plt.xlabel("Reward")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, "reward_distribution.png"))
            plt.close()
            
            # 5. Setpoint coverage
            if 'setpoints' in dataset:
                plt.figure(figsize=(10, 6))
                plt.hist(setpoints[:, 0], bins=30, alpha=0.7)
                plt.title("Setpoint Cb Distribution")
                plt.xlabel("Setpoint Cb Value")
                plt.ylabel("Frequency")
                plt.grid(True)
                plt.savefig(os.path.join(plots_dir, "setpoint_distribution.png"))
                plt.close()
        
        # Return statistics dictionary
        stats = {
            'n_transitions': n_transitions,
            'n_episodes': n_episodes,
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'action_means': np.mean(actions, axis=0),
            'action_stds': np.std(actions, axis=0),
            'episode_rewards': episode_rewards
        }
        
        if 'pid_gains' in dataset:
            stats['pid_means'] = np.mean(pid_gains, axis=0)
            stats['pid_stds'] = np.std(pid_gains, axis=0)
        
        if 'setpoints' in dataset:
            stats['unique_setpoints'] = unique_setpoints
        
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
        combined = {key: [] for key in datasets[0].keys() if key != 'episode_metadata'}
        
        # Initialize list for episode metadata
        combined_metadata = []
        
        # Track episode ID offset
        episode_id_offset = 0
        
        # Combine all datasets
        for i, dataset in enumerate(datasets):
            # Combine all arrays except metadata
            for key in combined:
                combined[key].append(dataset[key])
            
            # Adjust episode IDs and metadata
            if 'episode_metadata' in dataset:
                for ep_info in dataset['episode_metadata']:
                    # Create a copy of the episode info
                    new_ep_info = ep_info.copy()
                    # Adjust the episode ID to avoid duplicates
                    new_ep_info['id'] += episode_id_offset
                    # Add dataset origin for reference
                    new_ep_info['origin_dataset'] = i
                    combined_metadata.append(new_ep_info)
                
                # Update the episode ID offset
                episode_id_offset += len(dataset['episode_metadata'])
            else:
                # If no metadata, estimate number of episodes from done flags
                episode_id_offset += np.sum(dataset['dones'])
        
        # Concatenate arrays
        for key in combined:
            combined[key] = np.concatenate(combined[key], axis=0)
        
        # Add the combined metadata
        combined['episode_metadata'] = combined_metadata
        
        print(f"Combined {len(datasets)} datasets with a total of {len(combined['states'])} transitions")
        
        return combined


def generate_diverse_dataset(env, n_episodes=40, save_dir="./data", seed=None):
    """
    Convenience function to generate a diverse dataset with multiple strategies.
    
    Args:
        env: CSTR environment instance
        n_episodes (int): Total number of episodes to simulate
        save_dir (str): Directory to save data
        seed (int): Random seed for reproducibility (optional)
        
    Returns:
        dict: Combined dataset
    """
    # Create data generator
    gen = DataGenerator(env, save_dir=save_dir, seed=seed)
    
    # Number of episodes per strategy
    eps_per_strategy = n_episodes // 4
    remainder = n_episodes % 4
    strategy_eps = [
        eps_per_strategy + (1 if i < remainder else 0) 
        for i in range(4)
    ]
    
    # Generate different setpoint schedules for diversity
    basic_schedules = [
        [0.15, 0.30, 0.45, 0.60, 0.75],  # Increasing steps
        [0.85, 0.70, 0.55, 0.40, 0.25],  # Decreasing steps
        [0.35, 0.65, 0.90, 0.65, 0.35],  # Peak
        [0.88, 0.68, 0.48, 0.68, 0.88],  # Valley
        [0.75, 0.75, 0.75, 0.75, 0.75]   # Constant
    ]
    
    # Add some random schedules
    random_schedules = []
    for _ in range(3):
        random_schedules.append(gen.generate_random_setpoint_schedule(n_setpoints=4))
    
    all_schedules = basic_schedules + random_schedules
    
    # Generate data with different exploration strategies
    datasets = []
    
    print(f"Generating diverse dataset with {n_episodes} total episodes")
    
    # Random exploration
    print(f"Generating {strategy_eps[0]} episodes with 'random' strategy")
    dataset_random = gen.generate_dataset(
        n_episodes=strategy_eps[0],
        exploration_strategy="random",
        custom_schedules=all_schedules,
        verbose=True
    )
    datasets.append(dataset_random)
    
    # Static PID exploration
    print(f"Generating {strategy_eps[1]} episodes with 'static_pid' strategy")
    dataset_static = gen.generate_dataset(
        n_episodes=strategy_eps[1],
        exploration_strategy="static_pid",
        custom_schedules=all_schedules,
        verbose=True
    )
    datasets.append(dataset_static)
    
    # Decaying exploration
    print(f"Generating {strategy_eps[2]} episodes with 'decaying' strategy")
    dataset_decay = gen.generate_dataset(
        n_episodes=strategy_eps[2],
        exploration_strategy="decaying",
        custom_schedules=all_schedules,
        verbose=True
    )
    datasets.append(dataset_decay)
    
    # Mixed exploration
    print(f"Generating {strategy_eps[3]} episodes with 'mixed' strategy")
    dataset_mixed = gen.generate_dataset(
        n_episodes=strategy_eps[3],
        exploration_strategy="mixed",
        custom_schedules=all_schedules,
        verbose=True
    )
    datasets.append(dataset_mixed)
    
    # Combine all datasets
    combined_dataset = gen.combine_datasets(datasets)
    
    # Analyze and save the combined dataset
    gen.analyze_dataset(combined_dataset, save_plots=True)
    gen.save_dataset(combined_dataset, filename="offline_train_dataset.pkl")
    
    return combined_dataset


if __name__ == "__main__":
    from CSTR_model_plus import CSTRRLEnv
    
    # Create environment with realistic conditions
    env = CSTRRLEnv(
        simulation_steps=100,
        dt=1.0,
        uncertainty_level=0.00,     # Add some uncertainty for realism
        noise_level=0.00,           # Add some noise for realism
        actuator_delay_steps=0,     # Add realistic delays
        transport_delay_steps=0,
        enable_disturbances=False   # Enable disturbances for robustness
    )
    
    # Generate dataset with default parameters
    dataset = generate_diverse_dataset(env, n_episodes=40)
    
    print("Dataset generation complete.")