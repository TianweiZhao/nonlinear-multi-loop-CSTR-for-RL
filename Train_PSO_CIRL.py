"""
Training approach for Control-Informed Reinforcement Learning (CIRL)

This module provides a trainer class for CIRL policies using a two-phase approach:
1. Supervised learning on offline data (initializes the policy to mimic effective controllers)
2. Direct policy optimization using Particle Swarm Optimization (PSO)

The CIRL approach integrates PID control structure into the reinforcement learning framework,
allowing for adaptive PID gain tuning with the interpretability of traditional control systems.
"""

import os
import torch
import numpy as np
import random
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from time import time

from CSTR_model_plus import CSTRRLEnv
from gen_sim_data import DataGenerator
from Replay_Buffer import ReplayBuffer
from cirl_policy_network import CIRLPolicyNetwork

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Particle:
    """
    A particle in the PSO algorithm, representing a set of policy parameters.
    """
    def __init__(self, policy, min_param, max_param):
        """
        Initialize a particle with random position and zero velocity.
        
        Args:
            policy (CIRLPolicyNetwork): The policy network template
            min_param (float): Minimum parameter value
            max_param (float): Maximum parameter value
        """
        # Create a new policy instance with the same architecture
        self.device = next(policy.parameters()).device
        self.policy = CIRLPolicyNetwork(
            state_dim=policy.state_dim,
            hidden_dims=policy.hidden_dims
        ).to(self.device)
        
        # Copy parameters from the template
        self.policy.load_state_dict(policy.state_dict())
        
        # Add small random noise to parameters
        for param in self.policy.parameters():
            noise = torch.rand_like(param.data) * (max_param - min_param) + min_param
            param.data += noise
        
        # Initialize velocity for each parameter
        self.velocity = {name: torch.zeros_like(param.data) 
                         for name, param in self.policy.named_parameters()}
        
        # Initialize best position and score
        self.best_position = {name: param.data.clone() 
                              for name, param in self.policy.named_parameters()}
        self.best_score = float('-inf')  # We're maximizing reward
    
    def update_velocity(self, global_best_position, w=0.7, c1=1.5, c2=1.5):
        """
        Update the particle's velocity based on PSO equations.
        
        Args:
            global_best_position (dict): The global best position
            w (float): Inertia weight
            c1 (float): Cognitive parameter (personal best influence)
            c2 (float): Social parameter (global best influence)
        """
        for name, param in self.policy.named_parameters():
            r1 = torch.rand_like(param.data)
            r2 = torch.rand_like(param.data)
            
            cognitive_component = c1 * r1 * (self.best_position[name] - param.data)
            social_component = c2 * r2 * (global_best_position[name] - param.data)
            
            self.velocity[name] = w * self.velocity[name] + cognitive_component + social_component
    
    def update_position(self, min_param, max_param):
        """
        Update the particle's position based on its velocity.
        
        Args:
            min_param (float): Minimum parameter value
            max_param (float): Maximum parameter value
        """
        for name, param in self.policy.named_parameters():
            param.data = param.data + self.velocity[name]
            param.data = torch.clamp(param.data, min_param, max_param)
    
    def evaluate(self, env, setpoints_Cb, setpoints_V, setpoint_durations):
        """
        Evaluate the particle by running an episode in the environment.
        
        Args:
            env: The environment to evaluate on
            setpoints_Cb: Setpoints for Cb concentration
            setpoints_V: Setpoints for volume
            setpoint_durations: Durations for each setpoint
            
        Returns:
            float: Total reward
        """
        # Reset environment with specific setpoint schedule
        state, _ = env.reset(options={
            'setpoints_Cb': setpoints_Cb,
            'setpoints_V': setpoints_V,
            'setpoint_durations': setpoint_durations
        })
        
        total_reward = 0
        done = False
        
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                action = self.policy(state_tensor).cpu().numpy()[0]
            
            # Take step in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            total_reward += reward
            state = next_state
            
            if done or truncated:
                break
        
        return total_reward

class CIRLTrainer:
    """
    Trainer for CIRL policies using offline data and combined optimization methods.
    """
    def __init__(self, env, save_dir="./results/CIRL", device="cuda"):
        """
        Initialize the CIRL Trainer.
        
        Args:
            env: The CSTR environment instance
            save_dir (str): Directory to save results
            device (str): Device to run training on ("cuda" or "cpu")
        """
        self.env = env
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        
        # Create directories
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "data"), exist_ok=True)
        
        # Get environment dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # PID gain mapping (from normalized [-1, 1] to actual values)
        self.pid_lower = env.pid_lower
        self.pid_upper = env.pid_upper
        
        # Initialize data generator
        self.data_generator = DataGenerator(env, save_dir=os.path.join(self.save_dir, "data"))
    
    def load_offline_dataset(self, file_path="/home/jzhao/work/nonlinear-multi-loop-CSTR-for-RL/offline_data/cstr_diverse_dataset.pkl", verbose=True):
        """
        Load the pre-generated offline dataset.
        
        Args:
            file_path (str): Path to the dataset file
            verbose (bool): Whether to print progress information
            
        Returns:
            dict: Loaded dataset
        """
        try:
            with open(file_path, 'rb') as f:
                dataset = pickle.load(f)
            
            if verbose:
                print(f"Loaded dataset from {file_path} with {len(dataset['states'])} transitions")
            
            return dataset
        except Exception as e:
            if verbose:
                print(f"Error loading dataset from {file_path}: {e}")
                print("Will try to generate a new dataset instead.")
            
            # Fall back to generating data if loading fails
            return self.generate_diverse_training_data(verbose=verbose)
    
    def generate_diverse_training_data(self, n_episodes=100, steps_per_setpoint=50, verbose=True):
        """
        Generate diverse training data for offline learning.
        
        Args:
            n_episodes (int): Number of episodes to generate
            steps_per_setpoint (int): Number of steps per setpoint
            verbose (bool): Whether to print progress information
            
        Returns:
            dict: Generated dataset
        """
        if verbose:
            print("Generating diverse training data...")
        
        # Generate various setpoint schedules for diverse learning
        setpoint_schedules = []
               
        # Random sequences
        for _ in range(3):
            random_setpoints = np.random.uniform(0.4, 0.9, 20).tolist()
            setpoint_schedules.append(random_setpoints)
        
        # Generate data with different exploration strategies
        datasets = []
        strategies = ["static_pid", "decaying"]
        n_per_strategy = n_episodes // len(strategies)
        
        for strategy in strategies:
            if verbose:
                print(f"Generating data with {strategy} exploration strategy...")
            
            dataset = self.data_generator.generate_dataset(
                n_episodes=n_per_strategy,
                steps_per_setpoint=steps_per_setpoint,
                exploration_strategy=strategy,
                custom_schedules=setpoint_schedules,
                verbose=verbose
            )
            datasets.append(dataset)
        
        # Combine all datasets
        combined_dataset = self.data_generator.combine_datasets(datasets)
        
        # Analyze and save the dataset
        self.data_generator.analyze_dataset(combined_dataset, save_plots=True)
        filepath = os.path.join(self.save_dir, "data", "cirl_training_data.pkl")
        self.data_generator.save_dataset(combined_dataset, filename=os.path.basename(filepath))
        
        if verbose:
            print(f"Training data generated with {len(combined_dataset['states'])} transitions")
            print(f"Saved to {filepath}")
        
        return combined_dataset
    
    def create_policy_network(self, hidden_dims=[128, 128]):
        """
        Create a CIRL Policy Network.
        
        Args:
            hidden_dims (list): Dimensions of hidden layers (default: [128, 128])
            
        Returns:
            CIRLPolicyNetwork: The policy network
        """
        policy = CIRLPolicyNetwork(
            state_dim=self.state_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        return policy
    
    def train_supervised(self, policy, dataset, n_epochs=50, batch_size=256, 
                        learning_rate=1e-3, weight_decay=1e-4, save_interval=10, verbose=True):
        """
        Phase 1: Train the CIRL policy using supervised learning on offline data.
        
        This phase teaches the policy to mimic the PID gains in the dataset,
        serving as initialization for subsequent optimization phases.
        
        Args:
            policy (CIRLPolicyNetwork): The policy network to train
            dataset (dict): The training dataset
            n_epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
            weight_decay (float): Weight decay for regularization
            save_interval (int): Interval for saving model checkpoints
            verbose (bool): Whether to print progress information
            
        Returns:
            dict: Training statistics
        """
        if verbose:
            print("Starting supervised learning phase...")
        
        # Create replay buffer
        replay_buffer = ReplayBuffer(
            capacity=len(dataset['states']) + 1000,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        # Load dataset into replay buffer
        replay_buffer.load_from_dataset(dataset)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create loss function (MSE)
        criterion = torch.nn.MSELoss()
        
        # Training statistics
        stats = {
            'epochs': [],
            'losses': [],
            'mean_losses': []
        }
        
        # Training loop
        progress_bar = tqdm(range(n_epochs)) if verbose else range(n_epochs)
        for epoch in progress_bar:
            epoch_losses = []
            
            # Compute number of batches
            n_batches = replay_buffer.size // batch_size
            
            for _ in range(n_batches):
                # Sample batch from replay buffer
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Forward pass
                predicted_actions = policy(states)
                
                # Compute loss
                loss = criterion(predicted_actions, actions)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # Record statistics
            mean_loss = np.mean(epoch_losses)
            stats['epochs'].append(epoch)
            stats['losses'].extend(epoch_losses)
            stats['mean_losses'].append(mean_loss)
            
            # Update progress bar
            if verbose:
                if isinstance(progress_bar, tqdm):
                    progress_bar.set_description(f"Epoch {epoch+1}/{n_epochs} | Loss: {mean_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0 or epoch == n_epochs - 1:
                checkpoint_path = os.path.join(self.save_dir, "models", f"cirl_policy_epoch_{epoch+1}.pt")
                policy.save(checkpoint_path)
        
        # Save final model
        final_model_path = os.path.join(self.save_dir, "models", "cirl_policy_supervised.pt")
        policy.save(final_model_path)
        
        # Save training statistics
        stats_path = os.path.join(self.save_dir, "supervised_training_stats.pkl")
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)
        
        # Plot training curve
        plt.figure(figsize=(10, 5))
        plt.plot(stats['epochs'], stats['mean_losses'])
        plt.title("CIRL Supervised Learning Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Loss")
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "plots", "supervised_learning_loss.png"))
        plt.close()
        
        if verbose:
            print(f"Supervised learning completed. Final model saved to {final_model_path}")
        
        return stats
    
    def train_pso(self, policy_template, n_iterations=50, num_particles=20, 
                min_param=-0.05, max_param=0.05, verbose=True):
        """
        Phase 2: Fine-tune the CIRL policy using Particle Swarm Optimization (PSO).
        
        This phase directly optimizes the policy for maximum reward, building on
        the initialized policy from the supervised learning phase.
        
        Args:
            policy_template (CIRLPolicyNetwork): Template for policy initialization
            n_iterations (int): Number of PSO iterations
            num_particles (int): Number of particles in the swarm
            min_param (float): Minimum parameter perturbation
            max_param (float): Maximum parameter perturbation
            verbose (bool): Whether to print progress information
            
        Returns:
            tuple: (best_policy, optimization_history)
        """
        if verbose:
            print("Starting PSO optimization phase...")
        
        # Initialize particles
        particles = []
        for _ in range(num_particles):
            particle = Particle(policy_template, min_param, max_param)
            particles.append(particle)
        
        # Define diverse setpoint schedules for optimization
        optimization_schedules = [
            [0.6, 0.7, 0.8],  # Increasing in optimal range
            [0.8, 0.7, 0.6],  # Decreasing in optimal range
            [0.6, 0.8, 0.6],  # Peak in optimal range
            [0.8, 0.6, 0.8]   # Valley in optimal range
        ]
        
        # Setup for setpoint tracking
        setpoints_V = [100.0] * 3      # 3 setpoints per schedule
        setpoint_durations = [50] * 3  # 50 steps per setpoint
        
        # Initialize global best
        global_best_position = {name: param.data.clone() 
                               for name, param in policy_template.named_parameters()}
        global_best_score = float('-inf')
        
        # Optimization history
        history = {
            'iterations': [],
            'best_scores': [],
            'mean_scores': []
        }
        
        # Create a fresh copy of the template for the best policy
        best_policy = self.create_policy_network(hidden_dims=policy_template.hidden_dims)
        
        # Main PSO loop
        progress_bar = tqdm(range(n_iterations)) if verbose else range(n_iterations)
        for iteration in progress_bar:
            iteration_scores = []
            
            # Evaluate each particle on multiple setpoint schedules
            for particle_idx, particle in enumerate(particles):
                total_score = 0
                
                # Evaluate on each schedule
                for schedule_idx, schedule in enumerate(optimization_schedules):
                    score = particle.evaluate(
                        self.env, schedule, setpoints_V, setpoint_durations
                    )
                    total_score += score
                    
                    if verbose and particle_idx == 0 and iteration % 10 == 0:
                        print(f"  Schedule {schedule_idx+1}: {score:.4f}")
                
                # Average score across schedules
                avg_score = total_score / len(optimization_schedules)
                iteration_scores.append(avg_score)
                
                # Update particle's best
                if avg_score > particle.best_score:
                    particle.best_score = avg_score
                    for name, param in particle.policy.named_parameters():
                        particle.best_position[name] = param.data.clone()
                
                # Update global best
                if avg_score > global_best_score:
                    global_best_score = avg_score
                    for name, param in particle.policy.named_parameters():
                        global_best_position[name] = param.data.clone()
                    
                    # Update best policy
                    for name, param in best_policy.named_parameters():
                        param.data = global_best_position[name].clone()
            
            # Update velocities and positions
            for particle in particles:
                # Adaptive parameters that change over time
                w = 0.9 - (0.5 * iteration / n_iterations)  # Decreasing inertia
                c1 = 2.0 - (0.5 * iteration / n_iterations)  # Decreasing cognitive weight
                c2 = 2.0 + (0.5 * iteration / n_iterations)  # Increasing social weight
                
                particle.update_velocity(global_best_position, w, c1, c2)
                particle.update_position(min_param, max_param)
            
            # Record statistics
            mean_score = np.mean(iteration_scores)
            history['iterations'].append(iteration)
            history['best_scores'].append(global_best_score)
            history['mean_scores'].append(mean_score)
            
            # Update progress bar
            if verbose:
                if isinstance(progress_bar, tqdm):
                    progress_bar.set_description(
                        f"Iteration {iteration+1}/{n_iterations} | "
                        f"Best: {global_best_score:.4f} | Mean: {mean_score:.4f}"
                    )
            
            # Save checkpoint every 20 iterations
            if (iteration + 1) % 20 == 0:
                checkpoint_path = os.path.join(self.save_dir, "models", f"cirl_policy_pso_iter_{iteration+1}.pt")
                best_policy.save(checkpoint_path)
        
        # Save final best policy
        final_model_path = os.path.join(self.save_dir, "models", "cirl_policy_pso.pt")
        best_policy.save(final_model_path)
        
        # Save optimization history
        history_path = os.path.join(self.save_dir, "pso_optimization_history.pkl")
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        
        # Plot optimization progress
        plt.figure(figsize=(10, 5))
        plt.plot(history['iterations'], history['best_scores'], 'b-', label='Best Score')
        plt.plot(history['iterations'], history['mean_scores'], 'r--', label='Mean Score')
        plt.title("CIRL PSO Optimization Progress")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "plots", "pso_optimization_progress.png"))
        plt.close()
        
        if verbose:
            print(f"PSO optimization completed. Final model saved to {final_model_path}")
            print(f"Best score: {global_best_score:.4f}")
        
        return best_policy, history
    
    def evaluate_policy(self, policy, n_episodes=5, render=False, verbose=True):
        """
        Evaluate a trained policy on the environment.
        
        Args:
            policy (CIRLPolicyNetwork): The policy to evaluate
            n_episodes (int): Number of episodes to evaluate for each schedule
            render (bool): Whether to render the environment
            verbose (bool): Whether to print progress information
            
        Returns:
            dict: Evaluation results
        """
        policy.eval()  # Set to evaluation mode
        
        # Define test setpoint schedules with focus on optimal Cb range
        test_schedules = [
            {"name": "Increasing", "setpoints": [0.65, 0.7, 0.75]},
            {"name": "Decreasing", "setpoints": [0.75, 0.7, 0.65]},
            {"name": "Peak", "setpoints": [0.70, 0.75, 0.70]},
            {"name": "Valley", "setpoints": [0.75, 0.70, 0.75]}
        ]
        
        results = {
            "schedules": {},
            "overall": {
                "rewards": [],
                "mse_cb": [],
                "mse_v": []
            }
        }
        
        # For each test schedule
        for schedule_idx, schedule in enumerate(test_schedules):
            schedule_name = schedule["name"]
            setpoints_Cb = schedule["setpoints"]
            
            if verbose:
                print(f"\nEvaluating on {schedule_name} schedule: {setpoints_Cb}")
            
            schedule_results = {
                "rewards": [],
                "mse_cb": [],
                "mse_v": [],
                "trajectories": []
            }
            
            # Run multiple episodes
            for episode in range(n_episodes):
                # Configure environment
                setpoints_V = [100.0] * len(setpoints_Cb)
                setpoint_durations = [50] * len(setpoints_Cb)
                
                # Reset environment
                state, _ = self.env.reset(options={
                    'setpoints_Cb': setpoints_Cb,
                    'setpoints_V': setpoints_V,
                    'setpoint_durations': setpoint_durations
                })
                
                # Initialize episode variables
                done = False
                episode_reward = 0
                step = 0
                
                # For tracking the episode
                trajectory = {
                    'states': [],
                    'actions': [],
                    'pid_gains': [],
                    'rewards': [],
                    'setpoints_Cb': [],
                    'setpoints_V': [],
                    'Cb': [],
                    'V': [],
                    'T': []
                }
                
                # Run episode
                while not done:
                    # Convert state to tensor
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    
                    # Get action from policy
                    with torch.no_grad():
                        action = policy(state_tensor).cpu().numpy()[0]
                    
                    # Step the environment
                    next_state, reward, done, truncated, info = self.env.step(action)
                    
                    # Extract info
                    pid_gains = ((action + 1) / 2) * (self.pid_upper - self.pid_lower) + self.pid_lower
                    true_state = info.get("true_state", None)
                    
                    # Store data
                    trajectory['states'].append(state.copy())
                    trajectory['actions'].append(action.copy())
                    trajectory['pid_gains'].append(pid_gains.copy())
                    trajectory['rewards'].append(reward)
                    
                    # Get current setpoints and values
                    current_setpoint_Cb = next_state[9]   # Index for current setpoint Cb
                    current_setpoint_V = next_state[10]   # Index for current setpoint V
                    
                    if true_state is not None:
                        current_Cb = true_state[1]  # Index 1 is Cb
                        current_V = true_state[4]   # Index 4 is V
                        current_T = true_state[3]   # Index 3 is T
                    else:
                        # If true_state not available, use measured state
                        current_Cb = next_state[0]  # Index 0 is current Cb
                        current_V = next_state[2]   # Index 2 is current V
                        current_T = next_state[1]   # Index 1 is current T
                    
                    trajectory['setpoints_Cb'].append(current_setpoint_Cb)
                    trajectory['setpoints_V'].append(current_setpoint_V)
                    trajectory['Cb'].append(current_Cb)
                    trajectory['V'].append(current_V)
                    trajectory['T'].append(current_T)
                    
                    # Render if requested
                    if render:
                        self.env.render()
                    
                    # Update state and counters
                    state = next_state
                    episode_reward += reward
                    step += 1
                    
                    if done or truncated:
                        break
                
                # Calculate MSE for this episode
                mse_cb = np.mean((np.array(trajectory['Cb']) - np.array(trajectory['setpoints_Cb'])) ** 2)
                mse_v = np.mean((np.array(trajectory['V']) - np.array(trajectory['setpoints_V'])) ** 2)
                
                # Store results
                schedule_results["rewards"].append(episode_reward)
                schedule_results["mse_cb"].append(mse_cb)
                schedule_results["mse_v"].append(mse_v)
                schedule_results["trajectories"].append(trajectory)
                
                # Also add to overall results
                results["overall"]["rewards"].append(episode_reward)
                results["overall"]["mse_cb"].append(mse_cb)
                results["overall"]["mse_v"].append(mse_v)
                
                if verbose:
                    print(f"  Episode {episode+1}: Reward = {episode_reward:.2f}, "
                          f"MSE Cb = {mse_cb:.6f}, MSE V = {mse_v:.6f}")
            
            # Calculate mean results for this schedule
            mean_reward = np.mean(schedule_results["rewards"])
            mean_mse_cb = np.mean(schedule_results["mse_cb"])
            mean_mse_v = np.mean(schedule_results["mse_v"])
            
            if verbose:
                print(f"  Schedule {schedule_name} results:")
                print(f"    Mean reward: {mean_reward:.2f}")
                print(f"    Mean MSE Cb: {mean_mse_cb:.6f}")
                print(f"    Mean MSE V: {mean_mse_v:.6f}")
            
            # Store schedule results
            results["schedules"][schedule_name] = schedule_results
        
        # Calculate overall results
        overall_mean_reward = np.mean(results["overall"]["rewards"])
        overall_mean_mse_cb = np.mean(results["overall"]["mse_cb"])
        overall_mean_mse_v = np.mean(results["overall"]["mse_v"])
        
        if verbose:
            print("\nOverall evaluation results:")
            print(f"  Mean reward: {overall_mean_reward:.2f}")
            print(f"  Mean MSE Cb: {overall_mean_mse_cb:.6f}")
            print(f"  Mean MSE V: {overall_mean_mse_v:.6f}")
        
        # Save results
        results_path = os.path.join(self.save_dir, "evaluation_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Plot results
        self._plot_evaluation_results(results)
        
        return results
    
    def _plot_evaluation_results(self, results):
        """
        Plot evaluation results.
        
        Args:
            results (dict): Evaluation results
        """
        plots_dir = os.path.join(self.save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot MSE by schedule
        plt.figure(figsize=(12, 8))
        
        # Collect data for bar plots
        schedule_names = list(results["schedules"].keys())
        mean_mse_cb = [np.mean(results["schedules"][name]["mse_cb"]) for name in schedule_names]
        mean_mse_v = [np.mean(results["schedules"][name]["mse_v"]) for name in schedule_names]
        mean_rewards = [np.mean(results["schedules"][name]["rewards"]) for name in schedule_names]
        
        # Plot MSE Cb by schedule
        plt.subplot(2, 2, 1)
        plt.bar(schedule_names, mean_mse_cb)
        plt.title("Mean MSE for Cb by Schedule")
        plt.ylabel("MSE")
        plt.grid(axis='y')
        
        # Plot MSE V by schedule
        plt.subplot(2, 2, 2)
        plt.bar(schedule_names, mean_mse_v)
        plt.title("Mean MSE for V by Schedule")
        plt.ylabel("MSE")
        plt.grid(axis='y')
        
        # Plot rewards by schedule
        plt.subplot(2, 2, 3)
        plt.bar(schedule_names, mean_rewards)
        plt.title("Mean Reward by Schedule")
        plt.ylabel("Reward")
        plt.grid(axis='y')
        
        # Plot trajectory for a sample episode
        # Find the best performing episode
        best_schedule = schedule_names[np.argmax(mean_rewards)]
        best_episode_idx = np.argmax(results["schedules"][best_schedule]["rewards"])
        best_trajectory = results["schedules"][best_schedule]["trajectories"][best_episode_idx]
        
        plt.subplot(2, 2, 4)
        plt.plot(best_trajectory["Cb"], 'b-', label='Actual Cb')
        plt.plot(best_trajectory["setpoints_Cb"], 'r--', label='Setpoint Cb')
        plt.title(f"Best Trajectory ({best_schedule})")
        plt.xlabel("Step")
        plt.ylabel("Concentration")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "evaluation_summary.png"))
        plt.close()
        
        # Plot detailed trajectory for the best episode
        plt.figure(figsize=(15, 10))
        
        # Plot Cb tracking
        plt.subplot(2, 2, 1)
        plt.plot(best_trajectory["Cb"], 'b-', label='Actual Cb')
        plt.plot(best_trajectory["setpoints_Cb"], 'r--', label='Setpoint Cb')
        plt.title("Concentration B Tracking")
        plt.xlabel("Step")
        plt.ylabel("Concentration")
        plt.legend()
        plt.grid(True)
        
        # Plot V tracking
        plt.subplot(2, 2, 2)
        plt.plot(best_trajectory["V"], 'g-', label='Actual V')
        plt.plot(best_trajectory["setpoints_V"], 'r--', label='Setpoint V')
        plt.title("Volume Tracking")
        plt.xlabel("Step")
        plt.ylabel("Volume")
        plt.legend()
        plt.grid(True)
        
        # Plot temperature
        plt.subplot(2, 2, 3)
        plt.plot(best_trajectory["T"], 'm-', label='Temperature')
        plt.title("Reactor Temperature")
        plt.xlabel("Step")
        plt.ylabel("Temperature (K)")
        plt.legend()
        plt.grid(True)
        
        # Plot PID gains
        plt.subplot(2, 2, 4)
        pid_gains = np.array(best_trajectory["pid_gains"])
        plt.plot(pid_gains[:, 0], 'r-', label='Kp_Cb')
        plt.plot(pid_gains[:, 1], 'g-', label='Ki_Cb')
        plt.plot(pid_gains[:, 2], 'b-', label='Kd_Cb')
        plt.plot(pid_gains[:, 3], 'c-', label='Kp_V')
        plt.plot(pid_gains[:, 4], 'm-', label='Ki_V')
        plt.plot(pid_gains[:, 5], 'y-', label='Kd_V')
        plt.title("Adaptive PID Gains")
        plt.xlabel("Step")
        plt.ylabel("Gain Value")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "best_trajectory_details.png"))
        plt.close()
    
    def train_combined(self, dataset_path=None, hidden_dims=[128, 128], n_supervised_epochs=50, 
                     n_pso_iterations=50, batch_size=256, learning_rate=3e-4, weight_decay=1e-4,
                     num_particles=20, min_param=-0.05, max_param=0.05, verbose=True):
        """
        Train CIRL policy using the combined approach:
        1. Load or generate dataset
        2. Pretrain with supervised learning
        3. Fine-tune with PSO
        4. Evaluate the policy
        
        Args:
            dataset_path (str): Path to existing dataset (if None, will use default or generate)
            hidden_dims (list): Dimensions of hidden layers
            n_supervised_epochs (int): Number of epochs for supervised learning
            n_pso_iterations (int): Number of iterations for PSO optimization
            batch_size (int): Batch size for supervised learning
            learning_rate (float): Learning rate for optimizer
            weight_decay (float): Weight decay for regularization
            num_particles (int): Number of particles for PSO
            min_param (float): Minimum parameter value for PSO
            max_param (float): Maximum parameter value for PSO
            verbose (bool): Whether to print progress information
            
        Returns:
            CIRLPolicyNetwork: The trained policy
        """
        # Track total training time
        start_time = time()
        
        # 1. Load or generate dataset
        if verbose:
            print("=== Phase 0: Preparing Dataset ===")
        
        if dataset_path:
            # Load specific dataset
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            if verbose:
                print(f"Loaded dataset from {dataset_path} with {len(dataset['states'])} transitions")
        else:
            # Try to load default dataset or generate new one
            dataset = self.load_offline_dataset(verbose=verbose)
        
        # 2. Create policy network
        policy = self.create_policy_network(hidden_dims=hidden_dims)
        
        # 3. Supervised pretraining
        if verbose:
            print("\n=== Phase 1: Supervised Learning ===")
        
        self.train_supervised(
            policy=policy,
            dataset=dataset,
            n_epochs=n_supervised_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            save_interval=10,
            verbose=verbose
        )
        
        # 4. PSO optimization
        if verbose:
            print("\n=== Phase 2: PSO Optimization ===")
        
        best_policy, _ = self.train_pso(
            policy_template=policy,
            n_iterations=n_pso_iterations,
            num_particles=num_particles,
            min_param=min_param,
            max_param=max_param,
            verbose=verbose
        )
        
        # 5. Final evaluation
        if verbose:
            print("\n=== Phase 3: Final Evaluation ===")
        
        self.evaluate_policy(
            policy=best_policy,
            n_episodes=5,
            render=False,
            verbose=verbose
        )
        
        # Calculate total training time
        training_time = time() - start_time
        
        # Save final model
        final_model_path = os.path.join(self.save_dir, "models", "cirl_policy_final.pt")
        best_policy.save(final_model_path)
        
        if verbose:
            print(f"\nTraining completed in {training_time:.2f} seconds!")
            print(f"Final model saved to: {final_model_path}")
        
        return best_policy


# Example usage 
if __name__ == "__main__":
    # Create environment with minimal noise/disturbance
    env = CSTRRLEnv(
        simulation_steps=150,
        dt=1.0,
        uncertainty_level=0.0,     # No uncertainty
        noise_level=0.0,           # No measurement noise
        actuator_delay_steps=0,    # No actuator delay
        transport_delay_steps=0,   # No transport delay
        enable_disturbances=False  # No disturbances
    )
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create trainer
    trainer = CIRLTrainer(env, save_dir="./results/cirl_training")
    
    # Train policy (shortened parameters for quick example)
    best_policy = trainer.train_combined(
        hidden_dims=[128, 128],
        n_supervised_epochs=30,
        n_pso_iterations=30,
        batch_size=256,
        verbose=True
    )
    
    print("CIRL training complete.")