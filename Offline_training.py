"""
Offline_training.py - Train RL agents using pre-collected data

This module provides functionality for offline training of reinforcement learning
agents using pre-collected data. It includes:
1. Functions for loading and preprocessing datasets
2. Offline training procedures for TD3, SAC, and CIRL algorithms
3. Evaluation of trained agents on the CSTR environment
"""

import os
import torch
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse

from CSTR_model_plus import CSTRRLEnv
from RL_algorithms import create_agent
from Replay_Buffer import ReplayBuffer
from gen_sim_data import DataGenerator


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_offline_dataset(env, save_dir="./data", verbose=True):
    """
    Generate an offline dataset spanning the full spectrum of setpoints.
    
    Args:
        env: The CSTR environment instance
        save_dir (str): Directory to save the dataset
        verbose (bool): Whether to show progress bars
        
    Returns:
        dict: Generated dataset
    """
    # Create data generator
    data_gen = DataGenerator(env, save_dir=save_dir)
    
    # Define setpoint schedule covering the full spectrum
    setpoint_Cb = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    setpoint_V = [100.0] * len(setpoint_Cb)
    steps_per_setpoint = 20  # Each setpoint for 20 timesteps
    
    # Generate dataset with multiple exploration strategies
    datasets = []
    
    # For each exploration strategy, generate a portion of the dataset
    strategies = ["random", "static_pid", "decaying", "mixed"]
    n_episodes_per_strategy = 5  # 5 episodes per strategy
    
    if verbose:
        print("Generating offline dataset...")
    
    for strategy in strategies:
        if verbose:
            print(f"Using {strategy} exploration strategy...")
        
        # Create custom schedules for this strategy
        custom_schedules = [setpoint_Cb]
        # Add some variations by shuffling and reversing
        shuffled_schedule = setpoint_Cb.copy()
        random.shuffle(shuffled_schedule)
        custom_schedules.append(shuffled_schedule)
        custom_schedules.append(list(reversed(setpoint_Cb)))
        
        # Generate dataset with this strategy
        dataset = data_gen.generate_dataset(
            n_episodes=n_episodes_per_strategy,
            steps_per_setpoint=steps_per_setpoint,
            exploration_strategy=strategy,
            custom_schedules=custom_schedules,
            verbose=verbose
        )
        
        datasets.append(dataset)
    
    # Combine datasets
    combined_dataset = data_gen.combine_datasets(datasets)
    
    # Analyze and save the dataset
    data_gen.analyze_dataset(combined_dataset, save_plots=True)
    data_gen.save_dataset(combined_dataset, filename="offline_full_spectrum_dataset.pkl")
    
    if verbose:
        print(f"Dataset generation complete: {len(combined_dataset['states'])} transitions")
    
    return combined_dataset


def load_offline_dataset(file_path, verbose=True):
    """
    Load an offline dataset from a file.
    
    Args:
        file_path (str): Path to the dataset file
        verbose (bool): Whether to print information
        
    Returns:
        dict: Loaded dataset
    """
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    
    if verbose:
        print(f"Loaded dataset from {file_path} with {len(dataset['states'])} transitions")
    
    return dataset


def offline_train(agent, replay_buffer, n_updates, batch_size=256, 
                  update_interval=1000, save_dir="./results", verbose=True):
    """
    Train an agent using offline data stored in a replay buffer.
    
    Args:
        agent: The RL agent to train
        replay_buffer: Replay buffer containing offline data
        n_updates (int): Number of gradient updates to perform
        batch_size (int): Batch size for gradient updates
        update_interval (int): Interval at which to save checkpoint models
        save_dir (str): Directory to save results
        verbose (bool): Whether to show progress bar
        
    Returns:
        dict: Training statistics
    """
    # Create directories for saving results
    os.makedirs(save_dir, exist_ok=True)
    model_dir = os.path.join(save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Determine agent type
    if hasattr(agent, 'automatic_entropy_tuning'):
        agent_type = 'sac'
    else:
        # Check for hidden_dims at the agent level (for CIRL)
        if hasattr(agent, 'hidden_dims') and isinstance(agent.hidden_dims, list) and agent.hidden_dims[0] <= 16:
            agent_type = 'cirl'
        else:
            agent_type = 'td3'
    
    # Training statistics
    stats = {
        'actor_losses': [],
        'critic_losses': [],
        'steps': []
    }
    
    # For SAC, also track alpha losses
    if agent_type == 'sac':
        stats['alpha_losses'] = []
    
    # Progress bar
    if verbose:
        pbar = tqdm(total=n_updates, desc=f"Training {agent_type.upper()}")
    
    # Training loop
    for i in range(n_updates):
        # Update networks
        if agent_type == 'sac':
            actor_loss, critic_loss, alpha_loss = agent.train(replay_buffer, batch_size)
            stats['actor_losses'].append(actor_loss)
            stats['critic_losses'].append(critic_loss)
            stats['alpha_losses'].append(alpha_loss)
        else:  # TD3 or CIRL
            actor_loss, critic_loss = agent.train(replay_buffer, batch_size)
            if actor_loss is not None:
                stats['actor_losses'].append(actor_loss)
            stats['critic_losses'].append(critic_loss)
        
        stats['steps'].append(i)
        
        # Save checkpoint models
        if (i + 1) % update_interval == 0:
            agent.save(os.path.join(model_dir, f"{agent_type}_offline_{i+1}"))
            
            # Update progress bar
            if verbose:
                pbar.update(update_interval)
                if len(stats['actor_losses']) > 0:
                    avg_actor_loss = np.mean(stats['actor_losses'][-update_interval:])
                    avg_critic_loss = np.mean(stats['critic_losses'][-update_interval:])
                    pbar.set_postfix({
                        'actor_loss': f"{avg_actor_loss:.4f}", 
                        'critic_loss': f"{avg_critic_loss:.4f}"
                    })
    
    # Save final model
    agent.save(os.path.join(model_dir, f"{agent_type}_offline_final"))
    
    # Close progress bar
    if verbose:
        pbar.close()
    
    # Save training statistics
    with open(os.path.join(save_dir, f"{agent_type}_offline_stats.pkl"), 'wb') as f:
        pickle.dump(stats, f)
    
    # Plot training curves
    if verbose:
        print("Plotting training curves...")
    
    plt.figure(figsize=(15, 5))
    
    # Plot actor loss
    plt.subplot(1, 2, 1)
    if len(stats['actor_losses']) > 0:
        # Only use the steps that correspond to actor loss updates
        actor_steps = stats['steps'][:len(stats['actor_losses'])]
        plt.plot(actor_steps, stats['actor_losses'])
        plt.title(f"{agent_type.upper()} Offline Training - Actor Loss")
        plt.xlabel("Update Step")
        plt.ylabel("Actor Loss")
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "No actor loss data available", 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f"{agent_type.upper()} - No Actor Loss Data")
    
    # Plot critic loss
    plt.subplot(1, 2, 2)
    # Only use the steps that correspond to critic loss updates
    critic_steps = stats['steps'][:len(stats['critic_losses'])]
    plt.plot(critic_steps, stats['critic_losses'])
    plt.title(f"{agent_type.upper()} Offline Training - Critic Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Critic Loss")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{agent_type}_offline_training_curves.png"))
    plt.close()
    
    return stats


def evaluate_agent(agent, env, n_episodes=5, max_steps=200, render=False, verbose=True):
    """
    Evaluate an agent on the environment.
    
    Args:
        agent: The RL agent to evaluate
        env: The environment to evaluate in
        n_episodes (int): Number of episodes to evaluate for
        max_steps (int): Maximum steps per episode
        render (bool): Whether to render the environment
        verbose (bool): Whether to print information
        
    Returns:
        tuple: (mean_reward, mean_steps)
    """
    episode_rewards = []
    episode_steps = []
    
    # Setpoints for evaluation
    evaluation_setpoints = [
        [0.12, 0.35, 0.48, 0.71, 0.84],  # Increasing steps (training schedule)
        [0.86, 0.73, 0.61, 0.48, 0.25],  # Decreasing steps
        [0.25, 0.30, 0.65, 0.30, 0.25],  # Peak
        [0.86, 0.71, 0.24, 0.63, 0.82]   # Valley
    ]
    
    # For each evaluation setpoint schedule
    for schedule_idx, setpoint_schedule in enumerate(evaluation_setpoints):
        if verbose:
            print(f"Evaluating on setpoint schedule {schedule_idx+1}: {setpoint_schedule}")
        
        schedule_rewards = []
        schedule_steps = []
        
        # Run multiple episodes for this schedule
        for episode in range(n_episodes):
            # Configure environment with setpoint schedule
            setpoints_Cb = setpoint_schedule
            setpoints_V = [100.0] * len(setpoint_schedule)
            setpoint_durations = [max_steps // len(setpoint_schedule)] * len(setpoint_schedule)
            
            # Reset environment
            state, _ = env.reset(options={
                'setpoints_Cb': setpoints_Cb,
                'setpoints_V': setpoints_V,
                'setpoint_durations': setpoint_durations
            })
            
            episode_reward = 0
            step = 0
            done = False
            
            while not done and step < max_steps:
                # Select action without noise (evaluation mode)
                if hasattr(agent, 'automatic_entropy_tuning'):  # SAC agent
                    action = agent.select_action(state, evaluate=True)
                else:  # TD3 or CIRL agent
                    action = agent.select_action(state, noise=0.0)
                
                # Take step in environment
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                
                if render:
                    env.render()
                
                # Update state and counters
                state = next_state
                episode_reward += reward
                step += 1
                
                if done or step >= max_steps:
                    break
            
            schedule_rewards.append(episode_reward)
            schedule_steps.append(step)
            
            if verbose:
                print(f"  Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {step}")
        
        # Append schedule results to overall results
        episode_rewards.extend(schedule_rewards)
        episode_steps.extend(schedule_steps)
        
        if verbose:
            mean_schedule_reward = np.mean(schedule_rewards)
            mean_schedule_steps = np.mean(schedule_steps)
            print(f"  Average for schedule {schedule_idx+1}: Reward = {mean_schedule_reward:.2f}, Steps = {mean_schedule_steps:.1f}")
    
    # Calculate overall means
    mean_reward = np.mean(episode_rewards)
    mean_steps = np.mean(episode_steps)
    
    if verbose:
        print(f"Overall Evaluation Results:")
        print(f"  Mean Reward: {mean_reward:.2f}")
        print(f"  Mean Steps: {mean_steps:.1f}")
    
    return mean_reward, mean_steps


def train_all_agents_offline(dataset=None, dataset_path=None, n_updates=50000, 
                             batch_size=256, update_interval=1000, save_dir="./results", 
                             seed=42, verbose=True):
    """
    Train TD3, SAC, and CIRL agents using offline data.
    
    Args:
        dataset (dict): Dataset to use (optional)
        dataset_path (str): Path to load dataset from (if dataset is None)
        n_updates (int): Number of updates to perform
        batch_size (int): Batch size for updates
        update_interval (int): Interval at which to save checkpoint models
        save_dir (str): Directory to save results
        seed (int): Random seed
        verbose (bool): Whether to print information
        
    Returns:
        dict: Training results for all agents
    """
    # Set random seed
    set_seed(seed)
    
    # Create directories for saving results
    os.makedirs(save_dir, exist_ok=True)
    
    # Load dataset if not provided
    if dataset is None:
        if dataset_path is None:
            raise ValueError("Either dataset or dataset_path must be provided")
        dataset = load_offline_dataset(dataset_path, verbose)
    
    # Create environment for evaluation
    env = CSTRRLEnv(
        simulation_steps=200,  # Should be enough for evaluation
        dt=1.0,
        uncertainty_level=0.0,  # No uncertainty for initial evaluation
        noise_level=0.0,        # No noise for initial evaluation
        actuator_delay_steps=0,  # No delays for initial evaluation
        transport_delay_steps=0,
        enable_disturbances=False  # No disturbances for initial evaluation
    )
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high[0]
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        capacity=len(dataset['states']) + 1000,  # Slightly larger than dataset size
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )
    
    # Load dataset into replay buffer
    replay_buffer.load_from_dataset(dataset)
    
    # Results dictionary
    results = {}
    
    # Train each agent
    for agent_type in ['td3', 'sac', 'cirl']:
    # for agent_type in ['td3', 'sac']:
        if verbose:
            print(f"\n===== Training {agent_type.upper()} Agent =====")
        
        # Create agent
        agent = create_agent(
            agent_type=agent_type,
            state_dim=state_dim,
            action_dim=action_dim,
            action_high=action_high,
            device=device
        )
        
        # Create agent-specific directory
        agent_dir = os.path.join(save_dir, agent_type)
        os.makedirs(agent_dir, exist_ok=True)
        
        # Train agent offline
        start_time = time.time()
        stats = offline_train(
            agent=agent,
            replay_buffer=replay_buffer,
            n_updates=n_updates,
            batch_size=batch_size,
            update_interval=update_interval,
            save_dir=agent_dir,
            verbose=verbose
        )
        training_time = time.time() - start_time
        
        # Evaluate agent
        mean_reward, mean_steps = evaluate_agent(
            agent=agent,
            env=env,
            n_episodes=3,
            max_steps=200,
            verbose=verbose
        )
        
        # Store results
        results[agent_type] = {
            'stats': stats,
            'mean_reward': mean_reward,
            'mean_steps': mean_steps,
            'training_time': training_time
        }
        
        if verbose:
            print(f"Training time: {training_time:.2f} seconds")
            print(f"Final evaluation: Reward = {mean_reward:.2f}, Steps = {mean_steps:.1f}")
    
    # Compare results
    if verbose:
        print("\n===== Comparing Agent Performance =====")
        for agent_type, result in results.items():
            print(f"{agent_type.upper()} agent: Reward = {result['mean_reward']:.2f}, Training time = {result['training_time']:.2f} s")
    
    # Save results
    with open(os.path.join(save_dir, "offline_training_results.pkl"), 'wb') as f:
        pickle.dump(results, f)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    agent_types = list(results.keys())
    rewards = [results[agent_type]['mean_reward'] for agent_type in agent_types]
    
    plt.bar(agent_types, rewards)
    plt.title("Offline Training - Agent Performance Comparison")
    plt.xlabel("Agent Type")
    plt.ylabel("Mean Evaluation Reward")
    plt.xticks([i for i in range(len(agent_types))], [agent.upper() for agent in agent_types])
    plt.grid(axis='y')
    
    for i, reward in enumerate(rewards):
        plt.text(i, reward + 0.5, f"{reward:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "agent_comparison.png"))
    plt.close()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline RL Training")
    parser.add_argument("--generate", action="store_true", help="Generate a new dataset")
    parser.add_argument("--dataset", type=str, default="./data/offline_full_spectrum_dataset.pkl", 
                       help="Path to dataset file")
    parser.add_argument("--updates", type=int, default=50000, help="Number of gradient updates")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for updates")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--agent", type=str, default="all", 
                       choices=["all", "td3", "sac", "cirl"], help="Agent type to train")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Generate dataset if requested
    if args.generate:
        print("Generating new dataset...")
        env = CSTRRLEnv(
            simulation_steps=200,
            dt=1.0,
            uncertainty_level=0.05,  # Add some uncertainty for realism
            noise_level=0.02,        # Add some noise for realism
            actuator_delay_steps=1,   # Add realistic delay
            transport_delay_steps=1,
            enable_disturbances=True  # Enable disturbances for robustness
        )
        dataset = generate_offline_dataset(env, save_dir="./data", verbose=True)
    else:
        dataset = None
    
    # If specified agent is "all", train all agents
    if args.agent == "all":
        results = train_all_agents_offline(
            dataset=dataset,
            dataset_path=args.dataset,
            n_updates=args.updates,
            batch_size=args.batch_size,
            save_dir=args.save_dir,
            seed=args.seed,
            verbose=True
        )
    else:
        # Train only the specified agent
        # Create environment
        env = CSTRRLEnv(
            simulation_steps=200,
            dt=1.0,
            uncertainty_level=0.0,
            noise_level=0.0,
            actuator_delay_steps=0,
            transport_delay_steps=0,
            enable_disturbances=False
        )
        
        # Get environment dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_high = env.action_space.high[0]
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create agent
        agent = create_agent(
            agent_type=args.agent,
            state_dim=state_dim,
            action_dim=action_dim,
            action_high=action_high,
            device=device
        )
        
        # Load dataset
        if dataset is None:
            dataset = load_offline_dataset(args.dataset, verbose=True)
        
        # Create replay buffer
        replay_buffer = ReplayBuffer(
            capacity=len(dataset['states']) + 1000,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device
        )
        
        # Load dataset into replay buffer
        replay_buffer.load_from_dataset(dataset)
        
        # Create agent-specific directory
        agent_dir = os.path.join(args.save_dir, args.agent)
        os.makedirs(agent_dir, exist_ok=True)
        
        # Train agent
        print(f"Training {args.agent.upper()} agent...")
        stats = offline_train(
            agent=agent,
            replay_buffer=replay_buffer,
            n_updates=args.updates,
            batch_size=args.batch_size,
            save_dir=agent_dir,
            verbose=True
        )
        
        # Evaluate agent
        mean_reward, mean_steps = evaluate_agent(
            agent=agent,
            env=env,
            n_episodes=5,
            max_steps=200,
            verbose=True
        )
        
        print(f"Final evaluation: Reward = {mean_reward:.2f}, Steps = {mean_steps:.1f}")
        
    print("Offline training complete!")