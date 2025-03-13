"""
fine_tune_RL_online.py - Online fine-tuning of RL agents for CSTR control

This script enables online fine-tuning of pre-trained RL agents (from offline training)
through direct interaction with the CSTR environment. It includes:
1. Loading pre-trained agents
2. Setting up more challenging environments
3. Online fine-tuning with exploration
4. Evaluation and performance comparison
5. Visualization of improvements
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import time
from tqdm import tqdm
import random

from CSTR_model_plus import CSTRRLEnv
from RL_algorithms import create_agent
from Replay_Buffer import ReplayBuffer
from Offline_training import evaluate_agent, set_seed
from policy_network import CIRLNetwork


def set_up_env(difficulty='medium'):
    """
    Set up CSTR environment with different difficulty levels.
    
    Args:
        difficulty (str): Difficulty level ('easy', 'medium', 'hard', or 'extreme')
        
    Returns:
        object: Configured CSTR environment
    """
    # Base parameters
    params = {
        'simulation_steps': 200,
        'dt': 1.0,
        'uncertainty_level': 0.0,
        'noise_level': 0.0,
        'actuator_delay_steps': 0,
        'transport_delay_steps': 0,
        'enable_disturbances': False
    }
    
    # Adjust parameters based on difficulty
    if difficulty == 'easy':
        params['uncertainty_level'] = 0.02
        params['noise_level'] = 0.01
    
    elif difficulty == 'medium':
        params['uncertainty_level'] = 0.05
        params['noise_level'] = 0.02
        params['actuator_delay_steps'] = 1
        params['transport_delay_steps'] = 1
        params['enable_disturbances'] = True
    
    elif difficulty == 'hard':
        params['uncertainty_level'] = 0.1
        params['noise_level'] = 0.05
        params['actuator_delay_steps'] = 2
        params['transport_delay_steps'] = 2
        params['enable_disturbances'] = True
    
    elif difficulty == 'extreme':
        params['uncertainty_level'] = 0.15
        params['noise_level'] = 0.08
        params['actuator_delay_steps'] = 3
        params['transport_delay_steps'] = 3
        params['enable_disturbances'] = True
    
    # Create and return environment
    env = CSTRRLEnv(**params)
    return env


def load_agent(agent_type, model_path, state_dim, action_dim, action_high, device):
    """
    Load a pre-trained agent from file.
    
    Args:
        agent_type (str): Type of agent ('td3', 'sac', or 'cirl')
        model_path (str): Path to model file
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        action_high (float): Maximum action value
        device (str): Device for torch tensors
        
    Returns:
        object: Loaded agent
    """
    # Create agent with default parameters
    agent = create_agent(agent_type, state_dim, action_dim, action_high, device)
    
    # Load parameters from file
    try:
        agent.load(model_path)
        print(f"Successfully loaded {agent_type.upper()} model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Starting with a freshly initialized model")
    
    return agent


def online_fine_tune(agent, env, n_episodes=100, max_steps=200, 
                     batch_size=64, buffer_size=int(1e5),
                     init_exploration=0.3, final_exploration=0.1,
                     save_dir="./results/fine_tuned", save_interval=10,
                     eval_interval=5, eval_episodes=3, render=False, 
                     device="cuda", verbose=True):
    """
    Fine-tune a pre-trained agent with online environment interaction.
    
    Args:
        agent: Pre-trained RL agent
        env: CSTR environment
        n_episodes (int): Number of episodes for fine-tuning
        max_steps (int): Maximum steps per episode
        batch_size (int): Batch size for updates
        buffer_size (int): Size of replay buffer
        init_exploration (float): Initial exploration noise
        final_exploration (float): Final exploration noise
        save_dir (str): Directory to save results
        save_interval (int): Interval for saving models
        eval_interval (int): Interval for evaluation
        eval_episodes (int): Number of episodes for evaluation
        render (bool): Whether to render the environment
        device (str): Device for torch tensors
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Fine-tuning statistics
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    model_dir = os.path.join(save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create replay buffer for online data
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    online_buffer = ReplayBuffer(buffer_size, state_dim, action_dim, device)
    
    # Training statistics
    stats = {
        'episode_rewards': [],
        'episode_steps': [],
        'eval_rewards': [],
        'eval_steps': [],
        'actor_losses': [],
        'critic_losses': [],
        'alpha_losses': [] if hasattr(agent, 'automatic_entropy_tuning') else None,
        'learning_curves': {'x': [], 'y': [], 'std': []}
    }
    
    # Determine agent type
    if hasattr(agent, 'automatic_entropy_tuning'):
        agent_type = 'sac'
    elif isinstance(agent.actor, CIRLNetwork):
        agent_type = 'cirl'
    else:
        agent_type = 'td3'
    
    # Setup progress bar
    progress_bar = tqdm(range(n_episodes)) if verbose else range(n_episodes)
    
    # Run fine-tuning episodes
    for episode in progress_bar:
        # Calculate exploration noise for this episode (linear decay)
        progress = episode / n_episodes
        exploration_noise = init_exploration - progress * (init_exploration - final_exploration)
        
        # Reset environment
        # Generate a random setpoint schedule for this episode
        n_setpoints = np.random.randint(2, 5)  # Random number of setpoints (2-4)
        setpoints_Cb = np.random.uniform(0.2, 0.8, n_setpoints).tolist()
        setpoints_V = [100.0] * n_setpoints  # Volume setpoint constant
        setpoint_durations = [max_steps // n_setpoints] * n_setpoints
        
        state, _ = env.reset(options={
            'setpoints_Cb': setpoints_Cb,
            'setpoints_V': setpoints_V,
            'setpoint_durations': setpoint_durations
        })
        
        episode_reward = 0
        episode_steps = 0
        episode_actor_losses = []
        episode_critic_losses = []
        episode_alpha_losses = []
        
        # Run an episode
        done = False
        while not done and episode_steps < max_steps:
            # Select action with exploration noise
            if agent_type == 'sac':
                # For SAC, no explicit noise is added as it's a stochastic policy
                action = agent.select_action(state, evaluate=False)
            else:
                # For TD3 and CIRL, add exploration noise
                action = agent.select_action(state, noise=exploration_noise)
            
            # Take step in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store transition in replay buffer
            online_buffer.add(state, action, reward, next_state, done or truncated)
            
            # Update agent if enough data is collected
            if online_buffer.size > batch_size * 10:  # Start training after collecting some data
                if agent_type == 'sac':
                    actor_loss, critic_loss, alpha_loss = agent.train(online_buffer, batch_size)
                    episode_actor_losses.append(actor_loss)
                    episode_critic_losses.append(critic_loss)
                    episode_alpha_losses.append(alpha_loss)
                else:  # TD3 or CIRL
                    actor_loss, critic_loss = agent.train(online_buffer, batch_size)
                    if actor_loss is not None:
                        episode_actor_losses.append(actor_loss)
                    episode_critic_losses.append(critic_loss)
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            if done or truncated:
                break
            
            if render:
                env.render()
        
        # Store episode statistics
        stats['episode_rewards'].append(episode_reward)
        stats['episode_steps'].append(episode_steps)
        
        # Store loss statistics if available
        if episode_actor_losses:
            stats['actor_losses'].extend(episode_actor_losses)
        if episode_critic_losses:
            stats['critic_losses'].extend(episode_critic_losses)
        if stats['alpha_losses'] is not None and episode_alpha_losses:
            stats['alpha_losses'].extend(episode_alpha_losses)
        
        # Update progress bar
        if verbose:
            progress_bar.set_description(
                f"Episode {episode+1}/{n_episodes} | Reward: {episode_reward:.2f} | "
                f"Buffer: {online_buffer.size}/{buffer_size} | "
                f"Noise: {exploration_noise:.3f}"
            )
        
        # Periodic evaluation
        if (episode + 1) % eval_interval == 0 or episode == n_episodes - 1:
            eval_reward, eval_steps = evaluate_agent(
                agent, env, n_episodes=eval_episodes, 
                max_steps=max_steps, render=False, verbose=False
            )
            
            stats['eval_rewards'].append(eval_reward)
            stats['eval_steps'].append(eval_steps)
            stats['learning_curves']['x'].append(episode + 1)
            stats['learning_curves']['y'].append(eval_reward)
            
            if verbose:
                print(f"\nEvaluation at episode {episode+1}: Reward = {eval_reward:.2f}, Steps = {eval_steps:.1f}")
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0 or episode == n_episodes - 1:
            checkpoint_path = os.path.join(model_dir, f"{agent_type}_episode_{episode+1}")
            agent.save(checkpoint_path)
            if verbose:
                print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(save_dir, f"{agent_type}_fine_tuned_final")
    agent.save(final_model_path)
    
    # Save statistics
    stats_path = os.path.join(save_dir, f"{agent_type}_fine_tuning_stats.pkl")
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(stats['learning_curves']['x'], stats['learning_curves']['y'], 'b-', marker='o')
    plt.title(f"{agent_type.upper()} Fine-Tuning Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Evaluation Reward")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{agent_type}_learning_curve.png"))
    
    # Plot training/loss curves
    plt.figure(figsize=(15, 5))
    
    # Episode rewards
    plt.subplot(1, 3, 1)
    plt.plot(range(1, n_episodes+1), stats['episode_rewards'])
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    
    # Actor loss
    plt.subplot(1, 3, 2)
    if stats['actor_losses']:
        plt.plot(range(len(stats['actor_losses'])), stats['actor_losses'])
        plt.title("Actor Loss")
        plt.xlabel("Update")
        plt.ylabel("Loss")
        plt.grid(True)
    else:
        plt.title("No Actor Loss Data")
    
    # Critic loss
    plt.subplot(1, 3, 3)
    plt.plot(range(len(stats['critic_losses'])), stats['critic_losses'])
    plt.title("Critic Loss")
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{agent_type}_training_curves.png"))
    
    if verbose:
        print(f"Fine-tuning complete. Final model saved to {final_model_path}")
    
    return stats


def compare_before_after(pre_trained_agent, fine_tuned_agent, env, 
                         n_episodes=10, max_steps=200, 
                         save_dir="./results/comparison", verbose=True):
    """
    Compare performance before and after fine-tuning.
    
    Args:
        pre_trained_agent: Pre-trained agent before fine-tuning
        fine_tuned_agent: Fine-tuned agent
        env: Environment for evaluation
        n_episodes (int): Number of episodes for evaluation
        max_steps (int): Maximum steps per episode
        save_dir (str): Directory to save results
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Comparison results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Define test scenarios (different setpoint schedules)
    test_scenarios = [
        {"name": "Increasing", "setpoints": [0.25, 0.50, 0.75]},
        {"name": "Decreasing", "setpoints": [0.75, 0.50, 0.25]},
        {"name": "Peak", "setpoints": [0.25, 0.80, 0.25]},
        {"name": "Valley", "setpoints": [0.75, 0.20, 0.75]},
        {"name": "Random", "setpoints": np.random.uniform(0.2, 0.8, 3).tolist()}
    ]
    
    # Results dictionary
    results = {
        "pre_trained": {scenario["name"]: [] for scenario in test_scenarios},
        "fine_tuned": {scenario["name"]: [] for scenario in test_scenarios},
        "summary": {}
    }
    
    # Test agents on each scenario
    for i, scenario in enumerate(test_scenarios):
        scenario_name = scenario["name"]
        setpoints = scenario["setpoints"]
        
        if verbose:
            print(f"\nTesting scenario {i+1}/{len(test_scenarios)}: {scenario_name} {setpoints}")
        
        # Test pre-trained agent
        pre_trained_rewards = []
        for episode in range(n_episodes):
            # Reset environment with scenario setpoints
            state, _ = env.reset(options={
                'setpoints_Cb': setpoints,
                'setpoints_V': [100.0] * len(setpoints),
                'setpoint_durations': [max_steps // len(setpoints)] * len(setpoints)
            })
            
            total_reward = 0
            done = False
            step = 0
            
            # Run episode
            while not done and step < max_steps:
                # Select action (evaluation mode)
                if hasattr(pre_trained_agent, 'automatic_entropy_tuning'):  # SAC
                    action = pre_trained_agent.select_action(state, evaluate=True)
                else:  # TD3 or CIRL
                    action = pre_trained_agent.select_action(state, noise=0)
                
                # Take step
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Update
                state = next_state
                total_reward += reward
                step += 1
                
                if done or truncated:
                    break
            
            pre_trained_rewards.append(total_reward)
        
        # Store pre-trained results
        results["pre_trained"][scenario_name] = pre_trained_rewards
        
        # Test fine-tuned agent
        fine_tuned_rewards = []
        for episode in range(n_episodes):
            # Reset environment with scenario setpoints
            state, _ = env.reset(options={
                'setpoints_Cb': setpoints,
                'setpoints_V': [100.0] * len(setpoints),
                'setpoint_durations': [max_steps // len(setpoints)] * len(setpoints)
            })
            
            total_reward = 0
            done = False
            step = 0
            
            # Run episode
            while not done and step < max_steps:
                # Select action (evaluation mode)
                if hasattr(fine_tuned_agent, 'automatic_entropy_tuning'):  # SAC
                    action = fine_tuned_agent.select_action(state, evaluate=True)
                else:  # TD3 or CIRL
                    action = fine_tuned_agent.select_action(state, noise=0)
                
                # Take step
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Update
                state = next_state
                total_reward += reward
                step += 1
                
                if done or truncated:
                    break
            
            fine_tuned_rewards.append(total_reward)
        
        # Store fine-tuned results
        results["fine_tuned"][scenario_name] = fine_tuned_rewards
        
        # Calculate improvement
        pre_trained_mean = np.mean(pre_trained_rewards)
        fine_tuned_mean = np.mean(fine_tuned_rewards)
        improvement = ((fine_tuned_mean - pre_trained_mean) / abs(pre_trained_mean)) * 100
        
        if verbose:
            print(f"  Pre-trained: {pre_trained_mean:.2f} ± {np.std(pre_trained_rewards):.2f}")
            print(f"  Fine-tuned:  {fine_tuned_mean:.2f} ± {np.std(fine_tuned_rewards):.2f}")
            print(f"  Improvement: {improvement:.2f}%")
    
    # Calculate overall summary
    all_pre_trained = [item for sublist in results["pre_trained"].values() for item in sublist]
    all_fine_tuned = [item for sublist in results["fine_tuned"].values() for item in sublist]
    
    results["summary"] = {
        "pre_trained_mean": np.mean(all_pre_trained),
        "pre_trained_std": np.std(all_pre_trained),
        "fine_tuned_mean": np.mean(all_fine_tuned),
        "fine_tuned_std": np.std(all_fine_tuned),
        "improvement": ((np.mean(all_fine_tuned) - np.mean(all_pre_trained)) / abs(np.mean(all_pre_trained))) * 100
    }
    
    if verbose:
        print("\nOverall Results:")
        print(f"  Pre-trained: {results['summary']['pre_trained_mean']:.2f} ± {results['summary']['pre_trained_std']:.2f}")
        print(f"  Fine-tuned:  {results['summary']['fine_tuned_mean']:.2f} ± {results['summary']['fine_tuned_std']:.2f}")
        print(f"  Improvement: {results['summary']['improvement']:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Bar plot for each scenario
    scenario_names = list(results["pre_trained"].keys())
    x = np.arange(len(scenario_names))
    width = 0.35
    
    pre_trained_means = [np.mean(results["pre_trained"][name]) for name in scenario_names]
    pre_trained_stds = [np.std(results["pre_trained"][name]) for name in scenario_names]
    
    fine_tuned_means = [np.mean(results["fine_tuned"][name]) for name in scenario_names]
    fine_tuned_stds = [np.std(results["fine_tuned"][name]) for name in scenario_names]
    
    plt.bar(x - width/2, pre_trained_means, width, label='Pre-trained', color='skyblue', yerr=pre_trained_stds, capsize=5)
    plt.bar(x + width/2, fine_tuned_means, width, label='Fine-tuned', color='orange', yerr=fine_tuned_stds, capsize=5)
    
    plt.ylabel('Mean Reward')
    plt.title('Performance Comparison: Pre-trained vs. Fine-tuned')
    plt.xticks(x, scenario_names)
    plt.legend()
    plt.grid(axis='y')
    
    # Add improvement percentage labels
    for i, name in enumerate(scenario_names):
        pre_mean = np.mean(results["pre_trained"][name])
        fine_mean = np.mean(results["fine_tuned"][name])
        improvement = ((fine_mean - pre_mean) / abs(pre_mean)) * 100
        
        plt.text(i, max(pre_mean, fine_mean) + 1, f"{improvement:.1f}%", 
                 ha='center', va='bottom', fontweight='bold', 
                 color='green' if improvement > 0 else 'red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_bar_plot.png"))
    
    # Save results
    with open(os.path.join(save_dir, "comparison_results.pkl"), 'wb') as f:
        pickle.dump(results, f)
    
    return results


def main():
    """Main function for parsing arguments and running fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune RL agents online")
    
    # Basic configuration
    parser.add_argument("--agent_type", type=str, default="td3", choices=["td3", "sac", "cirl"], 
                        help="Type of agent to fine-tune")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to pre-trained model")
    parser.add_argument("--difficulty", type=str, default="medium", 
                        choices=["easy", "medium", "hard", "extreme"], 
                        help="Difficulty level of environment")
    parser.add_argument("--save_dir", type=str, default="./results/fine_tuned", 
                        help="Directory to save results")
    
    # Fine-tuning parameters
    parser.add_argument("--n_episodes", type=int, default=100, 
                        help="Number of episodes for fine-tuning")
    parser.add_argument("--max_steps", type=int, default=200, 
                        help="Maximum steps per episode")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size for updates")
    parser.add_argument("--buffer_size", type=int, default=100000, 
                        help="Size of replay buffer")
    parser.add_argument("--init_noise", type=float, default=0.3, 
                        help="Initial exploration noise")
    parser.add_argument("--final_noise", type=float, default=0.1, 
                        help="Final exploration noise")
    
    # Evaluation parameters
    parser.add_argument("--eval_interval", type=int, default=5, 
                        help="Interval for evaluation")
    parser.add_argument("--eval_episodes", type=int, default=3, 
                        help="Number of episodes for evaluation")
    parser.add_argument("--save_interval", type=int, default=10, 
                        help="Interval for saving models")
    
    # Technical parameters
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", 
                        choices=["cuda", "cpu"], 
                        help="Device for training")
    parser.add_argument("--render", action="store_true", 
                        help="Render environment during training")
    parser.add_argument("--compare", action="store_true", 
                        help="Compare pre-trained and fine-tuned agents")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up environment
    env = set_up_env(args.difficulty)
    print(f"Created {args.difficulty} difficulty environment")
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high[0]
    
    # Load pre-trained agent
    pre_trained_agent = load_agent(
        args.agent_type, args.model_path, 
        state_dim, action_dim, action_high, device
    )
    
    # Evaluate pre-trained agent
    print("Evaluating pre-trained agent...")
    pre_trained_reward, pre_trained_steps = evaluate_agent(
        pre_trained_agent, env, n_episodes=5, max_steps=args.max_steps, verbose=True
    )
    print(f"Pre-trained agent: Reward = {pre_trained_reward:.2f}, Steps = {pre_trained_steps:.1f}")
    
    # Fine-tune agent
    print(f"\nStarting fine-tuning for {args.n_episodes} episodes...")
    fine_tuning_stats = online_fine_tune(
        agent=pre_trained_agent,  # This will be modified in-place
        env=env,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        init_exploration=args.init_noise,
        final_exploration=args.final_noise,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        render=args.render,
        device=device,
        verbose=True
    )
    
    # Evaluate fine-tuned agent
    print("\nEvaluating fine-tuned agent...")
    fine_tuned_reward, fine_tuned_steps = evaluate_agent(
        pre_trained_agent, env, n_episodes=5, max_steps=args.max_steps, verbose=True
    )
    print(f"Fine-tuned agent: Reward = {fine_tuned_reward:.2f}, Steps = {fine_tuned_steps:.1f}")
    
    # Calculate improvement
    improvement = ((fine_tuned_reward - pre_trained_reward) / abs(pre_trained_reward)) * 100
    print(f"Improvement: {improvement:.2f}%")
    
    # Compare pre-trained and fine-tuned agents
    if args.compare:
        print("\nComparing pre-trained and fine-tuned agents in detail...")
        
        # Load a fresh copy of the pre-trained agent for comparison
        fresh_pre_trained = load_agent(
            args.agent_type, args.model_path, 
            state_dim, action_dim, action_high, device
        )
        
        # The fine-tuned agent is already in pre_trained_agent (modified in-place)
        comparison_results = compare_before_after(
            fresh_pre_trained, pre_trained_agent, env,
            n_episodes=5, max_steps=args.max_steps,
            save_dir=os.path.join(args.save_dir, "comparison"),
            verbose=True
        )
    
    print("\nFine-tuning process complete!")


if __name__ == "__main__":
    main()