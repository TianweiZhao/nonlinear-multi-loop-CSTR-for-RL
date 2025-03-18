"""
train_pso_cirl.py - Combined PSO+CIRL training for CSTR control system

This script implements a complete training pipeline for CSTR control optimization:
1. PSO-based initialization of CIRL policy parameters
2. Offline RL training using generated dataset
3. Online fine-tuning with direct environment interaction
4. Performance evaluation against baseline static PID

The approach combines the global search capability of PSO with the adaptability
of reinforcement learning to find optimal control policies.
"""

import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt
import pickle
import argparse
from tqdm import tqdm

from CSTR_model_plus import CSTRRLEnv
from RL_algorithms import create_agent
from Offline_training import load_offline_dataset, offline_train, evaluate_agent, set_seed
from online_fine_tuning import online_fine_tune
from evaluate_controllers import ControllerEvaluator


class PSOParticle:
    """
    Represents a particle in the PSO algorithm for CIRL policy optimization.
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
    
    def update_velocity(self, global_best_position, w=0.9, c1=1.5, c2=1.5):
        """
        Update particle velocity based on PSO equations.
        
        Args:
            global_best_position: Global best position across all particles
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
        Update particle position based on velocity and enforce bounds.
        
        Args:
            bounds: List of tuples (min, max) for each dimension
        """
        self.position = self.position + self.velocity
        
        # Enforce bounds
        for i in range(len(self.position)):
            self.position[i] = np.clip(self.position[i], bounds[i][0], bounds[i][1])
    
    def evaluate(self, env, setpoints_Cb, setpoints_V, setpoint_durations):
        """
        Evaluate the particle by running a policy rollout in the environment.
        
        Args:
            env: The environment to evaluate on
            setpoints_Cb, setpoints_V, setpoint_durations: Setpoint parameters
            
        Returns:
            float: Total reward from the rollout
        """
        # Reset environment with specific setpoint schedule
        options = {
            'setpoints_Cb': setpoints_Cb,
            'setpoints_V': setpoints_V,
            'setpoint_durations': setpoint_durations
        }
        state, _ = env.reset(options=options)
        
        total_reward = 0
        done = False
        
        while not done:
            # The position represents normalized PID gains in [-1, 1]
            # Convert to actual PID gains based on environment bounds
            min_bounds = np.array([-5, 0, 0.02, 0, 0, 0.01])  # Lower bounds from environment
            max_bounds = np.array([25, 20, 10, 1, 2, 1])      # Upper bounds from environment
            actual_gains = min_bounds + (max_bounds - min_bounds) * (self.position + 1) / 2
                        
            # Now convert back to normalized [-1, 1] for the environment
            action = np.clip(self.position, -1, 1)
            
            # Take a step in the environment
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            # Update state
            state = next_state
            
            # Break if done or truncated
            if done or truncated:
                break
        
        return total_reward


class PSOCIRL:
    """
    Combined PSO+CIRL approach for CSTR control optimization.
    """
    def __init__(self, env, state_dim, action_dim, action_high, device="cuda", 
                 num_particles=20, max_iterations=50, save_dir="./results/pso_cirl"):
        """
        Initialize the PSO+CIRL trainer.
        
        Args:
            env: The CSTR environment instance
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            action_high: Maximum action value
            device: Device for torch tensors
            num_particles: Number of particles in the PSO swarm
            max_iterations: Maximum PSO iterations
            save_dir: Directory to save results
        """
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = action_high
        self.device = device
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
        
        # The search space is 6 PID gains 
        self.dim = 6
        self.bounds = [
            (0, 50),      # Kp_Cb - centered around 25
            (0, 2),       # Ki_Cb - centered around 0.73
            (0.01, 0.1),  # Kd_Cb - centered around 0.02
            (0, 1),       # Kp_V - centered around 0.35
            (0, 1),       # Ki_V - centered around 0.26
            (0, 1)        # Kd_V - centered around 0.27
        ]
        
        # Initialize particles
        self.particles = [PSOParticle(self.dim, self.bounds) for _ in range(num_particles)]
        
        # Initialize best position and score
        self.global_best_position = None
        self.global_best_score = -np.inf
        
        # For tracking progress
        self.best_scores_history = []
        self.mean_scores_history = []
    
    def initialize_with_pso(self, setpoints_Cb, setpoints_V, setpoint_durations, verbose=True):
        """
        Use PSO to find good initial policy parameters.
        
        Args:
            setpoints_Cb: List of Cb setpoints for training
            setpoints_V: List of V setpoints for training
            setpoint_durations: List of durations for each setpoint
            verbose: Whether to print progress
            
        Returns:
            numpy.ndarray: Best policy parameters found by PSO
        """
        print("Initializing CIRL policy with PSO...")
        start_time = time.time()
        
        # PSO main loop
        for iteration in tqdm(range(self.max_iterations), disable=not verbose):
            iteration_scores = []
            
            # Evaluate each particle
            for particle in self.particles:
                # Evaluate the particle
                score = particle.evaluate(self.env, setpoints_Cb, setpoints_V, setpoint_durations)
                iteration_scores.append(score)
                
                # Update particle's best known position
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                
                # Update global best if needed
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle.position.copy()
            
            # Update velocities and positions
            for particle in self.particles:
                # Adaptive parameters: reduce exploration over time
                w = 0.9 - (0.5 * iteration / self.max_iterations)
                c1 = 2.0 - (0.5 * iteration / self.max_iterations)
                c2 = 2.0 + (0.5 * iteration / self.max_iterations)
                
                particle.update_velocity(self.global_best_position, w, c1, c2)
                particle.update_position(self.bounds)
            
            # Track statistics
            mean_score = np.mean(iteration_scores)
            self.mean_scores_history.append(mean_score)
            self.best_scores_history.append(self.global_best_score)
            
            # Print progress
            if verbose and ((iteration + 1) % 5 == 0 or iteration == 0):
                print(f"Iteration {iteration+1}/{self.max_iterations}: "
                      f"Best Score = {self.global_best_score:.4f}, "
                      f"Mean Score = {mean_score:.4f}")
        
        # Save PSO results
        np.save(os.path.join(self.save_dir, "pso_best_policy.npy"), self.global_best_position)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        print(f"PSO initialization completed in {elapsed_time:.2f} seconds")
        print(f"Best policy found with score: {self.global_best_score:.4f}")
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.max_iterations + 1), self.best_scores_history, 'b-', label='Best Score')
        plt.plot(range(1, self.max_iterations + 1), self.mean_scores_history, 'r--', label='Mean Score')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('PSO Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "plots", "pso_learning_curve.png"))
        plt.close()
        
        return self.global_best_position
    
    def create_cirl_agent(self, pso_policy=None):
        """
        Create a CIRL agent, optionally initialized with PSO policy.
        
        Args:
            pso_policy: Initial policy parameters from PSO (optional)
            
        Returns:
            object: CIRL agent
        """
        # Create CIRL agent
        agent = create_agent("cirl", self.state_dim, self.action_dim, self.action_high, self.device)
        
        # If PSO policy is provided, initialize the agent with it
        if pso_policy is not None:
            print("Initializing CIRL agent with PSO policy parameters...")
            
            # CIRL network
            if hasattr(agent, 'actor'):
                # Save the initial parameters for comparison
                with torch.no_grad():
                    if hasattr(agent.actor, 'output'):
                        output_layer = agent.actor.output

                        policy_tensor = torch.FloatTensor(pso_policy).to(self.device)

                        output_layer.bias.data = policy_tensor

                        print("CIRL agent initialized with PSO policy")
                    else:
                        print("Warning: Could not initialize CIRL agent with PSO policy")
                        
                print("Warning: CIRL agent does not have an actor network")
                
        return agent
    
    def train_offline(self, agent, dataset_path, n_updates=20000, batch_size=256, update_interval=1000, verbose=True):
        """
        Train the CIRL agent offline using a pre-collected dataset.
        
        Args:
            agent: CIRL agent to train
            dataset_path: Path to the offline dataset
            n_updates: Number of training updates
            batch_size: Batch size for updates
            update_interval: Interval for saving models
            verbose: Whether to print progress
            
        Returns:
            dict: Training statistics
        """
        print("Starting offline training of CIRL agent...")
        
        # Create directories for saving results
        os.makedirs(os.path.join(self.save_dir, "models"), exist_ok=True)
        
        # Load dataset
        from Replay_Buffer import ReplayBuffer
        dataset = load_offline_dataset(dataset_path, verbose)
        
        # Create replay buffer and load dataset
        replay_buffer = ReplayBuffer(
            capacity=len(dataset['states']) + 1000,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        replay_buffer.load_from_dataset(dataset)
        
        # Train agent
        from Offline_training import offline_train
        stats = offline_train(
            agent=agent,
            replay_buffer=replay_buffer,
            n_updates=n_updates,
            batch_size=batch_size,
            update_interval=update_interval,
            save_dir=self.save_dir,
            verbose=verbose
        )
        
        # Save final model
        agent.save(os.path.join(self.save_dir, "models", "cirl_offline_final"))
        
        # Plot learning curves (assuming offline_train hasn't already done this)
        plt.figure(figsize=(12, 5))
        
        # Plot actor loss
        plt.subplot(1, 2, 1)
        if len(stats['actor_losses']) > 0:
            # Only use the steps that correspond to actor loss updates
            actor_steps = stats['steps'][:len(stats['actor_losses'])]
            plt.plot(actor_steps, stats['actor_losses'])
            plt.title("CIRL Offline Training - Actor Loss")
            plt.xlabel("Update Step")
            plt.ylabel("Loss")
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, "No actor loss data available", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title("CIRL Offline Training - No Actor Loss Data")
        
        # Plot critic loss
        plt.subplot(1, 2, 2)
        critic_steps = stats['steps'][:len(stats['critic_losses'])]
        plt.plot(critic_steps, stats['critic_losses'])
        plt.title("CIRL Offline Training - Critic Loss")
        plt.xlabel("Update Step")
        plt.ylabel("Loss")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "plots", "offline_learning_curves.png"))
        plt.close()
        
        print("Offline training completed")
        return stats
    
    def fine_tune_online(self, agent, n_episodes=100, batch_size=64, buffer_size=100000, 
                         init_exploration=0.1, final_exploration=0.05, verbose=True):
        """
        Fine-tune the CIRL agent with online interaction.
        
        Args:
            agent: CIRL agent to fine-tune
            n_episodes: Number of episodes for fine-tuning
            batch_size: Batch size for updates
            buffer_size: Size of replay buffer
            init_exploration: Initial exploration noise
            final_exploration: Final exploration noise
            verbose: Whether to print progress
            
        Returns:
            dict: Fine-tuning statistics
        """
        print("Starting online fine-tuning of CIRL agent...")
        
        # Create more challenging environment for fine-tuning
        fine_tune_env = CSTRRLEnv(
            simulation_steps=100,
            dt=1.0,
            uncertainty_level=0.01,    # Higher uncertainty
            noise_level=0.00,          # Higher noise
            actuator_delay_steps=0,    # Increased delays
            transport_delay_steps=0,
            enable_disturbances=False   # Enable disturbances
        )
        
        # Fine-tune the agent
        from online_fine_tuning import online_fine_tune
        fine_tune_stats = online_fine_tune(
            agent=agent,
            env=fine_tune_env,
            n_episodes=n_episodes,
            max_steps=100,
            batch_size=batch_size,
            buffer_size=buffer_size,
            init_exploration=init_exploration,
            final_exploration=final_exploration,
            save_dir=os.path.join(self.save_dir, "fine_tuned"),
            save_interval=10,
            eval_interval=5,
            eval_episodes=3,
            render=False,
            device=self.device,
            verbose=verbose
        )
        
        # Save final fine-tuned model
        agent.save(os.path.join(self.save_dir, "models", "cirl_fine_tuned_final"))
        
        print("Online fine-tuning completed")
        return fine_tune_stats
    
    def evaluate_and_compare(self, agent, static_pid_path, n_episodes=5, verbose=True):
        """
        Evaluate and compare the PSO+CIRL agent against the baseline static PID.
        
        Args:
            agent: Trained CIRL agent
            static_pid_path: Path to the optimized static PID gains
            n_episodes: Number of episodes for evaluation
            verbose: Whether to print progress
            
        Returns:
            dict: Evaluation results
        """
        print("Evaluating PSO+CIRL agent against baseline static PID...")
        
        # Load static PID gains
        static_pid_gains = np.load(static_pid_path)
        
        # Create evaluation environment
        eval_env = CSTRRLEnv(
            simulation_steps=100,
            dt=1.0,
            uncertainty_level=0.01,
            noise_level=0.00,
            actuator_delay_steps=0,
            transport_delay_steps=0,
            enable_disturbances=False
        )
        
        # Create controller evaluator
        evaluator = ControllerEvaluator(save_dir=os.path.join(self.save_dir, "evaluation"))
        
        # Add controllers
        evaluator.add_static_pid_controller("Static PID", static_pid_gains)
        evaluator.add_rl_controller("PSO+CIRL", agent, "cirl")
        
        # Define evaluation scenarios
        setpoints_schedule = [
            {"name": "Increasing", "setpoints": [0.40, 0.60, 0.80]},
            {"name": "Decreasing", "setpoints": [0.90, 0.70, 0.50]},
            {"name": "Peak", "setpoints": [0.55, 0.80, 0.55]},
            {"name": "Valley", "setpoints": [0.85, 0.60, 0.85]},
            {"name": "Step", "setpoints": [0.6, 0.6, 0.8, 0.8]}
        ]
        
        # Run evaluation
        results = evaluator.evaluate_setpoint_tracking(
            env=eval_env,
            setpoints_schedule=setpoints_schedule,
            n_episodes=n_episodes,
            max_steps=100,
            render=False,
            verbose=verbose
        )
        
        # Generate plots
        for scenario in [s["name"] for s in setpoints_schedule]:
            evaluator.plot_setpoint_tracking(scenario, save_plots=True, show_plots=False)
        
        evaluator.plot_comparative_metrics(save_plots=True, show_plots=False)
        evaluator.plot_controller_heatmap(metric="reward", save_plots=True, show_plots=False)
        evaluator.plot_controller_heatmap(metric="mse", save_plots=True, show_plots=False)
        
        # Generate statistical report
        report = evaluator.generate_statistical_report(baseline_controller="Static PID")
        
        print("Evaluation completed")
        return results


def run_complete_pso_cirl_training(static_pid_path, dataset_path, 
                                  save_dir="./results/pso_cirl",
                                  pso_iterations=30,
                                  pso_particles=20,
                                  offline_updates=20000,
                                  online_episodes=100,
                                  seed=42):
    """
    Run the complete PSO+CIRL training pipeline.
    
    Args:
        static_pid_path: Path to the optimized static PID gains
        dataset_path: Path to the offline dataset
        save_dir: Directory to save results
        pso_iterations: Number of PSO iterations
        pso_particles: Number of PSO particles
        offline_updates: Number of offline training updates
        online_episodes: Number of online fine-tuning episodes
        seed: Random seed
    """
    # Set random seed
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment for PSO initialization
    env = CSTRRLEnv(
        simulation_steps=100,
        dt=1.0,
        uncertainty_level=0.0,     # No uncertainty for PSO
        noise_level=0.0,           # No noise for PSO
        actuator_delay_steps=0,    # No delays for PSO
        transport_delay_steps=0,
        enable_disturbances=False  # No disturbances for PSO
    )
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high[0]
    
    # Could you justify why we need to define a setpoint for PSO training? 
    # Define setpoints for PSO training
    setpoints_Cb = [0.2, 0.4, 0.6, 0.8]
    setpoints_V = [100.0] * len(setpoints_Cb)
    setpoint_durations = [40] * len(setpoints_Cb)
    
    # Initialize PSO+CIRL trainer
    trainer = PSOCIRL(
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        action_high=action_high,
        device=device,
        num_particles=pso_particles,
        max_iterations=pso_iterations,
        save_dir=save_dir
    )
    
    # Step 1: Initialize with PSO
    print("\n=== Step 1: PSO Initialization ===")
    pso_policy = trainer.initialize_with_pso(
        setpoints_Cb=setpoints_Cb,
        setpoints_V=setpoints_V,
        setpoint_durations=setpoint_durations,
        verbose=True
    )
    
    # Step 2: Create CIRL agent initialized with PSO policy
    print("\n=== Step 2: Create CIRL Agent ===")
    cirl_agent = trainer.create_cirl_agent(pso_policy)
    
    # Step 3: Offline training
    print("\n=== Step 3: Offline Training ===")
    offline_stats = trainer.train_offline(
        agent=cirl_agent,
        dataset_path=dataset_path,
        n_updates=offline_updates,
        batch_size=256,
        update_interval=1000,
        verbose=True
    )
    
    # Step 4: Online fine-tuning
    print("\n=== Step 4: Online Fine-tuning ===")
    fine_tune_stats = trainer.fine_tune_online(
        agent=cirl_agent,
        n_episodes=online_episodes,
        batch_size=64,
        buffer_size=100000,
        init_exploration=0.3,
        final_exploration=0.1,
        verbose=True
    )
    
    # Step 5: Evaluation and comparison
    print("\n=== Step 5: Evaluation and Comparison ===")
    eval_results = trainer.evaluate_and_compare(
        agent=cirl_agent,
        static_pid_path=static_pid_path,
        n_episodes=5,
        verbose=True
    )
    
    print("\n=== Training Pipeline Completed ===")
    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PSO+CIRL Training for CSTR Control")
    parser.add_argument("--static_pid", type=str, required=True, help="Path to optimized static PID gains")
    parser.add_argument("--dataset", type=str, required=True, help="Path to offline dataset")
    parser.add_argument("--save_dir", type=str, default="./results/pso_cirl", help="Directory to save results")
    parser.add_argument("--pso_iter", type=int, default=30, help="Number of PSO iterations")
    parser.add_argument("--pso_particles", type=int, default=20, help="Number of PSO particles")
    parser.add_argument("--offline_updates", type=int, default=20000, help="Number of offline training updates")
    parser.add_argument("--online_episodes", type=int, default=100, help="Number of online fine-tuning episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Run complete PSO+CIRL training
    run_complete_pso_cirl_training(
        static_pid_path=args.static_pid,
        dataset_path=args.dataset,
        save_dir=args.save_dir,
        pso_iterations=args.pso_iter,
        pso_particles=args.pso_particles,
        offline_updates=args.offline_updates,
        online_episodes=args.online_episodes,
        seed=args.seed
    )