"""
static_pid_gains.py - Optimize static PID gains for CSTR control system

This module uses differential evolution to find optimal static PID gains
for the CSTR system. The optimization is performed with:
- No noise
- No disturbance
- No delays

The optimized gains can serve as a baseline for comparing RL approaches.
"""

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import os
import time
import pickle

class StaticPIDOptimizer:
    """
    Optimizer for finding static PID gains for the CSTR control system.
    """
    
    def __init__(self, env, simulation_steps=150, steps_per_setpoint=20):
        """
        Initialize the Static PID Optimizer.
        
        Args:
            env: CSTR environment instance
            simulation_steps (int): Total simulation steps
            steps_per_setpoint (int): Number of steps for each setpoint
        """
        self.env = env
        self.simulation_steps = simulation_steps
        self.steps_per_setpoint = steps_per_setpoint
        
        # Define default setpoint schedule used for optimization
        self.default_setpoint_schedule = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.8]
        
        # Define bounds for the PID gains (normalized to [-1, 1])
        self.bounds = [(-1, 1)] * 6
        
        # Directory to save results
        self.save_dir = "./results"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Store the setpoint schedule for the objective function
        self.current_setpoint_schedule = self.default_setpoint_schedule
    
    def objective_function(self, gains):
        """
        Global objective function for optimization (needed for multiprocessing).
        
        Args:
            gains (numpy.ndarray): Normalized PID gains in range [-1, 1]
            
        Returns:
            float: Negative mean reward (for minimization)
        """
        return self.evaluate_pid_gains(
            gains, 
            n_repetitions=3, 
            setpoint_schedule=self.current_setpoint_schedule
        )
    
    def evaluate_pid_gains(self, gains, n_repetitions=3, setpoint_schedule=None):
        """
        Evaluate PID gains by running the environment and calculating total reward.
        
        Args:
            gains (numpy.ndarray): Normalized PID gains in range [-1, 1]
            n_repetitions (int): Number of repetitions to average over
            setpoint_schedule (list): Setpoint schedule to use (optional)
            
        Returns:
            float: Negative mean reward (for minimization)
        """
        # Use default schedule if none provided
        if setpoint_schedule is None:
            setpoint_schedule = self.default_setpoint_schedule
        
        # Calculate total steps needed
        total_steps = len(setpoint_schedule) * self.steps_per_setpoint
        
        # Store rewards from each repetition
        repetition_rewards = []
        
        # Run n_repetitions times and average the results
        for rep in range(n_repetitions):
            # Configure setpoints for this run
            setpoints_Cb = setpoint_schedule
            setpoints_V = [100.0] * len(setpoint_schedule)
            setpoint_durations = [self.steps_per_setpoint] * len(setpoint_schedule)
            
            # Reset environment with this configuration
            state, _ = self.env.reset(seed=rep, options={
                'setpoints_Cb': setpoints_Cb,
                'setpoints_V': setpoints_V,
                'setpoint_durations': setpoint_durations
            })
            
            # Initialize variables for this run
            done = False
            total_reward = 0
            step = 0
            
            # Run the episode with fixed PID gains
            while not done and step < total_steps:
                # Use the fixed gains as action
                action = gains
                
                # Take a step in the environment
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # Accumulate reward
                total_reward += reward
                
                # Move to next state
                state = next_state
                step += 1
                
                if done or truncated:
                    break
            
            # Store total reward for this repetition
            repetition_rewards.append(total_reward)
        
        # Calculate mean reward across repetitions
        mean_reward = np.mean(repetition_rewards)
        
        # Return negative reward (since we're minimizing)
        return -mean_reward
    
    def optimize_pid_gains(self, maxiter=100, popsize=15, setpoint_schedule=None, 
                           verbose=True, workers=1):
        """
        Optimize PID gains using differential evolution.
        
        Args:
            maxiter (int): Maximum number of iterations for optimization
            popsize (int): Population size for differential evolution
            setpoint_schedule (list): Setpoint schedule to use (optional)
            verbose (bool): Whether to print progress
            workers (int): Number of parallel workers (1 = no parallelization)
            
        Returns:
            tuple: Optimal normalized gains and optimization result
        """
        # Use default schedule if none provided
        if setpoint_schedule is None:
            setpoint_schedule = self.default_setpoint_schedule
        
        # Store the setpoint schedule for the objective function
        self.current_setpoint_schedule = setpoint_schedule
        
        print(f"Starting optimization with {maxiter} iterations and population size {popsize}")
        # print(f"Setpoint schedule: {setpoint_schedule}")
        
        start_time = time.time()
        
        # Run differential evolution
        result = differential_evolution(
            self.objective_function,
            bounds=self.bounds,  # keep bounds in original values
            maxiter=maxiter,
            popsize=popsize,
            disp=verbose,
            tol=0.01,
            atol=0.01,
            updating='deferred',
            workers=workers  # Set to 1 to avoid multiprocessing issues
        )
        
        end_time = time.time()
        
        # Print results
        if verbose:
            print(f"Optimization completed in {end_time - start_time:.2f} seconds")
            print(f"Optimal normalized gains: {result.x}")
            print(f"Final function value: {result.fun}")
            
            # Convert normalized gains to actual PID values
            actual_gains = self.normalized_to_actual_gains(result.x)
            print("Optimal PID gains:")
            print(f"  Kp_Cb: {actual_gains[0]:.4f}")
            print(f"  Ki_Cb: {actual_gains[1]:.4f}")
            print(f"  Kd_Cb: {actual_gains[2]:.4f}")
            print(f"  Kp_V:  {actual_gains[3]:.4f}")
            print(f"  Ki_V:  {actual_gains[4]:.4f}")
            print(f"  Kd_V:  {actual_gains[5]:.4f}")
        
        return result.x, result, actual_gains
    
    def normalized_to_actual_gains(self, normalized_gains):
        """
        Convert normalized gains in [-1, 1] to actual PID gains.
        
        Args:
            normalized_gains (numpy.ndarray): Normalized gains
            
        Returns:
            numpy.ndarray: Actual PID gains
        """
        return ((normalized_gains + 1) / 2) * (self.env.pid_upper - self.env.pid_lower) + self.env.pid_lower
    
    def actual_to_normalized_gains(self, actual_gains):
        """
        Convert actual PID gains to normalized gains in [-1, 1].
        
        Args:
            actual_gains (numpy.ndarray): Actual PID gains
            
        Returns:
            numpy.ndarray: Normalized gains
        """
        return 2 * (actual_gains - self.env.pid_lower) / (self.env.pid_upper - self.env.pid_lower) - 1
    
    def save_gains(self, gains, filename="optimal_pid_gains.npy"):
        """
        Save optimal gains to a file.
        
        Args:
            gains (numpy.ndarray): Optimal normalized gains
            filename (str): Filename to save to
        """
        filepath = os.path.join(self.save_dir, filename)
        np.save(filepath, gains)
        print(f"Optimal gains saved to {filepath}")
    
    def load_gains(self, filename="optimal_pid_gains.npy"):
        """
        Load optimal gains from a file.
        
        Args:
            filename (str): Filename to load from
            
        Returns:
            numpy.ndarray: Loaded normalized gains
        """
        filepath = os.path.join(self.save_dir, filename)
        gains = np.load(filepath)
        print(f"Loaded optimal gains from {filepath}")
        return gains
    
    def evaluate_and_plot(self, gains, setpoint_schedules, repetitions=3, 
                          save_plots=True, show_plots=False):
        """
        Evaluate gains on multiple setpoint schedules and plot results.
        
        Args:
            gains (numpy.ndarray): Normalized PID gains
            setpoint_schedules (list): List of setpoint schedules to evaluate
            repetitions (int): Number of repetitions for each schedule
            save_plots (bool): Whether to save plots to disk
            show_plots (bool): Whether to display plots
            
        Returns:
            dict: Evaluation results
        """
        # Create directory for plots
        plots_dir = os.path.join(self.save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Convert normalized gains to actual PID values for display
        actual_gains = self.normalized_to_actual_gains(gains)
        gain_str = (f"Kp_Cb={actual_gains[0]:.2f}, Ki_Cb={actual_gains[1]:.2f}, "
                   f"Kd_Cb={actual_gains[2]:.2f}, Kp_V={actual_gains[3]:.2f}, "
                   f"Ki_V={actual_gains[4]:.2f}, Kd_V={actual_gains[5]:.2f}")
        
        # Results dictionary
        results = {}
        
        # Evaluate on each setpoint schedule
        for i, schedule in enumerate(setpoint_schedules):
            schedule_name = f"schedule_{i+1}"
            
            # Store performance metrics for this schedule
            schedule_results = {
                'setpoints': schedule,
                'mean_reward': 0,
                'mse_cb': 0,
                'mse_v': 0,
                'histories': []
            }
            
            # Run multiple repetitions and collect histories
            for rep in range(repetitions):
                # Configure environment with this schedule
                setpoints_Cb = schedule
                setpoints_V = [100.0] * len(schedule)
                setpoint_durations = [self.steps_per_setpoint] * len(schedule)
                
                # Reset environment
                state, _ = self.env.reset(seed=rep, options={
                    'setpoints_Cb': setpoints_Cb,
                    'setpoints_V': setpoints_V,
                    'setpoint_durations': setpoint_durations
                })
                
                # Initialize history for this run
                history = {
                    'time': [],
                    'Cb': [],
                    'V': [],
                    'reward': [],
                    'setpoint_Cb': [],
                    'setpoint_V': []
                }
                
                # Run episode
                done = False
                total_reward = 0
                step = 0
                
                while not done:
                    # Use fixed gains
                    action = gains
                    
                    # Take step
                    next_state, reward, done, truncated, info = self.env.step(action)
                    
                    # Extract current values from state
                    current_Cb = next_state[0]  # Current Cb is index 0
                    current_V = next_state[2]   # Current V is index 2
                    current_setpoint_Cb = next_state[9]  # Setpoint Cb is index 9
                    current_setpoint_V = next_state[10]  # Setpoint V is index 10
                    
                    # Store data
                    history['time'].append(step * self.env.dt)
                    history['Cb'].append(current_Cb)
                    history['V'].append(current_V)
                    history['setpoint_Cb'].append(current_setpoint_Cb)
                    history['setpoint_V'].append(current_setpoint_V)
                    history['reward'].append(reward)
                    
                    # Update
                    total_reward += reward
                    state = next_state
                    step += 1
                    
                    if done or truncated:
                        break
                
                # Calculate metrics for this repetition
                cb_errors = np.array(history['Cb']) - np.array(history['setpoint_Cb'])
                v_errors = np.array(history['V']) - np.array(history['setpoint_V'])
                
                mse_cb = np.mean(cb_errors ** 2)
                mse_v = np.mean(v_errors ** 2)
                
                # Update schedule results
                schedule_results['histories'].append(history)
                schedule_results['mean_reward'] += total_reward / repetitions
                schedule_results['mse_cb'] += mse_cb / repetitions
                schedule_results['mse_v'] += mse_v / repetitions
            
            # Add results for this schedule
            results[schedule_name] = schedule_results
            
            # Plot results for this schedule
            if save_plots or show_plots:
                self._plot_schedule_results(
                    schedule_results, 
                    schedule_name, 
                    gain_str,
                    save_path=os.path.join(plots_dir, f"{schedule_name}.png") if save_plots else None,
                    show=show_plots
                )
        
        # Print summary of results
        print("\nEvaluation Results Summary:")
        for schedule_name, res in results.items():
            print(f"  {schedule_name} (setpoints {res['setpoints']}):")
            print(f"    Mean Reward: {res['mean_reward']:.4f}")
            print(f"    MSE Cb: {res['mse_cb']:.6f}")
            print(f"    MSE V: {res['mse_v']:.6f}")
        
        return results
    
    def _plot_schedule_results(self, schedule_results, schedule_name, gain_str, 
                               save_path=None, show=False):
        """
        Plot results for a single setpoint schedule.
        
        Args:
            schedule_results (dict): Results for this schedule
            schedule_name (str): Name of the schedule
            gain_str (str): String describing the PID gains
            save_path (str): Path to save the plot (optional)
            show (bool): Whether to display the plot
        """
        # Calculate average history across repetitions
        histories = schedule_results['histories']
        n_histories = len(histories)
        
        # Find minimum length among histories
        min_length = min(len(h['time']) for h in histories)
        
        # Initialize average history
        avg_history = {
            'time': histories[0]['time'][:min_length],
            'Cb': np.zeros(min_length),
            'V': np.zeros(min_length),
            'setpoint_Cb': histories[0]['setpoint_Cb'][:min_length],
            'setpoint_V': histories[0]['setpoint_V'][:min_length],
            'reward': np.zeros(min_length)
        }
        
        # Calculate average values
        for h in histories:
            avg_history['Cb'] += np.array(h['Cb'][:min_length]) / n_histories
            avg_history['V'] += np.array(h['V'][:min_length]) / n_histories
            avg_history['reward'] += np.array(h['reward'][:min_length]) / n_histories
        
        # Create the plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot Cb tracking
        axes[0].plot(avg_history['time'], avg_history['Cb'], 'b-', label='Measured Cb')
        axes[0].plot(avg_history['time'], avg_history['setpoint_Cb'], 'r--', label='Setpoint Cb')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Concentration of B')
        axes[0].set_title(f'{schedule_name} - Cb Tracking')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot Volume tracking
        axes[1].plot(avg_history['time'], avg_history['V'], 'g-', label='Measured Volume')
        axes[1].plot(avg_history['time'], avg_history['setpoint_V'], 'r--', label='Setpoint Volume')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Volume')
        axes[1].set_title(f'{schedule_name} - Volume Tracking')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot reward
        axes[2].plot(avg_history['time'], avg_history['reward'], 'm-', label='Reward')
        axes[2].plot(avg_history['time'], np.cumsum(avg_history['reward']), 'c-', label='Cumulative Reward')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Reward')
        axes[2].set_title(f'{schedule_name} - Reward')
        axes[2].legend()
        axes[2].grid(True)
        
        # Add PID gains as suptitle
        fig.suptitle(f'Static PID Performance - {schedule_name}\n{gain_str}')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)


def run_optimization(env, maxiter=50, popsize=15):
    """
    Convenience function to run PID optimization.
    
    Args:
        env: CSTR environment instance
        maxiter (int): Maximum iterations for optimization
        popsize (int): Population size
        
    Returns:
        numpy.ndarray: Optimal normalized gains
    """
    # Create optimizer
    optimizer = StaticPIDOptimizer(env)
    
    # Run optimization
    optimal_gains, _, actual_gains = optimizer.optimize_pid_gains(
        maxiter=maxiter, 
        popsize=popsize,
        verbose=True,
        workers=1  # Set to 1 to avoid multiprocessing issues
    )
    
    # Save gains
    optimizer.save_gains(actual_gains)
    
    # Evaluate on multiple setpoint schedules
    test_schedules = [
        [0.15, 0.35, 0.55, 0.65, 0.75],  # Increasing steps
        [0.75, 0.65, 0.55, 0.35, 0.15],  # Decreasing steps
        [0.45, 0.65, 0.75, 0.65, 0.45],  # Peak
        [0.75, 0.65, 0.45, 0.65, 0.75]   # Valley
    ]
    
    optimizer.evaluate_and_plot(
        optimal_gains,
        test_schedules,
        repetitions=3,
        save_plots=True,
        show_plots=False
    )
    
    return optimal_gains

# -------------------------------------------------------------
# Example usage
# -------------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    from CSTR_model_plus import CSTRRLEnv
    
    # Create environment with minimal noise/disturbance/delay for optimization
    env = CSTRRLEnv(
        simulation_steps=150,
        dt=1.0,
        uncertainty_level=0.0,     # No uncertainty
        noise_level=0.0,           # No measurement noise
        actuator_delay_steps=0,    # No actuator delay
        transport_delay_steps=0,   # No transport delay
        enable_disturbances=False  # No disturbances
    )
    
    # Run optimization
    optimal_gains = run_optimization(env, maxiter=20, popsize=15)
    
    print("Optimization complete.")