"""
Section 1: Visualize CIRL policy performance on all test schedules
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pickle

from CSTR_model_plus import CSTRRLEnv
from cirl_policy_network import CIRLPolicyNetwork

def plot_all_schedules(policy, env, save_dir=None, render=False):
    """
    Evaluate and plot policy performance on all test schedules.
    
    Args:
        policy: The trained CIRL policy
        env: CSTR environment
        save_dir: Directory to save plots (optional)
        render: Whether to render the environment during evaluation
        
    Returns:
        dict: Results for all schedules
    """
    policy.eval()  # Set to evaluation mode
    device = next(policy.parameters()).device
    
    # Set up test schedules
    test_schedules = [
        {"name": "Increasing", "setpoints": [0.12, 0.35, 0.48, 0.71, 0.84]},
        {"name": "Decreasing", "setpoints": [0.86, 0.73, 0.61, 0.48, 0.25]},
        {"name": "Peak", "setpoints": [0.25, 0.30, 0.65, 0.30, 0.25]},
        {"name": "Valley", "setpoints": [0.86, 0.71, 0.24, 0.63, 0.82]}
    ]
    
    # Create figure for all schedules
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle("CIRL Policy Performance on All Schedules", fontsize=16)
    
    # Create directory for saving plots if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Results to return
    results = {}
    
    # Run each schedule
    for idx, schedule in enumerate(test_schedules):
        schedule_name = schedule["name"]
        setpoints_Cb = schedule["setpoints"]
        print(f"Evaluating on {schedule_name} schedule: {setpoints_Cb}")
        
        # Configure environment
        setpoints_V = [100.0] * len(setpoints_Cb)
        setpoint_durations = [30] * len(setpoints_Cb)
        
        # Reset environment
        state, _ = env.reset(options={
            'setpoints_Cb': setpoints_Cb,
            'setpoints_V': setpoints_V,
            'setpoint_durations': setpoint_durations
        })
        
        # Initialize storage for trajectory
        trajectory = {
            'states': [],
            'actions': [],
            'pid_gains': [],
            'rewards': [],
            'setpoints_Cb': [],
            'setpoints_V': [],
            'Cb': [],
            'V': [],
            'T': [],
            'Tc': [],
            'Fin': []
        }
        
        # Run episode
        done = False
        total_reward = 0
        
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Get action from policy
            with torch.no_grad():
                action = policy(state_tensor).cpu().numpy()[0]
            
            # Step the environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Extract info
            pid_gains = ((action + 1) / 2) * (env.pid_upper - env.pid_lower) + env.pid_lower
            true_state = info.get("true_state", None)
            control_action = info.get("control_action", None)
            
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
            
            if control_action is not None:
                trajectory['Tc'].append(control_action[0])
                trajectory['Fin'].append(control_action[1])
            
            # Render if requested
            if render:
                env.render()
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            if done or truncated:
                break
        
        # Calculate MSEs
        mse_cb = np.mean((np.array(trajectory['Cb']) - np.array(trajectory['setpoints_Cb'])) ** 2)
        mse_v = np.mean((np.array(trajectory['V']) - np.array(trajectory['setpoints_V'])) ** 2)
        
        print(f"  Reward: {total_reward:.2f}, MSE Cb: {mse_cb:.6f}, MSE V: {mse_v:.6f}")
        
        # Store results
        results[schedule_name] = {
            'trajectory': trajectory,
            'total_reward': total_reward,
            'mse_cb': mse_cb,
            'mse_v': mse_v
        }
        
        # Plot results for this schedule
        # Row 1: Concentration B Tracking
        axes[idx, 0].plot(trajectory['Cb'], 'b-', label='Actual Cb')
        axes[idx, 0].plot(trajectory['setpoints_Cb'], 'r--', label='Setpoint Cb')
        axes[idx, 0].set_title(f"{schedule_name} - Cb Tracking")
        axes[idx, 0].set_xlabel('Step')
        axes[idx, 0].set_ylabel('Concentration')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True)
        
        # Row 2: Volume Tracking
        axes[idx, 1].plot(trajectory['V'], 'g-', label='Actual V')
        axes[idx, 1].plot(trajectory['setpoints_V'], 'r--', label='Setpoint V')
        axes[idx, 1].set_title(f"{schedule_name} - Volume Tracking")
        axes[idx, 1].set_xlabel('Step')
        axes[idx, 1].set_ylabel('Volume')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True)
        
        # Row 3: PID Gains
        pid_gains = np.array(trajectory['pid_gains'])
        axes[idx, 2].plot(pid_gains[:, 0], 'r-', label='Kp_Cb')
        axes[idx, 2].plot(pid_gains[:, 1], 'g-', label='Ki_Cb')
        axes[idx, 2].plot(pid_gains[:, 2], 'b-', label='Kd_Cb')
        axes[idx, 2].plot(pid_gains[:, 3], 'c-', label='Kp_V')
        axes[idx, 2].plot(pid_gains[:, 4], 'm-', label='Ki_V')
        axes[idx, 2].plot(pid_gains[:, 5], 'y-', label='Kd_V')
        axes[idx, 2].set_title(f"{schedule_name} - PID Gains")
        axes[idx, 2].set_xlabel('Step')
        axes[idx, 2].set_ylabel('Gain Value')
        axes[idx, 2].legend()
        axes[idx, 2].grid(True)
    
    # Add metrics to the plot as text
    metrics_text = "Performance Metrics:\n"
    for schedule_name, schedule_results in results.items():
        metrics_text += f"\n{schedule_name}:\n"
        metrics_text += f"  Reward: {schedule_results['total_reward']:.2f}\n"
        metrics_text += f"  MSE Cb: {schedule_results['mse_cb']:.6f}\n"
        metrics_text += f"  MSE V: {schedule_results['mse_v']:.6f}\n"
    
    # Make room for text
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the bottom margin
    
    # Save figure if directory is specified
    if save_dir:
        plt.savefig(os.path.join(save_dir, "all_schedules_performance.png"), dpi=300)
    
    # Show plot
    plt.show()
    
    # Also create a separate detailed plot for each schedule
    for schedule_name, schedule_results in results.items():
        trajectory = schedule_results['trajectory']
        
        # Create detailed figure
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle(f"Detailed Performance - {schedule_name} Schedule", fontsize=16)
        
        # Plot Cb tracking
        axes[0, 0].plot(trajectory['Cb'], 'b-', label='Actual Cb')
        axes[0, 0].plot(trajectory['setpoints_Cb'], 'r--', label='Setpoint Cb')
        axes[0, 0].set_title('Concentration B Tracking')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Concentration')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot Volume tracking
        axes[0, 1].plot(trajectory['V'], 'g-', label='Actual V')
        axes[0, 1].plot(trajectory['setpoints_V'], 'r--', label='Setpoint V')
        axes[0, 1].set_title('Volume Tracking')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Volume')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot temperature
        axes[1, 0].plot(trajectory['T'], 'm-', label='Temperature')
        axes[1, 0].set_title('Reactor Temperature')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Temperature (K)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot control actions if available
        if 'Tc' in trajectory and len(trajectory['Tc']) > 0:
            axes[1, 1].plot(trajectory['Tc'], 'r-', label='Cooling Temp (Tc)')
            axes[1, 1].plot(trajectory['Fin'], 'b-', label='Inlet Flow (Fin)')
            axes[1, 1].set_title('Control Actions')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # Plot PID gains
        pid_gains = np.array(trajectory['pid_gains'])
        
        # Split PID gains into two plots for better visibility
        axes[2, 0].plot(pid_gains[:, 0], 'r-', label='Kp_Cb')
        axes[2, 0].plot(pid_gains[:, 1], 'g-', label='Ki_Cb')
        axes[2, 0].plot(pid_gains[:, 2], 'b-', label='Kd_Cb')
        axes[2, 0].set_title('Cb PID Gains')
        axes[2, 0].set_xlabel('Step')
        axes[2, 0].set_ylabel('Gain Value')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        axes[2, 1].plot(pid_gains[:, 3], 'c-', label='Kp_V')
        axes[2, 1].plot(pid_gains[:, 4], 'm-', label='Ki_V')
        axes[2, 1].plot(pid_gains[:, 5], 'y-', label='Kd_V')
        axes[2, 1].set_title('Volume PID Gains')
        axes[2, 1].set_xlabel('Step')
        axes[2, 1].set_ylabel('Gain Value')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save detailed figure if directory is specified
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{schedule_name}_detailed_performance.png"), dpi=300)
        
        plt.show()
    
    return results

# Example usage
if __name__ == "__main__":
    # Create environment
    env = CSTRRLEnv(
        simulation_steps=150,
        dt=1.0,
        uncertainty_level=0.0,
        noise_level=0.0,
        actuator_delay_steps=0,
        transport_delay_steps=0,
        enable_disturbances=False
    )
    
    # Load trained policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = CIRLPolicyNetwork(state_dim=env.observation_space.shape[0], hidden_dims=[128, 128]).to(device)
    policy.load("./results/CIRL/models/cirl_policy_final.pt")
    
    # Run evaluation and plotting
    results = plot_all_schedules(policy, env, save_dir="./results/CIRL/test_plots")




"""
Section 2: Analyze and compare PID gains across different setpoint schedules
"""

def analyze_pid_gain_patterns(policy, env, save_dir=None):
    """
    Analyze how PID gains adapt to different setpoint schedules.
    
    Args:
        policy: The trained CIRL policy
        env: CSTR environment
        save_dir: Directory to save plots (optional)
        
    Returns:
        dict: Analysis results
    """
    policy.eval()  # Set to evaluation mode
    device = next(policy.parameters()).device
    
    # Set up test schedules
    test_schedules = [
        {"name": "Increasing", "setpoints": [0.12, 0.35, 0.48, 0.71, 0.84]},
        {"name": "Decreasing", "setpoints": [0.86, 0.73, 0.61, 0.48, 0.25]},
        {"name": "Peak", "setpoints": [0.25, 0.30, 0.65, 0.30, 0.25]},
        {"name": "Valley", "setpoints": [0.86, 0.71, 0.24, 0.63, 0.82]}
    ]
    
    # Parameters to store gain transitions at setpoint changes
    setpoint_transitions = {}
    setpoint_response = {}
    gain_statistics = {}
    trajectories = {}
    
    # Create directory for saving plots if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Run each schedule
    for schedule in test_schedules:
        schedule_name = schedule["name"]
        setpoints_Cb = schedule["setpoints"]
        print(f"Analyzing PID gains for {schedule_name} schedule: {setpoints_Cb}")
        
        # Configure environment
        setpoints_V = [100.0] * len(setpoints_Cb)
        setpoint_durations = [50] * len(setpoints_Cb)
        
        # Reset environment
        state, _ = env.reset(options={
            'setpoints_Cb': setpoints_Cb,
            'setpoints_V': setpoints_V,
            'setpoint_durations': setpoint_durations
        })
        
        # Storage for trajectory
        trajectory = {
            'pid_gains': [],
            'setpoints_Cb': [],
            'Cb': [],
            'error_Cb': [],
            'setpoint_change_indices': []
        }
        
        # Run episode
        done = False
        step = 0
        last_setpoint = None
        
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Get action from policy
            with torch.no_grad():
                action = policy(state_tensor).cpu().numpy()[0]
            
            # Step the environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Calculate PID gains from normalized action
            pid_gains = ((action + 1) / 2) * (env.pid_upper - env.pid_lower) + env.pid_lower
            
            # Get current values
            current_setpoint_Cb = next_state[9]  # Setpoint Cb
            true_state = info.get("true_state", None)
            
            if true_state is not None:
                current_Cb = true_state[1]  # Actual Cb
            else:
                current_Cb = next_state[0]  # Use measured state if true state unavailable
            
            # Calculate error
            error_Cb = current_setpoint_Cb - current_Cb
            
            # Check for setpoint change
            if last_setpoint is not None and current_setpoint_Cb != last_setpoint:
                trajectory['setpoint_change_indices'].append(step)
            
            last_setpoint = current_setpoint_Cb
            
            # Store data
            trajectory['pid_gains'].append(pid_gains)
            trajectory['setpoints_Cb'].append(current_setpoint_Cb)
            trajectory['Cb'].append(current_Cb)
            trajectory['error_Cb'].append(error_Cb)
            
            # Update state
            state = next_state
            step += 1
            
            if done or truncated:
                break
        
        # Convert lists to numpy arrays for easier analysis
        trajectory['pid_gains'] = np.array(trajectory['pid_gains'])
        trajectory['setpoints_Cb'] = np.array(trajectory['setpoints_Cb'])
        trajectory['Cb'] = np.array(trajectory['Cb'])
        trajectory['error_Cb'] = np.array(trajectory['error_Cb'])
        
        # Store the trajectory
        trajectories[schedule_name] = trajectory
        
        # Analyze gain patterns around setpoint changes
        transitions = []
        responses = []
        
        for idx in trajectory['setpoint_change_indices']:
            # Get gains just before and after setpoint change
            before_gains = trajectory['pid_gains'][max(0, idx-5):idx].mean(axis=0)
            after_gains = trajectory['pid_gains'][idx:min(len(trajectory['pid_gains']), idx+5)].mean(axis=0)
            
            # Store transition
            transitions.append({
                'before': before_gains,
                'after': after_gains,
                'change': after_gains - before_gains
            })
            
            # Analyze response after setpoint change
            response_window = min(20, len(trajectory['Cb']) - idx)
            if response_window > 0:
                error_response = trajectory['error_Cb'][idx:idx+response_window]
                responses.append({
                    'error_profile': error_response,
                    'settling_time': np.argmin(np.abs(error_response)) if len(error_response) > 0 else None,
                    'max_error': np.max(np.abs(error_response)) if len(error_response) > 0 else None
                })
        
        setpoint_transitions[schedule_name] = transitions
        setpoint_response[schedule_name] = responses
        
        # Calculate gain statistics for this schedule
        gain_statistics[schedule_name] = {
            'mean': trajectory['pid_gains'].mean(axis=0),
            'std': trajectory['pid_gains'].std(axis=0),
            'min': trajectory['pid_gains'].min(axis=0),
            'max': trajectory['pid_gains'].max(axis=0),
            'range': trajectory['pid_gains'].max(axis=0) - trajectory['pid_gains'].min(axis=0)
        }
    
    # Create comparative plots
    
    # 1. Overall PID gain statistics by schedule
    gain_names = ['Kp_Cb', 'Ki_Cb', 'Kd_Cb', 'Kp_V', 'Ki_V', 'Kd_V']
    
    # Plot mean gains for each schedule
    plt.figure(figsize=(15, 10))
    
    for i, gain_name in enumerate(gain_names):
        plt.subplot(2, 3, i+1)
        
        # Collect data for this gain across schedules
        schedules = list(gain_statistics.keys())
        mean_values = [gain_statistics[s]['mean'][i] for s in schedules]
        std_values = [gain_statistics[s]['std'][i] for s in schedules]
        
        # Create bar chart with error bars
        plt.bar(schedules, mean_values, yerr=std_values, capsize=5)
        plt.title(f'Mean {gain_name} by Schedule')
        plt.ylabel('Gain Value')
        plt.grid(axis='y')
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "pid_gain_statistics.png"), dpi=300)
    plt.show()
    
    # 2. Plot gain adaptations over time for each schedule on one figure
    plt.figure(figsize=(15, 12))
    
    for i, gain_name in enumerate(gain_names):
        plt.subplot(3, 2, i+1)
        
        for schedule_name, trajectory in trajectories.items():
            # Plot this gain over time
            plt.plot(trajectory['pid_gains'][:, i], label=schedule_name)
            
            # Mark setpoint changes
            for idx in trajectory['setpoint_change_indices']:
                plt.axvline(x=idx, color='gray', linestyle='--', alpha=0.5)
        
        plt.title(f'{gain_name} Adaptation')
        plt.xlabel('Step')
        plt.ylabel('Gain Value')
        plt.legend()
        plt.grid(True)
    
    # Add a subplot for Cb tracking (average across schedules)
    plt.subplot(3, 2, 6)
    
    for schedule_name, trajectory in trajectories.items():
        plt.plot(trajectory['Cb'], label=f"{schedule_name} Cb")
        plt.plot(trajectory['setpoints_Cb'], '--', label=f"{schedule_name} Setpoint", alpha=0.5)
    
    plt.title('Concentration B Tracking')
    plt.xlabel('Step')
    plt.ylabel('Concentration')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "pid_gain_adaptations.png"), dpi=300)
    plt.show()
    
    # 3. Compare gain changes at setpoint transitions
    # Extract transitions for the Cb controller gains
    cb_gains_indices = [0, 1, 2]  # Kp_Cb, Ki_Cb, Kd_Cb
    
    plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(cb_gains_indices):
        plt.subplot(2, 2, i+1)
        
        # For each schedule, plot the changes in this gain at setpoint transitions
        x_positions = []
        y_values = []
        labels = []
        
        position = 0
        for schedule_name, transitions in setpoint_transitions.items():
            for j, transition in enumerate(transitions):
                x_positions.append(position)
                y_values.append(transition['change'][idx])
                labels.append(f"{schedule_name}\nTransition {j+1}")
                position += 1
        
        plt.bar(x_positions, y_values)
        plt.xticks(x_positions, labels, rotation=45, ha='right')
        plt.title(f'Change in {gain_names[idx]} at Setpoint Transitions')
        plt.ylabel('Gain Change')
        plt.grid(axis='y')
    
    # Add a subplot for error response
    plt.subplot(2, 2, 4)
    
    # For each schedule, plot average error profile after setpoint change
    for schedule_name, responses in setpoint_response.items():
        # Collect all error profiles
        error_profiles = []
        for response in responses:
            if response['error_profile'] is not None and len(response['error_profile']) > 0:
                error_profiles.append(response['error_profile'])
        
        # If we have profiles, plot the average
        if error_profiles:
            # Find the maximum length
            max_len = max(len(profile) for profile in error_profiles)
            
            # Pad shorter profiles with NaN
            padded_profiles = []
            for profile in error_profiles:
                padded = np.full(max_len, np.nan)
                padded[:len(profile)] = profile
                padded_profiles.append(padded)
            
            # Calculate mean error profile, ignoring NaN values
            mean_profile = np.nanmean(padded_profiles, axis=0)
            
            # Plot mean error profile
            plt.plot(mean_profile, label=schedule_name)
    
    plt.title('Average Error Response After Setpoint Change')
    plt.xlabel('Steps After Change')
    plt.ylabel('Error (Setpoint - Actual)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "pid_gain_transitions.png"), dpi=300)
    plt.show()
    
    # Return analysis results
    results = {
        'trajectories': trajectories,
        'gain_statistics': gain_statistics,
        'setpoint_transitions': setpoint_transitions,
        'setpoint_response': setpoint_response
    }
    
    return results

# Example usage
if __name__ == "__main__":
    # Create environment
    env = CSTRRLEnv(
        simulation_steps=150,
        dt=1.0,
        uncertainty_level=0.0,
        noise_level=0.0,
        actuator_delay_steps=0,
        transport_delay_steps=0,
        enable_disturbances=False
    )
    
    # Load trained policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = CIRLPolicyNetwork(state_dim=env.observation_space.shape[0], hidden_dims=[128, 128]).to(device)
    policy.load("./results/CIRL/models/cirl_policy_final.pt")
    
    # Run PID gain analysis
    analysis_results = analyze_pid_gain_patterns(policy, env, save_dir="./results/CIRL/test_plots")