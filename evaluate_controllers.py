"""
evaluate_controllers.py - Comprehensive evaluation of CSTR controllers

This module provides functionality for evaluating and comparing different control
approaches for the CSTR system, including:
1. Static PID controllers
2. RL-based controllers (TD3, SAC, CIRL)
3. PSO-optimized controllers

It includes:
- Standardized evaluation procedures
- Visualization of setpoint tracking performance
- Statistical comparison of controllers
- Export of results for reporting
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import argparse
from tqdm import tqdm
import pandas as pd
from scipy import stats
import torch
from CSTR_model_plus import CSTRRLEnv
from static_pid_gain import StaticPIDOptimizer
from RL_algorithms import create_agent
from Example_Training_with_CIRLPSO import CIRL_PSO, PSOParticle


class ControllerEvaluator:
    """
    Class for evaluating and comparing different controllers for the CSTR system.
    """
    
    def __init__(self, save_dir="./results/evaluation"):
        """
        Initialize the controller evaluator.
        
        Args:
            save_dir (str): Directory to save evaluation results
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Dictionary to store controllers
        self.controllers = {}
        
        # Dictionary to store evaluation results
        self.results = {}
    
    def add_static_pid_controller(self, name, gains, env=None):
        """
        Add a static PID controller to the evaluation.
        
        Args:
            name (str): Name of the controller
            gains (numpy.ndarray): Normalized PID gains in range [-1, 1]
            env (object): Optional environment instance
        """
        self.controllers[name] = {
            "type": "static_pid",
            "gains": gains,
            "env": env
        }
    
    def add_rl_controller(self, name, agent, agent_type):
        """
        Add an RL controller to the evaluation.
        
        Args:
            name (str): Name of the controller
            agent: RL agent instance
            agent_type (str): Type of agent ("td3", "sac", or "cirl")
        """
        self.controllers[name] = {
            "type": "rl",
            "agent": agent,
            "agent_type": agent_type
        }
    
    def add_pso_controller(self, name, policy_params):
        """
        Add a PSO-optimized controller to the evaluation.
        
        Args:
            name (str): Name of the controller
            policy_params (numpy.ndarray): Policy parameters optimized by PSO
        """
        self.controllers[name] = {
            "type": "pso",
            "policy_params": policy_params
        }
    
    def evaluate_setpoint_tracking(self, env, setpoints_schedule, n_episodes=3, 
                                   max_steps=200, render=False, verbose=True):
        """
        Evaluate controllers on setpoint tracking scenarios.
        
        Args:
            env: Environment for evaluation
            setpoints_schedule (list): List of dictionaries with setpoint scenarios
            n_episodes (int): Number of episodes per scenario
            max_steps (int): Maximum steps per episode
            render (bool): Whether to render the environment
            verbose (bool): Whether to print progress
            
        Returns:
            dict: Evaluation results
        """
        results = {}
        
        # Evaluate each controller
        for controller_name, controller in self.controllers.items():
            if verbose:
                print(f"\nEvaluating controller: {controller_name}")
            
            controller_results = {}
            
            # Evaluate on each setpoint schedule
            for scenario in setpoints_schedule:
                scenario_name = scenario["name"]
                setpoints_Cb = scenario["setpoints"]
                
                if verbose:
                    print(f"  Scenario: {scenario_name} - Setpoints: {setpoints_Cb}")
                
                scenario_results = {
                    "rewards": [],
                    "trajectories": [],
                    "errors": [],
                    "mse": []
                }
                
                # Run multiple episodes
                for episode in range(n_episodes):
                    # Configure environment for this scenario
                    setpoints_V = [100.0] * len(setpoints_Cb)  # Constant volume setpoint
                    setpoint_durations = [max_steps // len(setpoints_Cb)] * len(setpoints_Cb)
                    
                    # Reset environment
                    state, _ = env.reset(options={
                        'setpoints_Cb': setpoints_Cb,
                        'setpoints_V': setpoints_V,
                        'setpoint_durations': setpoint_durations
                    })
                    
                    # Prepare to store trajectory data
                    trajectory = {
                        "time": [],
                        "Cb": [],
                        "V": [],
                        "T": [],
                        "setpoint_Cb": [],
                        "setpoint_V": [],
                        "Tc": [],
                        "Fin": [],
                        "reward": []
                    }
                    
                    total_reward = 0
                    step = 0
                    done = False
                    
                    # Run episode with this controller
                    while not done and step < max_steps:
                        # Get action based on controller type
                        if controller["type"] == "static_pid":
                            action = controller["gains"]
                        
                        elif controller["type"] == "rl":
                            agent = controller["agent"]
                            agent_type = controller["agent_type"]
                            
                            if agent_type == "sac":
                                action = agent.select_action(state, evaluate=True)
                            else:  # td3 or cirl
                                action = agent.select_action(state, noise=0.0)
                        
                        elif controller["type"] == "pso":
                            # For PSO, the policy parameters are directly used as the action
                            action = controller["policy_params"]
                        
                        # Take step in environment
                        next_state, reward, done, truncated, info = env.step(action)
                        
                        # Update trajectory data
                        trajectory["time"].append(step * env.dt)
                        trajectory["Cb"].append(info["true_state"][1])  # Cb is at index 1
                        trajectory["V"].append(info["true_state"][4])   # V is at index 4
                        trajectory["T"].append(info["true_state"][3])   # T is at index 3
                        trajectory["setpoint_Cb"].append(info["setpoint_Cb"])
                        trajectory["setpoint_V"].append(info["setpoint_V"])
                        trajectory["Tc"].append(info["control_action"][0])
                        trajectory["Fin"].append(info["control_action"][1])
                        trajectory["reward"].append(reward)
                        
                        # Update for next step
                        state = next_state
                        total_reward += reward
                        step += 1
                        
                        if done or truncated:
                            break
                        
                        if render and episode == 0:  # Only render first episode
                            env.render()
                    
                    # Calculate MSE for Cb
                    cb_error = np.array(trajectory["Cb"]) - np.array(trajectory["setpoint_Cb"])
                    cb_mse = np.mean(cb_error ** 2)
                    
                    # Store results for this episode
                    scenario_results["rewards"].append(total_reward)
                    scenario_results["trajectories"].append(trajectory)
                    scenario_results["errors"].append(cb_error)
                    scenario_results["mse"].append(cb_mse)
                
                # Calculate statistics for this scenario
                mean_reward = np.mean(scenario_results["rewards"])
                std_reward = np.std(scenario_results["rewards"])
                mean_mse = np.mean(scenario_results["mse"])
                std_mse = np.std(scenario_results["mse"])
                
                if verbose:
                    print(f"    Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
                    print(f"    Mean MSE: {mean_mse:.6f} ± {std_mse:.6f}")
                
                # Store results for this scenario
                controller_results[scenario_name] = scenario_results
            
            # Store results for this controller
            results[controller_name] = controller_results
        
        # Store all results
        self.results = results
        
        # Save results
        results_path = os.path.join(self.save_dir, "evaluation_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        if verbose:
            print(f"\nEvaluation complete. Results saved to {results_path}")
        
        return results
    
    def plot_setpoint_tracking(self, scenario_name, controllers_to_plot=None, 
                              save_plots=True, show_plots=True):
        """
        Plot setpoint tracking performance for a specific scenario.
        
        Args:
            scenario_name (str): Name of the scenario to plot
            controllers_to_plot (list): List of controller names to plot (if None, plot all)
            save_plots (bool): Whether to save plots to disk
            show_plots (bool): Whether to display plots
        """
        if not self.results:
            print("No evaluation results available. Run evaluate_setpoint_tracking first.")
            return
        
        if controllers_to_plot is None:
            controllers_to_plot = list(self.results.keys())
        
        # Create directory for plots
        plots_dir = os.path.join(self.save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Get first trajectory for each controller (assuming all episodes have same setpoints)
        trajectories = {}
        for controller_name in controllers_to_plot:
            if controller_name in self.results and scenario_name in self.results[controller_name]:
                trajectories[controller_name] = self.results[controller_name][scenario_name]["trajectories"][0]
        
        if not trajectories:
            print(f"No data available for scenario: {scenario_name}")
            return
        
        # Create plots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot Cb tracking
        ax = axes[0]
        for controller_name, trajectory in trajectories.items():
            ax.plot(trajectory["time"], trajectory["Cb"], '-', linewidth=2, label=f"{controller_name}")
        
        # Plot setpoint (using the first controller's trajectory)
        first_controller = list(trajectories.keys())[0]
        ax.plot(trajectories[first_controller]["time"], trajectories[first_controller]["setpoint_Cb"], 
                'k--', linewidth=2, label="Setpoint")
        
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Concentration B (mol/m³)")
        ax.set_title(f"Concentration B Tracking - {scenario_name}")
        ax.legend()
        ax.grid(True)
        
        # Plot control actions - Cooling temperature
        ax = axes[1]
        for controller_name, trajectory in trajectories.items():
            ax.plot(trajectory["time"], trajectory["Tc"], '-', linewidth=2, label=f"{controller_name}")
        
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Cooling Temperature (K)")
        ax.set_title(f"Control Action (Cooling Temperature) - {scenario_name}")
        ax.legend()
        ax.grid(True)
        
        # Plot control actions - Inlet flow
        ax = axes[2]
        for controller_name, trajectory in trajectories.items():
            ax.plot(trajectory["time"], trajectory["Fin"], '-', linewidth=2, label=f"{controller_name}")
        
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Inlet Flow Rate (m³/min)")
        ax.set_title(f"Control Action (Inlet Flow Rate) - {scenario_name}")
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        if save_plots:
            plot_path = os.path.join(plots_dir, f"setpoint_tracking_{scenario_name}.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        
        # Show plot
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_comparative_metrics(self, save_plots=True, show_plots=True):
        """
        Plot comparative metrics for all controllers across all scenarios.
        
        Args:
            save_plots (bool): Whether to save plots to disk
            show_plots (bool): Whether to display plots
        """
        if not self.results:
            print("No evaluation results available. Run evaluate_setpoint_tracking first.")
            return
        
        # Create directory for plots
        plots_dir = os.path.join(self.save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Get all scenario names
        scenario_names = set()
        for controller_results in self.results.values():
            scenario_names.update(controller_results.keys())
        scenario_names = sorted(list(scenario_names))
        
        # Get all controller names
        controller_names = sorted(list(self.results.keys()))
        
        # Extract mean rewards and MSEs for all controllers and scenarios
        mean_rewards = np.zeros((len(controller_names), len(scenario_names)))
        mean_mses = np.zeros((len(controller_names), len(scenario_names)))
        
        for i, controller_name in enumerate(controller_names):
            for j, scenario_name in enumerate(scenario_names):
                if scenario_name in self.results[controller_name]:
                    mean_rewards[i, j] = np.mean(self.results[controller_name][scenario_name]["rewards"])
                    mean_mses[i, j] = np.mean(self.results[controller_name][scenario_name]["mse"])
        
        # Create bar plot for rewards
        plt.figure(figsize=(12, 6))
        x = np.arange(len(scenario_names))
        width = 0.8 / len(controller_names)
        
        for i, controller_name in enumerate(controller_names):
            plt.bar(x + (i - len(controller_names)/2 + 0.5) * width, mean_rewards[i], 
                   width, label=controller_name)
        
        plt.xlabel("Scenario")
        plt.ylabel("Mean Reward")
        plt.title("Mean Reward by Controller and Scenario")
        plt.xticks(x, scenario_names)
        plt.legend()
        plt.grid(axis='y')
        
        if save_plots:
            plt.savefig(os.path.join(plots_dir, "comparative_rewards.png"))
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Create bar plot for MSEs
        plt.figure(figsize=(12, 6))
        
        for i, controller_name in enumerate(controller_names):
            plt.bar(x + (i - len(controller_names)/2 + 0.5) * width, mean_mses[i], 
                   width, label=controller_name)
        
        plt.xlabel("Scenario")
        plt.ylabel("Mean MSE")
        plt.title("Mean MSE by Controller and Scenario")
        plt.xticks(x, scenario_names)
        plt.legend()
        plt.grid(axis='y')
        
        if save_plots:
            plt.savefig(os.path.join(plots_dir, "comparative_mses.png"))
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def generate_statistical_report(self, baseline_controller="Static PID", 
                                   test_name="t-test", alpha=0.05):
        """
        Generate a statistical report comparing controllers to a baseline.
        
        Args:
            baseline_controller (str): Name of the baseline controller
            test_name (str): Statistical test to use ("t-test" or "wilcoxon")
            alpha (float): Significance level
            
        Returns:
            pandas.DataFrame: Statistical report
        """
        if not self.results:
            print("No evaluation results available. Run evaluate_setpoint_tracking first.")
            return None
        
        if baseline_controller not in self.results:
            print(f"Baseline controller '{baseline_controller}' not found in results.")
            return None
        
        # Get all scenario names
        scenario_names = set()
        for controller_results in self.results.values():
            scenario_names.update(controller_results.keys())
        scenario_names = sorted(list(scenario_names))
        
        # Get controller names excluding baseline
        controller_names = [name for name in self.results.keys() if name != baseline_controller]
        
        # Prepare data for report
        report_data = []
        
        for scenario_name in scenario_names:
            # Get baseline rewards and MSEs
            if scenario_name in self.results[baseline_controller]:
                baseline_rewards = self.results[baseline_controller][scenario_name]["rewards"]
                baseline_mses = self.results[baseline_controller][scenario_name]["mse"]
                
                # Calculate baseline statistics
                baseline_mean_reward = np.mean(baseline_rewards)
                baseline_std_reward = np.std(baseline_rewards)
                baseline_mean_mse = np.mean(baseline_mses)
                baseline_std_mse = np.std(baseline_mses)
                
                # Compare each controller to baseline
                for controller_name in controller_names:
                    if scenario_name in self.results[controller_name]:
                        # Get controller rewards and MSEs
                        controller_rewards = self.results[controller_name][scenario_name]["rewards"]
                        controller_mses = self.results[controller_name][scenario_name]["mse"]
                        
                        # Calculate controller statistics
                        controller_mean_reward = np.mean(controller_rewards)
                        controller_std_reward = np.std(controller_rewards)
                        controller_mean_mse = np.mean(controller_mses)
                        controller_std_mse = np.std(controller_mses)
                        
                        # Calculate percent improvement
                        reward_improvement = ((controller_mean_reward - baseline_mean_reward) / 
                                             abs(baseline_mean_reward)) * 100
                        mse_improvement = ((baseline_mean_mse - controller_mean_mse) / 
                                          abs(baseline_mean_mse)) * 100
                        
                        # Perform statistical test
                        if test_name == "t-test":
                            # T-test for rewards
                            reward_t, reward_p = stats.ttest_ind(controller_rewards, baseline_rewards)
                            # T-test for MSEs
                            mse_t, mse_p = stats.ttest_ind(controller_mses, baseline_mses)
                            
                            test_statistic_reward = reward_t
                            p_value_reward = reward_p
                            test_statistic_mse = mse_t
                            p_value_mse = mse_p
                        
                        elif test_name == "wilcoxon":
                            # Wilcoxon test requires equal length arrays
                            min_len = min(len(controller_rewards), len(baseline_rewards))
                            # Wilcoxon test for rewards
                            reward_w, reward_p = stats.wilcoxon(controller_rewards[:min_len], 
                                                               baseline_rewards[:min_len])
                            # Wilcoxon test for MSEs
                            min_len = min(len(controller_mses), len(baseline_mses))
                            mse_w, mse_p = stats.wilcoxon(controller_mses[:min_len], 
                                                         baseline_mses[:min_len])
                            
                            test_statistic_reward = reward_w
                            p_value_reward = reward_p
                            test_statistic_mse = mse_w
                            p_value_mse = mse_p
                        
                        # Check significance
                        reward_significant = "Yes" if p_value_reward < alpha else "No"
                        mse_significant = "Yes" if p_value_mse < alpha else "No"
                        
                        # Add to report data
                        report_data.append({
                            "Scenario": scenario_name,
                            "Controller": controller_name,
                            "Baseline": baseline_controller,
                            "Baseline Mean Reward": baseline_mean_reward,
                            "Baseline Std Reward": baseline_std_reward,
                            "Controller Mean Reward": controller_mean_reward,
                            "Controller Std Reward": controller_std_reward,
                            "Reward Improvement (%)": reward_improvement,
                            "Reward Test Statistic": test_statistic_reward,
                            "Reward P-Value": p_value_reward,
                            "Reward Significant": reward_significant,
                            "Baseline Mean MSE": baseline_mean_mse,
                            "Baseline Std MSE": baseline_std_mse,
                            "Controller Mean MSE": controller_mean_mse,
                            "Controller Std MSE": controller_std_mse,
                            "MSE Improvement (%)": mse_improvement,
                            "MSE Test Statistic": test_statistic_mse,
                            "MSE P-Value": p_value_mse,
                            "MSE Significant": mse_significant
                        })
        
        # Create DataFrame
        report_df = pd.DataFrame(report_data)
        
        # Save report
        report_path = os.path.join(self.save_dir, "statistical_report.csv")
        report_df.to_csv(report_path, index=False)
        print(f"Statistical report saved to {report_path}")
        
        return report_df
    
    def plot_controller_heatmap(self, metric="reward", save_plots=True, show_plots=True):
        """
        Create a heatmap comparing controllers across scenarios.
        
        Args:
            metric (str): Metric to compare ("reward" or "mse")
            save_plots (bool): Whether to save plots to disk
            show_plots (bool): Whether to display plots
        """
        if not self.results:
            print("No evaluation results available. Run evaluate_setpoint_tracking first.")
            return
        
        # Create directory for plots
        plots_dir = os.path.join(self.save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Get all scenario names
        scenario_names = set()
        for controller_results in self.results.values():
            scenario_names.update(controller_results.keys())
        scenario_names = sorted(list(scenario_names))
        
        # Get all controller names
        controller_names = sorted(list(self.results.keys()))
        
        # Extract data for heatmap
        heatmap_data = np.zeros((len(controller_names), len(scenario_names)))
        
        for i, controller_name in enumerate(controller_names):
            for j, scenario_name in enumerate(scenario_names):
                if scenario_name in self.results[controller_name]:
                    if metric == "reward":
                        heatmap_data[i, j] = np.mean(self.results[controller_name][scenario_name]["rewards"])
                    elif metric == "mse":
                        heatmap_data[i, j] = np.mean(self.results[controller_name][scenario_name]["mse"])
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap_data, cmap='viridis' if metric == "reward" else 'coolwarm_r')
        
        # Add text annotations
        for i in range(len(controller_names)):
            for j in range(len(scenario_names)):
                plt.text(j, i, f"{heatmap_data[i, j]:.2f}", ha="center", va="center", 
                        color="white" if 0.2 < (heatmap_data[i, j] - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min()) < 0.8 else "black")
        
        # Add labels and title
        plt.xticks(np.arange(len(scenario_names)), scenario_names, rotation=45, ha="right")
        plt.yticks(np.arange(len(controller_names)), controller_names)
        plt.xlabel("Scenario")
        plt.ylabel("Controller")
        plt.title(f"Mean {'Reward' if metric == 'reward' else 'MSE'} by Controller and Scenario")
        plt.colorbar(label=f"Mean {'Reward' if metric == 'reward' else 'MSE'}")
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(plots_dir, f"heatmap_{metric}.png"))
        
        if show_plots:
            plt.show()
        else:
            plt.close()


def evaluate_all_controllers(static_pid_path=None, td3_path=None, sac_path=None, 
                            cirl_path=None, pso_path=None, save_dir="./results/evaluation",
                            difficulty="medium", n_episodes=5, render=False, verbose=True):
    """
    Convenience function to evaluate all available controllers.
    
    Args:
        static_pid_path (str): Path to static PID gains
        td3_path (str): Path to TD3 model
        sac_path (str): Path to SAC model
        cirl_path (str): Path to CIRL model
        pso_path (str): Path to PSO policy parameters
        save_dir (str): Directory to save results
        difficulty (str): Difficulty level for the environment
        n_episodes (int): Number of episodes per scenario
        render (bool): Whether to render the environment
        verbose (bool): Whether to print progress
        
    Returns:
        ControllerEvaluator: Controller evaluator with results
    """
    # Create environment with specified difficulty
    params = {
        'simulation_steps': 200,
        'dt': 1.0,
        'uncertainty_level': 0.0,
        'noise_level': 0.0,
        'actuator_delay_steps': 0,
        'transport_delay_steps': 0,
        'enable_disturbances': False
    }
    
    if difficulty == "easy":
        params['uncertainty_level'] = 0.02
        params['noise_level'] = 0.01
    elif difficulty == "medium":
        params['uncertainty_level'] = 0.05
        params['noise_level'] = 0.02
        params['actuator_delay_steps'] = 1
        params['transport_delay_steps'] = 1
        params['enable_disturbances'] = True
    elif difficulty == "hard":
        params['uncertainty_level'] = 0.1
        params['noise_level'] = 0.05
        params['actuator_delay_steps'] = 2
        params['transport_delay_steps'] = 2
        params['enable_disturbances'] = True
    elif difficulty == "extreme":
        params['uncertainty_level'] = 0.15
        params['noise_level'] = 0.08
        params['actuator_delay_steps'] = 3
        params['transport_delay_steps'] = 3
        params['enable_disturbances'] = True
    
    env = CSTRRLEnv(**params)
    
    # Create evaluator
    evaluator = ControllerEvaluator(save_dir=save_dir)
    
    # Add Static PID controller if available
    if static_pid_path:
        if verbose:
            print(f"Loading Static PID gains from {static_pid_path}")
        try:
            gains = np.load(static_pid_path)
            evaluator.add_static_pid_controller("Static PID", gains, env)
        except Exception as e:
            print(f"Error loading Static PID gains: {e}")
    
    # Set device for RL models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high[0]
    
    # Add TD3 agent if available
    if td3_path:
        if verbose:
            print(f"Loading TD3 model from {td3_path}")
        try:
            from RL_algorithms import create_agent
            td3_agent = create_agent("td3", state_dim, action_dim, action_high, device)
            td3_agent.load(td3_path)
            evaluator.add_rl_controller("TD3", td3_agent, "td3")
        except Exception as e:
            print(f"Error loading TD3 model: {e}")
    
    # Add SAC agent if available
    if sac_path:
        if verbose:
            print(f"Loading SAC model from {sac_path}")
        try:
            from RL_algorithms import create_agent
            sac_agent = create_agent("sac", state_dim, action_dim, action_high, device)
            sac_agent.load(sac_path)
            evaluator.add_rl_controller("SAC", sac_agent, "sac")
        except Exception as e:
            print(f"Error loading SAC model: {e}")
    
    # Add CIRL agent if available
    if cirl_path:
        if verbose:
            print(f"Loading CIRL model from {cirl_path}")
        try:
            from RL_algorithms import create_agent
            cirl_agent = create_agent("cirl", state_dim, action_dim, action_high, device)
            cirl_agent.load(cirl_path)
            evaluator.add_rl_controller("CIRL", cirl_agent, "cirl")
        except Exception as e:
            print(f"Error loading CIRL model: {e}")
    
    # Add PSO policy if available
    if pso_path:
        if verbose:
            print(f"Loading PSO policy from {pso_path}")
        try:
            policy_params = np.load(pso_path)
            evaluator.add_pso_controller("PSO-CIRL", policy_params)
        except Exception as e:
            print(f"Error loading PSO policy: {e}")
    
    # Define setpoint scenarios for evaluation
    setpoints_schedule = [
        {"name": "Increasing", "setpoints": [0.25, 0.50, 0.75]},
        {"name": "Decreasing", "setpoints": [0.75, 0.50, 0.25]},
        {"name": "Peak", "setpoints": [0.25, 0.80, 0.25]},
        {"name": "Valley", "setpoints": [0.75, 0.20, 0.75]},
        {"name": "Step", "setpoints": [0.3, 0.3, 0.7, 0.7]},
        {"name": "Random", "setpoints": np.random.uniform(0.2, 0.8, 4).tolist()}
    ]
    
    # Run evaluation
    evaluator.evaluate_setpoint_tracking(
        env=env,
        setpoints_schedule=setpoints_schedule,
        n_episodes=n_episodes,
        max_steps=200,
        render=render,
        verbose=verbose
    )
    
    # Generate plots and reports
    for scenario in [s["name"] for s in setpoints_schedule]:
        evaluator.plot_setpoint_tracking(scenario, save_plots=True, show_plots=False)
    
    evaluator.plot_comparative_metrics(save_plots=True, show_plots=False)
    evaluator.plot_controller_heatmap(metric="reward", save_plots=True, show_plots=False)
    evaluator.plot_controller_heatmap(metric="mse", save_plots=True, show_plots=False)
    
    # Generate statistical report
    if "Static PID" in evaluator.controllers:
        evaluator.generate_statistical_report(baseline_controller="Static PID")
    
    return evaluator


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate and compare CSTR controllers")
    
    parser.add_argument("--static_pid", type=str, default=None, 
                        help="Path to static PID gains")
    parser.add_argument("--td3", type=str, default=None, 
                        help="Path to TD3 model")
    parser.add_argument("--sac", type=str, default=None, 
                        help="Path to SAC model")
    parser.add_argument("--cirl", type=str, default=None, 
                        help="Path to CIRL model")
    parser.add_argument("--pso", type=str, default=None, 
                        help="Path to PSO policy")
    
    parser.add_argument("--save_dir", type=str, default="./results/evaluation", 
                        help="Directory to save results")
    parser.add_argument("--difficulty", type=str, default="medium", 
                        choices=["easy", "medium", "hard", "extreme"], 
                        help="Environment difficulty")
    parser.add_argument("--episodes", type=int, default=5, 
                        help="Number of episodes per scenario")
    parser.add_argument("--render", action="store_true", 
                        help="Render environment during evaluation")
    parser.add_argument("--show_plots", action="store_true", 
                        help="Show plots during evaluation")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = evaluate_all_controllers(
        static_pid_path=args.static_pid,
        td3_path=args.td3,
        sac_path=args.sac,
        cirl_path=args.cirl,
        pso_path=args.pso,
        save_dir=args.save_dir,
        difficulty=args.difficulty,
        n_episodes=args.episodes,
        render=args.render,
        verbose=True
    )
    
    # Show plots if requested
    if args.show_plots:
        for scenario in evaluator.results[list(evaluator.results.keys())[0]].keys():
            evaluator.plot_setpoint_tracking(scenario, save_plots=True, show_plots=True)
        
        evaluator.plot_comparative_metrics(save_plots=True, show_plots=True)
        evaluator.plot_controller_heatmap(metric="reward", save_plots=True, show_plots=True)
        evaluator.plot_controller_heatmap(metric="mse", save_plots=True, show_plots=True)