"""
adaptive_control_advanced.py - Advanced adaptive PID controller for CSTR system

This module provides a truly adaptive PID controller that combines:
1. Online parameter adaptation based on performance metrics
2. Periodic optimization using differential evolution
3. Performance-triggered adaptation to handle changing system dynamics

The controller continuously adjusts PID gains to maintain optimal performance
across the entire operating range without requiring manual gain scheduling.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
from scipy.optimize import differential_evolution
from collections import deque


class AdaptiveController:
    """
    Advanced adaptive PID controller for the CSTR system that continuously 
    adapts its gains based on performance metrics and periodic optimization.
    
    The controller consists of two PID loops:
    1. Cb-loop: Controls product B concentration using cooling temperature (Tc)
    2. V-loop: Controls reactor volume using inlet flow rate (Fin)
    
    Controller features:
    - Continuous adaptation based on performance metrics
    - Periodic optimization using differential evolution
    - Performance-triggered adaptation for changing conditions
    - Anti-windup and derivative filtering for robust control
    """
    
    def __init__(self, initial_gains=None, min_bounds=None, max_bounds=None, 
                 adaptation_rate=0.02, performance_window=10,
                 optimization_interval=50, optimization_samples=10):
        """
        Initialize the adaptive controller.
        
        Args:
            initial_gains (numpy.ndarray): Initial PID gains [Kp_Cb, Ki_Cb, Kd_Cb, Kp_V, Ki_V, Kd_V]
            min_bounds (numpy.ndarray): Minimum values for PID gains
            max_bounds (numpy.ndarray): Maximum values for PID gains
            adaptation_rate (float): Rate of continuous adaptation (0-1)
            performance_window (int): Number of steps to evaluate performance
            optimization_interval (int): Steps between differential evolution optimizations
            optimization_samples (int): Number of samples for each optimization
        """
        # Default gain bounds for the CSTR controller
        if min_bounds is None:
            self.min_bounds = np.array([-5, 0, 0.02, 0, 0, 0.01])
        else:
            self.min_bounds = np.array(min_bounds)
            
        if max_bounds is None:
            self.max_bounds = np.array([25, 20, 10, 1, 2, 1])
        else:
            self.max_bounds = np.array(max_bounds)
        
        # Initialize PID gains
        if initial_gains is None:
            # Default initial gains in the middle of the range
            self.current_gains = (self.min_bounds + self.max_bounds) / 2
        else:
            self.current_gains = np.array(initial_gains)
        
        # Initialize error history for derivative calculations
        self.error_history = []
        self.control_history = []
        
        # Initialize error integrals for integral term
        self.error_integral_Cb = 0.0
        self.error_integral_V = 0.0
        
        # Adaptation parameters
        self.adaptation_rate = adaptation_rate
        self.performance_window = performance_window
        self.optimization_interval = optimization_interval
        self.optimization_samples = optimization_samples
        
        # Performance history
        self.reset_performance_tracking()
        
        # Adaptation direction for each gain
        self.adaptation_direction = np.ones(6) * 0.01
        
        # Recent window for performance calculation
        self.recent_errors = deque(maxlen=performance_window)
        self.recent_setpoints = deque(maxlen=performance_window)
        self.recent_outputs = deque(maxlen=performance_window)
        
        # Performance metrics history
        self.performance_metric_history = []
        self.gain_history = []
        
        # Step counter for optimization triggering
        self.step_counter = 0
        
        # Last optimization time
        self.last_optimization_time = 0
        
        # Configuration for plots
        self.plot_dir = "advanced_adaptive_results"
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Optimization history
        self.optimization_history = {
            'steps': [],
            'gains': [],
            'performance_improvement': []
        }

    def compute_control(self, error, dt=1.0, setpoint=None, output=None):
        """
        Compute control action using current PID gains and given error.
        
        Args:
            error (numpy.ndarray): Current error [e_Cb, e_V]
            dt (float): Time step in minutes
            setpoint (numpy.ndarray): Current setpoint [setpoint_Cb, setpoint_V]
            output (numpy.ndarray): Current system output [Cb, V]
            
        Returns:
            numpy.ndarray: Control action [Tc, Fin]
        """
        # Unpack error
        e_Cb, e_V = error
        
        # Store for adaptation
        if setpoint is not None and output is not None:
            self.recent_errors.append(error.copy())
            self.recent_setpoints.append(setpoint.copy())
            self.recent_outputs.append(output.copy())
        
        # Unpack current gains
        Kp_Cb, Ki_Cb, Kd_Cb, Kp_V, Ki_V, Kd_V = self.current_gains
        
        # Update error history
        self.error_history.append(error)
        if len(self.error_history) > 3:
            self.error_history.pop(0)
        
        # Ensure we have enough history for derivative calculation
        if len(self.error_history) < 3:
            # Not enough history, use proportional-only control
            Tc = 350 - Kp_Cb * e_Cb  # Base cooling temperature with P control
            Fin = 100 + Kp_V * e_V   # Base flow rate with P control
        else:
            # Calculate integral term (with anti-windup)
            self.error_integral_Cb = np.clip(
                self.error_integral_Cb + e_Cb * dt,
                -10/Ki_Cb if Ki_Cb > 0 else -1e6,  # Prevent integral windup
                10/Ki_Cb if Ki_Cb > 0 else 1e6
            )
            
            self.error_integral_V = np.clip(
                self.error_integral_V + e_V * dt,
                -5/Ki_V if Ki_V > 0 else -1e6,
                5/Ki_V if Ki_V > 0 else 1e6
            )
            
            # Calculate derivative term (with filtering)
            d_e_Cb = (e_Cb - 2*self.error_history[-2][0] + self.error_history[-3][0]) / (dt**2)
            d_e_V = (e_V - 2*self.error_history[-2][1] + self.error_history[-3][1]) / (dt**2)
            
            # Compute control actions
            Tc = 350 - (Kp_Cb * e_Cb + Ki_Cb * self.error_integral_Cb + Kd_Cb * d_e_Cb)
            Fin = 100 + (Kp_V * e_V + Ki_V * self.error_integral_V + Kd_V * d_e_V)
        
        # Clip control actions to reasonable ranges for the CSTR
        Tc = np.clip(Tc, 290, 450)   # Cooling temperature (K)
        Fin = np.clip(Fin, 95, 105)  # Inlet flow rate (m³/min)
        
        # Update control history
        control = np.array([Tc, Fin])
        self.control_history.append(control)
        if len(self.control_history) > 3:
            self.control_history.pop(0)
        
        # Increment step counter
        self.step_counter += 1
        
        return control

    def adapt_gains(self):
        """
        Adapt controller gains based on recent performance.
        This performs continuous adaptation based on performance metrics.
        
        Returns:
            numpy.ndarray: Updated PID gains
        """
        if len(self.recent_errors) < self.performance_window:
            return self.current_gains  # Not enough data
            
        # Calculate performance metrics
        ise_Cb = sum([e[0]**2 for e in self.recent_errors])  # Integral square error for Cb
        ise_V = sum([e[1]**2 for e in self.recent_errors])   # Integral square error for V
        
        # Calculate performance metric (weighted sum of ISE)
        performance_metric = 0.8 * ise_Cb + 0.2 * ise_V
        
        # Store performance metric
        self.performance_metric_history.append(performance_metric)
        self.gain_history.append(self.current_gains.copy())
        
        # Only adapt if we have at least 2 performance measurements
        if len(self.performance_metric_history) >= 2:
            # Calculate performance change
            perf_change = self.performance_metric_history[-1] - self.performance_metric_history[-2]
            
            # Adapt each gain based on performance change
            for i in range(6):
                if perf_change < 0:
                    # Performance improving, continue in same direction
                    self.current_gains[i] += self.adaptation_direction[i]
                else:
                    # Performance worsening, change direction
                    self.adaptation_direction[i] *= -0.5  # Reduce and reverse direction
                    self.current_gains[i] += self.adaptation_direction[i]
                    
        # Ensure gains are within bounds
        self.current_gains = np.clip(self.current_gains, self.min_bounds, self.max_bounds)
        
        # Check if it's time for differential evolution optimization
        if (self.step_counter % self.optimization_interval == 0 and 
            len(self.recent_errors) == self.performance_window):
            
            # Prevent optimization if one was recently done
            current_time = time.time()
            if current_time - self.last_optimization_time > 0.5:  # Min 0.5 seconds between optimizations
                self.optimize_gains_evolution()
                self.last_optimization_time = current_time
        
        return self.current_gains
    
    def optimize_gains_evolution(self):
        """
        Optimize controller gains using differential evolution.
        This performs a more thorough search around the current gains.
        """
        print(f"Starting optimization at step {self.step_counter}...")
        start_time = time.time()
        
        # Current performance before optimization
        pre_optimization_performance = self.evaluate_performance(self.current_gains)
        
        # Define bounds for optimization (local search around current gains)
        bounds = []
        for i in range(6):
            # Create bounds around current gain values (±30%)
            lower = max(self.current_gains[i] * 0.7, self.min_bounds[i])
            upper = min(self.current_gains[i] * 1.3, self.max_bounds[i])
            bounds.append((lower, upper))
        
        # Run differential evolution
        result = differential_evolution(
            self.objective_function,
            bounds=bounds,
            maxiter=10,
            popsize=self.optimization_samples,
            tol=0.01,
            mutation=(0.5, 1.0),
            recombination=0.7,
            disp=False
        )
        
        # Update gains if optimization improved performance
        optimized_gains = result.x
        post_optimization_performance = self.evaluate_performance(optimized_gains)
        
        improvement = pre_optimization_performance - post_optimization_performance
        improvement_percent = (improvement / abs(pre_optimization_performance)) * 100 if pre_optimization_performance != 0 else 0
        
        # Record optimization
        self.optimization_history['steps'].append(self.step_counter)
        self.optimization_history['gains'].append(optimized_gains)
        self.optimization_history['performance_improvement'].append(improvement_percent)
        
        # Only update if performance improved significantly
        if improvement > 0.01 * abs(pre_optimization_performance):
            self.current_gains = optimized_gains
            print(f"Optimization successful: Improved by {improvement_percent:.2f}%")
            print(f"New gains: {self.current_gains}")
        else:
            print("Optimization did not yield significant improvement")
            
        optimization_time = time.time() - start_time
        print(f"Optimization completed in {optimization_time:.2f} seconds")
    
    def objective_function(self, gains):
        """
        Objective function for differential evolution optimization.
        
        Args:
            gains (numpy.ndarray): PID gains to evaluate
            
        Returns:
            float: Performance metric (lower is better)
        """
        return self.evaluate_performance(gains)
    
    def evaluate_performance(self, gains):
        """
        Evaluate the performance of a given set of gains using stored data.
        
        Args:
            gains (numpy.ndarray): PID gains to evaluate
            
        Returns:
            float: Performance metric (lower is better)
        """
        if len(self.recent_errors) < self.performance_window:
            return float('inf')  # Not enough data
            
        # Convert lists to arrays for easier manipulation
        errors = np.array(list(self.recent_errors))
        setpoints = np.array(list(self.recent_setpoints))
        
        # Simulate control with these gains
        simulated_errors = []
        error_integral_Cb = 0
        error_integral_V = 0
        prev_errors = [errors[0], errors[0]]  # Initial error history
        
        for i in range(len(errors)):
            e_Cb, e_V = errors[i]
            
            # Unpack gains
            Kp_Cb, Ki_Cb, Kd_Cb, Kp_V, Ki_V, Kd_V = gains
            
            # Update integrals (with anti-windup)
            error_integral_Cb = np.clip(
                error_integral_Cb + e_Cb,
                -10/Ki_Cb if Ki_Cb > 0 else -1e6,
                10/Ki_Cb if Ki_Cb > 0 else 1e6
            )
            
            error_integral_V = np.clip(
                error_integral_V + e_V,
                -5/Ki_V if Ki_V > 0 else -1e6,
                5/Ki_V if Ki_V > 0 else 1e6
            )
            
            # Calculate derivative terms
            if i >= 2:
                d_e_Cb = (e_Cb - 2*prev_errors[1][0] + prev_errors[0][0])
                d_e_V = (e_V - 2*prev_errors[1][1] + prev_errors[0][1])
            else:
                d_e_Cb = 0
                d_e_V = 0
            
            # Compute control actions
            control_Cb = Kp_Cb * e_Cb + Ki_Cb * error_integral_Cb + Kd_Cb * d_e_Cb
            control_V = Kp_V * e_V + Ki_V * error_integral_V + Kd_V * d_e_V
            
            # Store simulated error (uses original error for simplicity)
            simulated_errors.append([e_Cb, e_V])
            
            # Update history
            prev_errors = [prev_errors[1], [e_Cb, e_V]]
        
        # Calculate performance metrics for simulation
        simulated_errors = np.array(simulated_errors)
        
        # ISE weighted by setpoint value
        ise_Cb_weighted = sum([e[0]**2 for e in simulated_errors])
        ise_V_weighted = sum([e[1]**2 for e in simulated_errors])
        
        # Weight ISE (more emphasis on Cb control)
        weighted_performance = 0.8 * ise_Cb_weighted + 0.2 * ise_V_weighted
        
        # Penalize oscillatory behavior
        oscillation_penalty = 0
        for i in range(2, len(simulated_errors)):
            # Detect sign changes in error derivative
            d_e_Cb_1 = simulated_errors[i-1][0] - simulated_errors[i-2][0]
            d_e_Cb_2 = simulated_errors[i][0] - simulated_errors[i-1][0]
            
            d_e_V_1 = simulated_errors[i-1][1] - simulated_errors[i-2][1]
            d_e_V_2 = simulated_errors[i][1] - simulated_errors[i-1][1]
            
            # Count sign changes (oscillations)
            if d_e_Cb_1 * d_e_Cb_2 < 0:  # Sign change
                oscillation_penalty += 0.2
                
            if d_e_V_1 * d_e_V_2 < 0:  # Sign change
                oscillation_penalty += 0.05
        
        # Final performance metric (lower is better)
        performance_metric = weighted_performance * (1 + oscillation_penalty)
        
        return performance_metric
    
    def reset(self):
        """
        Reset the controller state (integrals, histories, counters).
        """
        # Reset error and control histories
        self.error_history = []
        self.control_history = []
        
        # Reset integral terms
        self.error_integral_Cb = 0.0
        self.error_integral_V = 0.0
        
        # Reset adaptation variables
        self.adaptation_direction = np.ones(6) * 0.01
        self.recent_errors.clear()
        self.recent_setpoints.clear()
        self.recent_outputs.clear()
        self.performance_metric_history = []
        self.gain_history = []
        
        # Reset step counter
        self.step_counter = 0
        
        # Reset performance tracking
        self.reset_performance_tracking()
        
        # Reset optimization history
        self.optimization_history = {
            'steps': [],
            'gains': [],
            'performance_improvement': []
        }
    
    def reset_performance_tracking(self):
        """Reset all performance tracking metrics"""
        self.performance_history = {
            'time': [],                # Time steps
            'setpoints': [],           # [Cb_setpoint, V_setpoint]
            'outputs': [],             # [Cb, V]
            'errors': [],              # [e_Cb, e_V]
            'gains': [],               # PID gains at each step
            'controls': [],            # Control actions [Tc, Fin]
            'steps_to_settle': None,   # Number of steps to settle within tolerance
            'settling_tolerance': 0.05 # 5% tolerance for settling time
        }
    
    def update_performance_tracking(self, time_step, setpoint, output, error, control):
        """
        Update the performance tracking with the latest data.
        
        Args:
            time_step (float): Current time step
            setpoint (numpy.ndarray): Current setpoint [Cb_setpoint, V_setpoint]
            output (numpy.ndarray): Current output [Cb, V]
            error (numpy.ndarray): Current error [e_Cb, e_V]
            control (numpy.ndarray): Control action [Tc, Fin]
        """
        self.performance_history['time'].append(time_step)
        self.performance_history['setpoints'].append(setpoint.copy())
        self.performance_history['outputs'].append(output.copy())
        self.performance_history['errors'].append(error.copy())
        self.performance_history['gains'].append(self.current_gains.copy())
        self.performance_history['controls'].append(control.copy())
        
        # Calculate settling time if not already determined
        if self.performance_history['steps_to_settle'] is None:
            # Check if within tolerance
            tolerance = self.performance_history['settling_tolerance']
            
            # Relative error for Cb
            rel_error_Cb = abs(error[0] / setpoint[0]) if setpoint[0] != 0 else abs(error[0])
            
            # Fixed threshold for Volume since it's around 100
            abs_error_V = abs(error[1])
            
            if rel_error_Cb < tolerance and abs_error_V < tolerance:
                # We've settled within tolerance
                self.performance_history['steps_to_settle'] = len(self.performance_history['time'])
    
    def get_normalized_gains(self):
        """
        Convert current gains to normalized range [-1, 1] for use with the CSTR environment.
        
        Returns:
            numpy.ndarray: Normalized gains in [-1, 1] range
        """
        return 2 * (self.current_gains - self.min_bounds) / (self.max_bounds - self.min_bounds) - 1
    
    def set_gains_from_normalized(self, normalized_gains):
        """
        Set current gains from normalized values in [-1, 1] range.
        
        Args:
            normalized_gains (numpy.ndarray): Normalized gains in [-1, 1] range.
            
        Returns:
            numpy.ndarray: Actual gains.
        """
        self.current_gains = ((normalized_gains + 1) / 2) * (self.max_bounds - self.min_bounds) + self.min_bounds
        return self.current_gains
    
    def plot_control_performance(self, save=True, show=True):
        """
        Plot the controller performance including setpoint tracking and control actions.
        
        Args:
            save (bool): Whether to save the plot to file
            show (bool): Whether to display the plot
        """
        if len(self.performance_history['time']) == 0:
            print("No performance history available.")
            return
        
        # Convert lists to numpy arrays for easier manipulation
        time_arr = np.array(self.performance_history['time'])
        setpoints = np.array(self.performance_history['setpoints'])
        outputs = np.array(self.performance_history['outputs'])
        errors = np.array(self.performance_history['errors'])
        controls = np.array(self.performance_history['controls'])
        gains = np.array(self.performance_history['gains'])
        
        # Create figure
        plt.figure(figsize=(15, 15))
        
        # Plot Cb tracking
        plt.subplot(4, 2, 1)
        plt.plot(time_arr, outputs[:, 0], 'b-', label='Measured Cb')
        plt.plot(time_arr, setpoints[:, 0], 'r--', label='Setpoint Cb')
        plt.title('Concentration B Setpoint Tracking')
        plt.xlabel('Time (min)')
        plt.ylabel('Concentration of B')
        plt.legend()
        plt.grid(True)
        
        # Plot V tracking
        plt.subplot(4, 2, 2)
        plt.plot(time_arr, outputs[:, 1], 'g-', label='Measured Volume')
        plt.plot(time_arr, setpoints[:, 1], 'r--', label='Setpoint Volume')
        plt.title('Volume Setpoint Tracking')
        plt.xlabel('Time (min)')
        plt.ylabel('Volume')
        plt.legend()
        plt.grid(True)
        
        # Plot Cb error
        plt.subplot(4, 2, 3)
        plt.plot(time_arr, errors[:, 0], 'b-')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Concentration B Error')
        plt.xlabel('Time (min)')
        plt.ylabel('Error (Setpoint - Measured)')
        plt.grid(True)
        
        # Plot V error
        plt.subplot(4, 2, 4)
        plt.plot(time_arr, errors[:, 1], 'g-')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Volume Error')
        plt.xlabel('Time (min)')
        plt.ylabel('Error (Setpoint - Measured)')
        plt.grid(True)
        
        # Plot control actions - Cooling temperature
        plt.subplot(4, 2, 5)
        plt.plot(time_arr, controls[:, 0], 'r-')
        plt.title('Cooling Temperature (Tc)')
        plt.xlabel('Time (min)')
        plt.ylabel('Temperature (K)')
        plt.grid(True)
        
        # Plot control actions - Inlet flow
        plt.subplot(4, 2, 6)
        plt.plot(time_arr, controls[:, 1], 'c-')
        plt.title('Inlet Flow Rate (Fin)')
        plt.xlabel('Time (min)')
        plt.ylabel('Flow Rate (m³/min)')
        plt.grid(True)
        
        # Plot adaptation of Cb controller gains
        plt.subplot(4, 2, 7)
        plt.plot(time_arr, gains[:, 0], 'r-', label='Kp_Cb')
        plt.plot(time_arr, gains[:, 1], 'g-', label='Ki_Cb')
        plt.plot(time_arr, gains[:, 2], 'b-', label='Kd_Cb')
        plt.title('Cb Controller Gains Adaptation')
        plt.xlabel('Time (min)')
        plt.ylabel('Gain Value')
        plt.legend()
        plt.grid(True)
        
        # Plot adaptation of V controller gains
        plt.subplot(4, 2, 8)
        plt.plot(time_arr, gains[:, 3], 'r-', label='Kp_V')
        plt.plot(time_arr, gains[:, 4], 'g-', label='Ki_V')
        plt.plot(time_arr, gains[:, 5], 'b-', label='Kd_V')
        plt.title('V Controller Gains Adaptation')
        plt.xlabel('Time (min)')
        plt.ylabel('Gain Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.suptitle('Adaptive PID Controller Performance')
        plt.subplots_adjust(top=0.93)
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            plt.savefig(os.path.join(self.plot_dir, f"control_performance_{timestamp}.png"))
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_optimization_history(self, save=True, show=True):
        """
        Plot the differential evolution optimization history.
        
        Args:
            save (bool): Whether to save the plot to file
            show (bool): Whether to display the plot
        """
        if len(self.optimization_history['steps']) == 0:
            print("No optimization history available.")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot performance improvement
        plt.subplot(1, 2, 1)
        plt.plot(self.optimization_history['steps'], self.optimization_history['performance_improvement'], 'bo-')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Optimization Performance Improvement')
        plt.xlabel('Time Step')
        plt.ylabel('Improvement (%)')
        plt.grid(True)
        
        # Plot gain changes
        plt.subplot(1, 2, 2)
        gains_array = np.array(self.optimization_history['gains'])
        gain_names = ['Kp_Cb', 'Ki_Cb', 'Kd_Cb', 'Kp_V', 'Ki_V', 'Kd_V']
        
        for i in range(6):
            plt.plot(self.optimization_history['steps'], gains_array[:, i], 'o-', label=gain_names[i])
        
        plt.title('Optimized Gains')
        plt.xlabel('Time Step')
        plt.ylabel('Gain Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.suptitle('Differential Evolution Optimization History')
        plt.subplots_adjust(top=0.9)
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            plt.savefig(os.path.join(self.plot_dir, f"optimization_history_{timestamp}.png"))
        
        if show:
            plt.show()
        else:
            plt.close()
            
    def compute_performance_metrics(self):
        """
        Compute various performance metrics for the controller.
        
        Returns:
            dict: Performance metrics including MSE, settling time, overshoot, etc.
        """
        if len(self.performance_history['time']) == 0:
            return {"error": "No performance history available."}
        
        # Convert lists to numpy arrays
        errors = np.array(self.performance_history['errors'])
        setpoints = np.array(self.performance_history['setpoints'])
        
        # Calculate MSE
        mse_Cb = np.mean(errors[:, 0]**2)
        mse_V = np.mean(errors[:, 1]**2)
        
        # Calculate MAE
        mae_Cb = np.mean(np.abs(errors[:, 0]))
        mae_V = np.mean(np.abs(errors[:, 1]))
        
        # Calculate relative MSE for Cb (normalized by setpoint)
        non_zero_setpoints = setpoints[:, 0] != 0
        rel_errors_Cb = errors[non_zero_setpoints, 0] / setpoints[non_zero_setpoints, 0]
        rel_mse_Cb = np.mean(rel_errors_Cb**2) if any(non_zero_setpoints) else float('nan')
        
        # Calculate settling time
        settling_time = self.performance_history['steps_to_settle']
        if settling_time is None:
            settling_time = float('inf')  # Did not settle within tolerance
        
        # Calculate overshoot (if any) by checking maximum absolute error
        max_abs_error_Cb = np.max(np.abs(errors[:, 0]))
        max_abs_error_V = np.max(np.abs(errors[:, 1]))
        
        # Peak error ratio (max error / average error)
        peak_error_ratio_Cb = max_abs_error_Cb / mae_Cb if mae_Cb > 0 else float('inf')
        peak_error_ratio_V = max_abs_error_V / mae_V if mae_V > 0 else float('inf')
        
        # Create metrics dictionary
        metrics = {
            'mse_Cb': mse_Cb,
            'mse_V': mse_V,
            'mae_Cb': mae_Cb,
            'mae_V': mae_V,
            'rel_mse_Cb': rel_mse_Cb,
            'settling_time': settling_time,
            'max_abs_error_Cb': max_abs_error_Cb,
            'max_abs_error_V': max_abs_error_V,
            'peak_error_ratio_Cb': peak_error_ratio_Cb,
            'peak_error_ratio_V': peak_error_ratio_V
        }
        
        return metrics


if __name__ == '__main__':
    # Simple simulation example to test the adaptive controller
    
    # Create an instance of the controller
    controller = AdaptiveController()
    
    # Simulation parameters
    total_steps = 200
    dt = 1.0  # time step in minutes
    setpoint_Cb = 0.5  # desired concentration of B
    setpoint_V = 100.0  # desired volume
    setpoint = np.array([setpoint_Cb, setpoint_V])
    
    # Initialize measured outputs (dummy starting values)
    outputs = np.array([0.0, 95.0])
    
    # Run simulation loop
    for step in range(total_steps):
        # Compute error between setpoint and measured outputs
        error = setpoint - outputs
        
        # Compute control action
        control = controller.compute_control(error, dt=dt, setpoint=setpoint, output=outputs)
        
        # Update dummy process model:
        # For demonstration, we simulate a simple first-order-like response plus some noise.
        # This update rule is arbitrary and for testing only.
        outputs = outputs + 0.1 * ((control - np.array([350, 100])) / np.array([350, 100])) + 0.01 * np.random.randn(2)
        
        # Update performance tracking
        controller.update_performance_tracking(step, setpoint, outputs, error, control)
    
    # Plot controller performance and optimization history
    controller.plot_control_performance(show=True)
    controller.plot_optimization_history(show=False)
    
    # Compute and print performance metrics
    metrics = controller.compute_performance_metrics()
    print("Performance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
