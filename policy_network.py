"""
This module provides the neural network architectures used in reinforcement learning
for the CSTR control system. It includes implementations for:
1. Basic MLP policy network
2. Actor-Critic networks for TD3 and SAC algorithms
3. Control-informed RL (CIRL) network with PID structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    """
    Basic Multi-Layer Perceptron (MLP) network.
    
    This network serves as a building block for both policy and value functions.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128], 
                 activation=nn.ReLU, output_activation=None):
        """
        Initialize MLP network.
        
        Args:
            input_dim (int): Dimension of input features
            output_dim (int): Dimension of output
            hidden_dims (list): List of hidden layer dimensions
            activation (torch.nn.Module): Activation function for hidden layers
            output_activation (torch.nn.Module): Activation function for output layer
        """
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Create layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # Activation functions
        self.activation = activation()
        self.output_activation = output_activation() if output_activation else None

        # # Input normalization layer
        # self.input_norm = nn.BatchNorm1d(input_dim)
    
    def forward(self, x):
        """Forward pass through the network."""
        # # Input normalization
        # x = self.input_norm(x)

        # Apply hidden layers with activation
        for layer in self.layers:
            x = self.activation(layer(x))
        
        # Apply output layer
        x = self.output_layer(x)
        
        # Apply output activation if specified
        if self.output_activation:
            x = self.output_activation(x)
            
        return x


class PolicyNetwork(nn.Module):
    """
    Policy Network for direct action prediction => PID Gains
    
    This network outputs actions in the range [-1, 1] suitable for normalized control.
    """
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        """
        Initialize Policy Network.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dims (list): List of hidden layer dimensions
        """
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Use MLP with tanh activation for output to bound actions in [-1, 1]
        self.network = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=nn.ReLU,
            output_activation=nn.Tanh
        )
    
    def forward(self, state):
        """
        Compute action from state.
        
        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Action tensor of shape (batch_size, action_dim)
        """
        return self.network(state)


class QNetwork(nn.Module):
    """
    Q-Network for value function approximation.
    
    This network estimates Q-values given state-action pairs.
    """
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        """
        Initialize Q-Network.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dims (list): List of hidden layer dimensions
        """
        super(QNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Q-network takes state and action as input
        self.network = MLP(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=nn.ReLU
        )
    
    def forward(self, state, action):
        """
        Compute Q-value from state-action pair.
        
        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim)
            action (torch.Tensor): Action tensor of shape (batch_size, action_dim)
            
        Returns:
            torch.Tensor: Q-value tensor of shape (batch_size, 1)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class TwinQNetwork(nn.Module):
    """
    Twin Q-Network for TD3 and SAC algorithms.
    
    This implements two Q-networks to mitigate overestimation bias.
    """
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        """
        Initialize Twin Q-Network.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dims (list): List of hidden layer dimensions
        """
        super(TwinQNetwork, self).__init__()
        
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims)
    
    def forward(self, state, action):
        """
        Compute Q-values from both networks.
        
        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor
            
        Returns:
            tuple: Q-values from both networks
        """
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2
    
    def q1_forward(self, state, action):
        """Forward pass through only the first Q-network."""
        return self.q1(state, action)


class GaussianPolicyNetwork(nn.Module):
    """
    Gaussian Policy Network for SAC algorithm.
    
    This outputs a normal distribution over actions.
    """
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128], log_std_min=-20, log_std_max=2):
        """
        Initialize Gaussian Policy Network.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dims (list): List of hidden layer dimensions
            log_std_min (float): Minimum log standard deviation
            log_std_max (float): Maximum log standard deviation
        """
        super(GaussianPolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Main network
        self.network = MLP(
            input_dim=state_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            activation=nn.ReLU
        )
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
    
    def forward(self, state):
        """
        Compute action distribution from state.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            tuple: Mean and log standard deviation of action distribution
        """
        x = self.network(state)
        
        # Compute mean and log_std
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        """
        Sample action from the policy.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            tuple: Sampled action, log probability, and mean action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Create normal distribution
        normal = torch.distributions.Normal(mean, std)
        
        # Sample using reparameterization trick
        x_t = normal.rsample()
        
        # Apply tanh to bound actions in [-1, 1]
        y_t = torch.tanh(x_t)
        
        # Calculate log probability, accounting for the tanh transformation
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return y_t, log_prob, mean


class CIRLNetwork(nn.Module):
    """
    Control-Informed Reinforcement Learning (CIRL) Network.
    
    This network is structured to output PID gains for a specific control structure.
    """
    def __init__(self, state_dim, n_fc1=128, n_fc2=128, activation=nn.ReLU, 
                 output_sz=6, pid_mode=True):
        """
        Initialize CIRL Network.
        
        Args:
            state_dim (int): Dimension of state space
            n_fc1 (int): Size of first hidden layer
            n_fc2 (int): Size of second hidden layer
            activation (torch.nn.Module): Activation function
            output_sz (int): Dimension of output (6 for PID, 2 for direct control)
            pid_mode (bool): Whether to use PID structure
        """
        super(CIRLNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.output_sz = output_sz
        self.pid_mode = pid_mode
        
        # Network structure
        self.hidden1 = nn.Linear(state_dim, n_fc1, bias=True)
        self.act = activation()
        self.hidden2 = nn.Linear(n_fc1, n_fc2, bias=True)
        self.output = nn.Linear(n_fc2, output_sz, bias=True)
    
    def forward(self, x):
        """
        Compute PID gains from state.
        
        Args:
            x (torch.Tensor): State tensor
            
        Returns:
            torch.Tensor: PID gains as a tensor, normalized to [-1, 1]
        """
        x = x.float()
        y = self.act(self.hidden1(x))
        y = self.act(self.hidden2(y))
        out = self.output(y)
        
        # Bound output to [-1, 1]
        y = F.tanh(out)
        
        return y


# Helper function to initialize network weights
def init_weights(module, gain=1.0):
    """
    Initialize network weights using orthogonal initialization.
    
    Args:
        module (torch.nn.Module): Network module
        gain (float): Scaling factor for weights
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)