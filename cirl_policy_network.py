"""
Control-Informed Reinforcement Learning Policy Network

This module provides a neural network architecture for the CIRL framework,
which integrates PID control components into deep RL policies. The network
outputs PID gains that are used to control the CSTR system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CIRLPolicyNetwork(nn.Module):
    """
    Control-Informed Reinforcement Learning Policy Network that outputs PID gains.
    
    This network maps states to PID gains, allowing the RL agent to learn
    adaptive PID controllers that combine the stability of traditional control
    with the adaptability of reinforcement learning.
    """
    def __init__(self, state_dim, hidden_dims=[128, 128], activation=nn.ReLU):
        """
        Initialize the CIRL Policy Network.
        
        Args:
            state_dim (int): Dimension of the state space
            hidden_dims (list): Dimensions of the hidden layers
            activation: Activation function to use in the network
        """
        super(CIRLPolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        
        # Create network layers
        self.hidden1 = nn.Linear(state_dim, hidden_dims[0])
        self.hidden2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        
        # Output layer for PID gains [Kp_Cb, Ki_Cb, Kd_Cb, Kp_V, Ki_V, Kd_V]
        self.output = nn.Linear(hidden_dims[1], 6)
        
        # Activation function
        self.activation = activation()
    
    def forward(self, state):
        """
        Forward pass through the network, mapping states to PID gains.
        
        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: PID gains normalized to [-1, 1] range
        """
        x = state.float()
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.output(x)
        
        # Use tanh to bound outputs to [-1, 1]
        x = torch.tanh(x)
        
        return x
    
    def save(self, filepath):
        """
        Save the model weights to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath):
        """
        Load model weights from a file.
        
        Args:
            filepath (str): Path to load the model from
        """
        self.load_state_dict(torch.load(filepath))