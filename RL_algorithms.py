"""
RL_algorithms.py - Reinforcement learning algorithms for CSTR control

This module implements various reinforcement learning algorithms for controlling
the CSTR system, including:
1. Twin Delayed Deep Deterministic Policy Gradient (TD3)
2. Soft Actor-Critic (SAC)
3. Control-Informed Reinforcement Learning (CIRL)

Each algorithm is implemented as a class with methods for training and inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from policy_network import PolicyNetwork, TwinQNetwork, GaussianPolicyNetwork, CIRLNetwork


class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
    
    TD3 is an actor-critic algorithm that addresses overestimation bias in Q-value
    estimation by using twin critics and delayed policy updates.
    """
    def __init__(self, state_dim, action_dim, action_high, hidden_dims=[256, 256], 
                 lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, tau=0.005, 
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, device="cuda"):
        """
        Initialize TD3 agent.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            action_high (float): Maximum action value
            hidden_dims (list): Hidden layer dimensions for networks
            lr_actor (float): Learning rate for actor
            lr_critic (float): Learning rate for critic
            gamma (float): Discount factor
            tau (float): Soft update coefficient
            policy_noise (float): Noise added to target policy
            noise_clip (float): Clipping value for policy noise
            policy_freq (int): Frequency of policy updates
            device (str): Device for torch tensors
        """
        self.action_dim = action_dim
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.device = device
        
        # Initialize actor network
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Initialize critic networks
        self.critic = TwinQNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.total_it = 0  # Total training iterations
    
    def select_action(self, state, noise=0.1):
        """
        Select an action given the current state.
        
        Args:
            state: Current state
            noise (float): Exploration noise scale
            
        Returns:
            numpy.ndarray: Selected action
        """
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        else:
            state = state.to(self.device)
        
        # Get action from policy
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        
        # Add exploration noise
        if noise > 0:
            action = action + np.random.normal(0, noise, size=self.action_dim)
            action = np.clip(action, -1, 1)
        
        return action
    
    def train(self, replay_buffer, batch_size=256):
        """
        Update the networks using a batch of transitions.
        
        Args:
            replay_buffer: Replay buffer containing transitions
            batch_size (int): Batch size for updates
            
        Returns:
            tuple: (actor_loss, critic_loss)
        """
        self.total_it += 1
        
        # Sample a batch of transitions from replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # ---------------------------- Update Critic ---------------------------- #
        # Get next action from target policy with noise for smoothing
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
        
        # Calculate target Q-values
        target_q1, target_q2 = self.critic_target(next_state, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward + (1 - done) * self.gamma * target_q
        
        # Calculate current Q-values
        current_q1, current_q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---------------------------- Update Actor ---------------------------- #
        actor_loss = None
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update_targets()
            
            actor_loss = actor_loss.item()
        
        return actor_loss, critic_loss.item()
    
    def _soft_update_targets(self):
        """
        Soft update target networks using polyak averaging.
        """
        # Update critic target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Update actor target
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filename):
        """
        Save model parameters to file.
        
        Args:
            filename (str): Path to save file
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_it': self.total_it
        }, filename)
        
    def load(self, filename):
        """
        Load model parameters from file.
        
        Args:
            filename (str): Path to load file
        """
        checkpoint = torch.load(filename)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_it = checkpoint['total_it']


class SAC:
    """
    Soft Actor-Critic (SAC) agent.
    
    SAC is an off-policy actor-critic algorithm that optimizes a stochastic policy 
    with entropy regularization.
    """
    def __init__(self, state_dim, action_dim, action_high, hidden_dims=[256, 256],
                 lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4, gamma=0.99, tau=0.005,
                 alpha=0.2, auto_alpha_tuning=True, device="cuda"):
        """
        Initialize SAC agent.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            action_high (float): Maximum action value
            hidden_dims (list): Hidden layer dimensions for networks
            lr_actor (float): Learning rate for actor
            lr_critic (float): Learning rate for critic
            lr_alpha (float): Learning rate for alpha (entropy coefficient)
            gamma (float): Discount factor
            tau (float): Soft update coefficient
            alpha (float): Initial entropy coefficient
            auto_alpha_tuning (bool): Whether to automatically tune alpha
            device (str): Device for torch tensors
        """
        self.action_dim = action_dim
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device
        self.automatic_entropy_tuning = auto_alpha_tuning
        
        # Initialize actor network (Gaussian policy)
        self.actor = GaussianPolicyNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Initialize critic networks
        self.critic = TwinQNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -action_dim  # Heuristic: -dim(A)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
    
    def select_action(self, state, evaluate=False):
        """
        Select an action given the current state.
        
        Args:
            state: Current state
            evaluate (bool): Whether to use deterministic policy for evaluation
            
        Returns:
            numpy.ndarray: Selected action
        """
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        else:
            state = state.to(self.device)
        
        with torch.no_grad():
            if evaluate:
                # For evaluation, use the mean of the policy (deterministic)
                _, _, action = self.actor.sample(state)
            else:
                # For training, sample from the policy (stochastic)
                action, _, _ = self.actor.sample(state)
        
        return action.cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size=256):
        """
        Update the networks using a batch of transitions.
        
        Args:
            replay_buffer: Replay buffer containing transitions
            batch_size (int): Batch size for updates
            
        Returns:
            tuple: (actor_loss, critic_loss, alpha_loss)
        """
        # Sample a batch of transitions from replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # ---------------------------- Update Critic ---------------------------- #
        with torch.no_grad():
            # Sample actions from the policy for next states
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            
            # Calculate target Q-values
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Calculate current Q-values
        current_q1, current_q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---------------------------- Update Actor ---------------------------- #
        # Sample actions and log probs from the policy
        pi_action, log_prob, _ = self.actor.sample(state)
        
        # Calculate Q-values for policy actions
        q1, q2 = self.critic(state, pi_action)
        min_q = torch.min(q1, q2)
        
        # Compute actor loss (maximizing expected Q - alpha * log_prob)
        actor_loss = (self.alpha * log_prob - min_q).mean()
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ---------------------------- Update Alpha ---------------------------- #
        alpha_loss = 0
        if self.automatic_entropy_tuning:
            # Calculate alpha loss
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            # Optimize alpha
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # Update alpha value
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target networks
        self._soft_update_targets()
        
        return actor_loss.item(), critic_loss.item(), alpha_loss.item() if self.automatic_entropy_tuning else 0
    
    def _soft_update_targets(self):
        """
        Soft update target networks using polyak averaging.
        """
        # Update critic target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filename):
        """
        Save model parameters to file.
        
        Args:
            filename (str): Path to save file
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None
        }, filename)
        
    def load(self, filename):
        """
        Load model parameters from file.
        
        Args:
            filename (str): Path to load file
        """
        checkpoint = torch.load(filename)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        if self.automatic_entropy_tuning:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.alpha = self.log_alpha.exp().item()


class CIRL:
    """
    Control-Informed Reinforcement Learning (CIRL) agent.
    
    CIRL integrates control theory with reinforcement learning by using a network
    that explicitly outputs PID gains.
    """
    def __init__(self, state_dim, action_dim, action_high, hidden_dims=[16, 16],
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005, 
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, device="cuda"):
        """
        Initialize CIRL agent.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            action_high (float): Maximum action value
            hidden_dims (list): Hidden layer dimensions for networks
            lr_actor (float): Learning rate for actor
            lr_critic (float): Learning rate for critic
            gamma (float): Discount factor
            tau (float): Soft update coefficient
            policy_noise (float): Noise added to target policy
            noise_clip (float): Clipping value for policy noise
            policy_freq (int): Frequency of policy updates
            device (str): Device for torch tensors
        """
        self.action_dim = action_dim
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.device = device
        
        # Initialize actor network with CIRL structure
        self.actor = CIRLNetwork(state_dim, n_fc1=hidden_dims[0], n_fc2=hidden_dims[1], 
                                 output_sz=action_dim, pid_mode=True).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Initialize critic networks (using standard TwinQNetwork)
        critic_hidden_dims = [128, 128]  # Larger networks for critics
        self.critic = TwinQNetwork(state_dim, action_dim, critic_hidden_dims).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.total_it = 0  # Total training iterations
        self.hidden_dims = hidden_dims  # Store for identification
    
    def select_action(self, state, noise=0.1):
        """
        Select an action given the current state.
        
        Args:
            state: Current state
            noise (float): Exploration noise scale
            
        Returns:
            numpy.ndarray: Selected action
        """
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        else:
            state = state.to(self.device)
        
        # Get action from policy
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        
        # Add exploration noise
        if noise > 0:
            action = action + np.random.normal(0, noise, size=self.action_dim)
            action = np.clip(action, -1, 1)
        
        return action
    
    def train(self, replay_buffer, batch_size=256):
        """
        Update the networks using a batch of transitions.
        
        Args:
            replay_buffer: Replay buffer containing transitions
            batch_size (int): Batch size for updates
            
        Returns:
            tuple: (actor_loss, critic_loss)
        """
        self.total_it += 1
        
        # Sample a batch of transitions from replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # ---------------------------- Update Critic ---------------------------- #
        # Get next action from target policy with noise for smoothing
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
        
        # Calculate target Q-values
        target_q1, target_q2 = self.critic_target(next_state, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward + (1 - done) * self.gamma * target_q
        
        # Calculate current Q-values
        current_q1, current_q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---------------------------- Update Actor ---------------------------- #
        actor_loss = None
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update_targets()
            
            actor_loss = actor_loss.item()
        
        return actor_loss, critic_loss.item()
    
    def _soft_update_targets(self):
        """
        Soft update target networks using polyak averaging.
        """
        # Update critic target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Update actor target
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filename):
        """
        Save model parameters to file.
        
        Args:
            filename (str): Path to save file
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_it': self.total_it
        }, filename)
        
    def load(self, filename):
        """
        Load model parameters from file.
        
        Args:
            filename (str): Path to load file
        """
        checkpoint = torch.load(filename)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_it = checkpoint['total_it']


def create_agent(agent_type, state_dim, action_dim, action_high, device):
    """
    Create an RL agent based on the specified type.
    
    Args:
        agent_type (str): Type of agent ("td3", "sac", or "cirl")
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        action_high (float): Maximum action value
        device (str): Device for torch tensors
        
    Returns:
        object: Initialized agent
    """
    if agent_type.lower() == "td3":
        return TD3(state_dim, action_dim, action_high, device=device)
    elif agent_type.lower() == "sac":
        return SAC(state_dim, action_dim, action_high, device=device)
    elif agent_type.lower() == "cirl":
        return CIRL(state_dim, action_dim, action_high, hidden_dims=[16, 16], device=device)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")