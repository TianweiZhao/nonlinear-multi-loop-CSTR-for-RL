import numpy as np
import torch
import torch.nn as nn

class ReplayBuffer:
    """
    Replay Buffer for storing and sampling experience transitions
    """

    def __init__(self, capacity, state_dim, action_dim, device="cuda"):
        """
        Initialize replay buffer

        Args:
            capacity (int): Maximum capacity of the buffer
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            device (str): Device to store tensors on
        """
        self.capacity = capacity
        self.device = device

        # Buffers for storing transitions
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.size = 0
        
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: State vector
            action: Action vector
            reward: Reward value
            next_state: Next state vector
            done: Done flag
        """
        # Convert to numpy arrays if needed
        if isinstance(state, torch.Tensor):
            state = state.cuda().numpy
        if isinstance(action, torch.Tensor):
            action = action.cuda().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.cuda().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cuda().numpy()
        if isinstance(done, torch.Tensor):
            done = done.cuda().numpy()
        
        # Store transition
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done
        
        # Update index and size
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def load_from_dataset(self, dataset):
        """
        Load transitions from a dataset.
        
        Args:
            dataset (dict): Dataset with keys 'states', 'actions', 'rewards', 'next_states', 'dones'
        """
        n_samples = len(dataset['states'])
        n_samples = min(n_samples, self.capacity)
        
        self.states[:n_samples] = dataset['states'][:n_samples]
        self.actions[:n_samples] = dataset['actions'][:n_samples]
        self.rewards[:n_samples] = dataset['rewards'][:n_samples].reshape(-1, 1)
        self.next_states[:n_samples] = dataset['next_states'][:n_samples]
        self.dones[:n_samples] = dataset['dones'][:n_samples].reshape(-1, 1)
        
        self.idx = n_samples % self.capacity
        self.size = n_samples
        
        print(f"Loaded {n_samples} transitions into replay buffer (capacity: {self.capacity})")
    
    def save(self, path):
        """
        Save the replay buffer to disk.
        
        Args:
            path (str): Path to save the buffer
        """
        data = {
            'states': self.states[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'next_states': self.next_states[:self.size],
            'dones': self.dones[:self.size],
            'idx': self.idx,
            'size': self.size
        }
        
        torch.save(data, path)
    
    def load(self, path):
        """
        Load the replay buffer from disk.
        
        Args:
            path (str): Path to load the buffer from
        """
        data = torch.load(path)
        
        n_samples = data['size']
        n_samples = min(n_samples, self.capacity)
        
        self.states[:n_samples] = data['states'][:n_samples]
        self.actions[:n_samples] = data['actions'][:n_samples]
        self.rewards[:n_samples] = data['rewards'][:n_samples]
        self.next_states[:n_samples] = data['next_states'][:n_samples]
        self.dones[:n_samples] = data['dones'][:n_samples]
        
        self.idx = data['idx'] % self.capacity
        self.size = n_samples
        
        print(f"Loaded {n_samples} transitions from {path}")
