import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#Actor
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, n_assets, hidden_dim:int=32):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, n_assets)
        self.log_std_head = nn.Linear(hidden_dim, n_assets)
    
    def forward(self, x):
        x = self.shared(x)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(-5, 3)  #Ensures a reasonable range
        std = torch.exp(log_std)
        return mu, std

#Critic
class ValueNet(nn.Module):
    def __init__(self, state_dim:int, hidden_dim:int=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x:torch.Tensor):
        return self.fc(x)

class PPOAgent:
    def __init__(self, state_dim:int, n_assets:int, lr:float=3e-4, gamma:float=0.99, clip_eps:float=0.2, lamda:float=0.95):
        self.policy = GaussianPolicy(state_dim, n_assets)
        self.value = ValueNet(state_dim)
        
        #Optimizer
        params = list(self.policy.parameters()) + list(self.value.parameters())
        self.optimizer = optim.Adam(params, lr=lr)
        
        #PPO
        self.clip_eps = clip_eps
        
        #GAE
        self.gamma = gamma
        self.lamda = lamda

    def choose_action(self, state:np.ndarray) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) #Pytorch expect [batchsize, features]
        
        #Sample weights
        mu, std = self.policy(state)
        dist = torch.distributions.Normal(mu, std)
        z = dist.sample()
        
        #Get Variables
        weights = torch.softmax(z, dim=-1) #dim=-1 selects the last dimension, features in our case. Size is already based on total assets.
        weights = weights.squeeze(0).detach().numpy() #Get features, detach tensor from network, convert to numpy for PortfolioEnv
        
        log_prob = dist.log_prob(z).sum(dim=-1)
        log_prob = log_prob.detach()
        
        z = z.detach()
        
        return weights, log_prob, z, dist

class LossFunctions:
    def __init__(self):
        pass
    
    def policy_loss_func(self, log_probs:torch.Tensor, old_log_probs:torch.Tensor, advantage:torch.Tensor, tol:float) -> torch.Tensor:
        ratio = torch.exp(log_probs - old_log_probs)
        clip_func = torch.clamp(ratio, 1-tol, 1+tol)*advantage
        choice = torch.min(ratio*advantage, clip_func) #Benefit yield from difference in action probabilities, if the beneit is too drastic, clip it to penalize
        return -choice.mean() #Optimizer set to minimize so we flip the sign, alternatively you can set it to maximize
    
    def value_loss_func(self, returns:torch.Tensor, values:torch.Tensor) -> torch.Tensor:
        diff = returns-values
        mse = diff.pow(2)
        return mse.mean()
    
    def total_loss(self, policy_loss:torch.Tensor, value_loss:torch.Tensor, value_coef:float=0.5) -> torch.Tensor:
        return policy_loss + value_coef*value_loss