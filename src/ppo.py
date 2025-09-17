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
        choice = torch.min(ratio*advantage, clip_func) 
        return -choice.mean() #Optimizer set to minimize so we flip the sign, alternatively you can set it to maximize
    
    def value_loss_func(self, returns:torch.Tensor, values:torch.Tensor) -> torch.Tensor:
        diff = returns-values
        mse = diff.pow(2)
        return mse.mean()
    
    def total_loss(self, policy_loss:torch.Tensor, value_loss:torch.Tensor, value_coef:float=0.5) -> torch.Tensor:
        return policy_loss + value_coef*value_loss

class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []
    
    def add(self, state:np.ndarray, action:torch.Tensor, log_prob:torch.Tensor, reward:float, done:bool, value:float):
        self.states.append(state)
        self.actions.append(action.numpy())
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        
    def clear(self):
        self.__init__()

def compute_gae(rewards:list[float], values:list[float], dones:list[bool], gamma:float, lamda:float) -> tuple[torch.Tensor, torch.Tensor]:
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    values = values + [0] #for looping index
    
    for t in reversed(range(T)):
        indicator_func = (1 - dones[t]) #if terminal, done=True, but we want to multiply by 0, so 1-True = 0
        v_obs = rewards[t] + gamma* values[t+1]* indicator_func
        delta = v_obs - values[t]
        gae = delta + (gamma* lamda* gae)* indicator_func
        advantages[t] = gae
    
    returns = advantages + values[:-1]
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    
    return advantages_tensor, returns_tensor 

def train_ppo(env, agent:PPOAgent, loss_func:LossFunctions, value_loss_coef:float=0.5, epochs=10, timesteps=256, update_steps=32):
    for ep in range(epochs):
        buffer = RolloutBuffer()
        state = env.reset()
        ep_reward = 0
        
        # Go through the data
        for _ in range(timesteps):
            weights, log_prob, action, dist = agent.choose_action(state)
            
            state_tensor = torch.tensor(state, dtype=torch.float32)
            value = agent.value(state_tensor).item() #No unsqueeze as we are getting a scalar anyways
            
            next_state, reward, done, info = env.step(weights)
            buffer.add(state, action, log_prob, reward, done, value)
            
            state = next_state
            ep_reward += reward #reward is log return, so we can sum
            if done:
                break
        
        # Compute GAE
        advantages, returns = compute_gae(buffer.rewards, buffer.values, buffer.dones, agent.gamma, agent.lamda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) #Improve stability and convergence
        
        # Convert to tensors
        states = torch.tensor(buffer.states, dtype=torch.float32)
        actions = torch.tensor(buffer.actions, dtype=torch.float32)
        old_log_probs = torch.stack(buffer.log_probs).detach()
        
        # Update PPO
        for _ in range(update_steps):
            mu, std = agent.policy(states)
            dist = torch.distributions.Normal(mu, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            
            values = agent.value(states).squeeze()
            
            policy_loss = loss_func.policy_loss_func(log_probs, old_log_probs, advantages, tol=agent.clip_eps)
            value_loss = loss_func.value_loss_func(returns, values)
            loss = loss_func.total_loss(policy_loss, value_loss, value_loss_coef)
            
            agent.optimizer.zero_grad() #clear old gradients
            loss.backward() #compute new gradients
            agent.optimizer.step() #update weights
        
        ep_reward = np.exp(ep_reward) - 1 #Convert back to simple return
        print(f"Epoch {ep+1}: Reward={ep_reward:.4%}, Balance={info['balance']:.6f}")