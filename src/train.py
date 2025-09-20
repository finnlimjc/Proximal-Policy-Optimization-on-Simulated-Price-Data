import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

class TrainPPO:
    def __init__(self, env, agent, loss_func):
        self.env = env
        self.agent = agent
        self.loss_func = loss_func
    
    def compute_gae(self, rewards:list[float], values:list[float], dones:list[bool], gamma:float, lamda:float) -> tuple[torch.Tensor, torch.Tensor]:
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
        
        returns = advantages + values[:-1] #A = R - V
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        
        return (advantages_tensor, returns_tensor)
    
    def _explore_timesteps(self, buffer:RolloutBuffer, timesteps:int) -> tuple[list, dict, float]:
        state = self.env.reset()
        ep_reward = 0
        
        for _ in range(timesteps):
            weights, log_prob, action, dist = self.agent.choose_action(state)
            
            state_tensor = torch.tensor(state, dtype=torch.float32)
            value = self.agent.value(state_tensor).item() #No unsqueeze as we are getting a scalar anyways
            
            next_state, reward, done, info = self.env.step(weights)
            buffer.add(state, action, log_prob, reward, done, value)
            
            state = next_state
            ep_reward += reward #reward is log return, so we can sum
            if done:
                break
        
        return (buffer, info, ep_reward)
    
    def _gradient_logger(self, agent_policy) -> float:
        total_norm = 0
        for p in agent_policy.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item()**2
        return np.sqrt(total_norm)
    
    def _update(self, value_loss_coef:float, entropy_coef:float, update_steps:int, 
                advantages:torch.Tensor, returns:torch.Tensor, states:torch.Tensor, 
                actions:torch.Tensor, old_log_probs:torch.Tensor, log_grads:bool=False) -> dict:
        
        grad_norms = []
        for _ in range(update_steps):
            mu, std = self.agent.policy(states)
            dist = torch.distributions.Normal(mu, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            
            values = self.agent.value(states).squeeze()
            entropy = dist.entropy().mean()
            
            policy_loss = self.loss_func.policy_loss_func(log_probs, old_log_probs, advantages, tol=self.agent.clip_eps)
            value_loss = self.loss_func.value_loss_func(returns, values)
            loss = self.loss_func.total_loss(policy_loss, value_loss, value_loss_coef) - entropy_coef*entropy
            
            self.agent.optimizer.zero_grad() #clear old gradients
            loss.backward() #compute new gradients
            if log_grads:
                total_norm = self._gradient_logger(self.agent.policy)
                grad_norms.append(total_norm)
            
            self.agent.optimizer.step() #update weights
        
        log_values = { 
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': loss.item(),
            'entropy': entropy.item()
        }
        if log_grads:
            log_values['avg_grad_norm'] = sum(grad_norms)/len(grad_norms)
        
        return log_values
    
    def train_ppo(self, value_loss_coef:float=0.5, entropy_coef:float=0.01, epochs=10, timesteps=256, update_steps=32, 
                  silent:bool=False, log_grads:bool=False) -> list[dict]:
        logger = []
        for ep in range(1, epochs+1):
            buffer = RolloutBuffer()
            
            buffer, info, ep_reward = self._explore_timesteps(buffer, timesteps)
            
            # Compute GAE
            advantages, returns = self.compute_gae(buffer.rewards, buffer.values, buffer.dones, self.agent.gamma, self.agent.lamda)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) #Improve stability and convergence
            
            # Convert to tensors
            states = torch.tensor(np.array(buffer.states), dtype=torch.float32) #Convert to numpy array for efficiency
            actions = torch.tensor(np.array(buffer.actions), dtype=torch.float32)
            old_log_probs = torch.stack(buffer.log_probs).detach()
            
            ep_log = self._update(value_loss_coef, entropy_coef, update_steps, advantages, returns, states, actions, old_log_probs, log_grads=log_grads)
            ep_reward = np.exp(ep_reward) - 1 #Convert back to simple return
            
            if not silent:
                print(f"Epoch {ep}: Reward={ep_reward:.4%}, Balance={info['balance']:.6f}")
            
            ep_log['epoch'] = ep
            ep_log['reward'] = ep_reward
            logger.append(ep_log)
        
        return logger