# Proximal Policy Optimization on Simulated Price Data

## Overview

This repository provides a learning framework for Proximal Policy Optimization (PPO) applied to data simulated with the Heston stochastic volatility model. The objective is to build an intuitive understanding of how an Actor–Critic architecture learns and adapts through interaction, while also exploring how different hyperparameter choices influence training dynamics and performance. The project emphasizes both the mechanics of PPO and the practical skills of tuning and evaluation.

### Heston Stochastic Volatility Model
$$dS_t = \mu{S_t}dt + \sqrt(v_t)S_tdW^s_t$$
$$dv_t = \kappa(\theta - v_t)dt + \zeta\sqrt{v_t}dW^v_t$$
$$dW^s_tdW^v_t = \rho{dt}$$

### Discretized Heston Stochastic Volatility Model
$$S_{t+1} = S_t\exp\left[(\mu - 0.5v^+_t)\triangle{t} + \sqrt{v^+_t\triangle{t}}z\right]$$
$$v_{t+1} = v_t + \kappa(\theta - v^+_t)\triangle{t} + \zeta\sqrt{v^+_t\triangle{t}}z + \mathbb{1}_{v_t \geq 0}\frac{\zeta^2}{4}[\triangle{t}(z^2 - 1)]$$
$$\triangle{W_S} = p\triangle{W_v} + \sqrt{1-p^2}\triangle{W_\perp}$$

### Generalized Advantage Estimation
$$V_{obs}(s_t) = R_{t+1} + \gamma{V(s_{t+1})}$$
$$\delta_t = V_{obs}(s_t) - V(s_t)$$
$$\hat{A_t} = \delta_t + \gamma\lambda(\hat{A}_{t+1})$$

### PPO Loss Function
$$r_t(\theta) = \exp(log\pi_{\theta} - log\pi_{\theta_{old}})$$
$$L^{PPO}_{actor}(\theta) = E\left[min(r_t(\theta)\hat{A}_t, \quad clip(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\hat{A}_t \right]$$
$$L_{critic}(\phi) = (V_{\phi}(s_t) - \hat{R_t})^2$$

## Contents

- `src/` — source code for the environment, agent, training loops, and utilities.  
- `notebooks/` — interactive notebooks for exploration, data generation, visualization, and prototyping.  
- `requirements.txt` — Python dependencies required. 
