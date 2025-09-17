import numpy as np
import pandas as pd

class SimulatePrices:
    def __init__(self, start_px:float, start_vol:float, time_steps:int, N:int=252, seed:int|None=None):
        self.start_px = start_px
        self.start_vol = start_vol
        self.time_steps = time_steps
        self.dt = 1/N
        self._rng = np.random.default_rng(seed)
    
    def _volatility(self, prev_vol:float, kappa:float, theta:float, zeta:float, z:float) -> float:
        v = max(prev_vol, 0)
        mean_revert_term = kappa*(theta - v)* self.dt
        
        indicator_func = (v >= 0)
        first_order = zeta* np.sqrt(v*self.dt)* z
        second_order = indicator_func* (zeta**2 /4)* self.dt*(z**2 - 1)
        
        vol = prev_vol + mean_revert_term + first_order + second_order
        return vol
    
    def _brownian_price(self, z_vol:float|np.ndarray, z_perp:float|np.ndarray, p:float) -> float|np.ndarray:
        return p*z_vol + np.sqrt(1-p**2)*z_perp
    
    def _price(self, prev_px:float, mu:float, vol:float, z:float) -> float:
        v = max(vol, 0)
        exp_term = (mu - 0.5*v)*self.dt + np.sqrt(v*self.dt)*z
        px = prev_px*np.exp(exp_term)
        return px
    
    def simulate(self, mu:float, kappa:float, theta:float, zeta:float, p:float) -> tuple[np.ndarray, np.ndarray]:
        S = np.empty(self.time_steps+1)
        V = np.empty(self.time_steps+1)
        S[0] = self.start_px
        V[0] = self.start_vol
        
        z_vol = self._rng.standard_normal(self.time_steps)
        z_perp = self._rng.standard_normal(self.time_steps)
        z_s = self._brownian_price(z_vol, z_perp, p)
        
        for n in range(self.time_steps):
            v = self._volatility(V[n], kappa, theta, zeta, z_vol[n])
            px = self._price(S[n], mu, v, z_s[n])
            S[n+1], V[n+1] = px, v
        
        return S, V

class PortfolioEnv:
    def __init__(self, df:pd.DataFrame, initial_balance:float=1000):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self) -> np.ndarray:
        self.balance = self.initial_balance
        self.t = 0
        return self._getstate()
    
    def _getstate(self) -> np.ndarray:
        return self.df.iloc[self.t].values.astype(np.float64)
    
    def _getreturns(self) -> np.ndarray:
        return_cols = self.df.filter(like='log_returns')
        return_vals = return_cols.iloc[self.t+1].values #next day return
        return return_vals
    
    def step(self, weights:np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        norm_weights = weights/ (weights.sum() + 1e-8) #Enforces that weights sum to 1
        
        return_vals = self._getreturns()
        reward = norm_weights@return_vals
        self.balance *= np.exp(reward)
        
        self.t += 1
        done = self.t >= (len(self.df) - 2)
        return self._getstate(), reward, done, {'balance': self.balance}