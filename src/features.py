import numpy as np
import pandas as pd

class Features:
    def __init__(self, prices:pd.Series):
        self.prices = prices.copy()
    
    def log_returns(self) -> pd.Series:
        return np.log(self.prices/self.prices.shift(1))
    
    def sma_ratio(self, window:int) -> pd.Series:
        sma = self.prices.rolling(window).mean()
        return self.prices/sma
    
    def rsi(self, window:int, normalize:bool=True) -> pd.Series:
        delta = self.prices.diff(1)
        gain = delta.clip(lower=0)
        loss = delta.clip(upper=0)
        
        avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
        avg_loss = -loss.ewm(alpha=1/window, adjust=False).mean() #flip the sign
        
        rs = avg_gain/avg_loss
        rsi = 100 - (100/ (1+rs))
        
        if normalize:
            rsi /= 100
        
        return rsi
    
    def pipeline(self, sma_window:int=14, rsi_window:int=14, normalize_rsi:bool=True) -> pd.DataFrame:
        f_log_returns = self.log_returns()
        f_sma_ratio = self.sma_ratio(sma_window)
        f_rsi = self.rsi(rsi_window, normalize_rsi)
        features = np.array([f_log_returns, f_sma_ratio, f_rsi])
        
        col_names = ['log_returns', 'sma_ratio', 'rsi']
        df = pd.DataFrame(features.T, columns=col_names).dropna()
        return df

def build_features(prices_df:pd.DataFrame, sma_window:int, rsi_window:int) -> pd.DataFrame:
    feature_dfs = []
    for col in prices_df.columns:
        f = Features(prices_df[col])
        df_temp = f.pipeline(sma_window, rsi_window)
        df_temp = df_temp.add_prefix(f'{col}_')
        feature_dfs.append(df_temp)
    df = pd.concat(feature_dfs, axis=1)
    return df