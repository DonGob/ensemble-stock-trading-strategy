from more_itertools import first
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
from stable_baselines import A2C, PPO1
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_trade import StockEnvTrade
from stable_baselines.common.vec_env import DummyVecEnv
from env import EnvMultipleStock_trade, EnvMultipleStock_train

INITIAL_ACCOUNT_BALANCE = 1000000

N_MODELS = 3
SELECTION_PERIOD = 62 #days so three weeks

class ModelSelectionEnvTrain(gym.Env): #now made to evaluate single day. Better to do it monthly? If we incorporate LSTM it will be no problem anymore to do learning per day.
    def __init__(self, df, a2c, ppo, td3, day = 0, selection_period = 62, stride=21):
        self.day = day
        self.df = df
        self.action_space = spaces.Discrete(N_MODELS)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (182,))
        self.data = self.df.loc[self.day,:]
        self.terminal = False   
        self.reward = 0
        self.cost = 0
        self.rewards_memory = []
        self.a2c = a2c
        self.ppo = ppo
        self.td3 = td3
        self.env_trade = DummyVecEnv([lambda: StockEnvTrain(df)])
        self.state = self.env_trade.reset()[0]
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.run_number = 0
        self.selection_period = selection_period
        self.stride = stride

    def reset(self):
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.cost = 0
        self.terminal = False 
        self.rewards_memory = []
        #initiate state
        self.env_trade = DummyVecEnv([lambda: StockEnvTrain(self.df)])
        self.state = self.env_trade.reset()[0]       # iteration += 1 
        return self.state
    
    def return_chosen_model(self, action):
        if action == 0:
            return self.a2c
        elif action == 1:
            return self.ppo
        elif action == 2:
            return self.td3

    def calc_sharpe(self, asset_memory):
        df_total_value = pd.DataFrame(asset_memory)
        df_total_value.columns = ['account_value']
        df_total_value['daily_return']=df_total_value.pct_change(1)

        sharpe = (self.selection_period**0.5)*df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
        return sharpe

    def step(self, action):
        if self.run_number%1000 == 0:
            print(f"ATTENTION! WE ARE AT RUN {self.run_number}")
        self.run_number += 1
        self.terminal = self.day >= len(self.df.index.unique())-self.selection_period

        if self.terminal:
            self.env_trade.env_method("set_terminal_true")
            state_, temp_reward, self.terminal, info_ = self.env_trade.step([np.zeros(30)])
            return self.state, self.reward, self.terminal, {}
        else:
            model = self.return_chosen_model(action=action)
            cumulative_reward = 0
            _state = self.state
            stockdim = EnvMultipleStock_train.STOCK_DIM
            asset_memory = self.asset_memory
            for i in range(self.selection_period):
                model_action, _states = model.predict(_state)
                state_, temp_reward, self.terminal, info_ = self.env_trade.step([model_action])
                state_ = state_[0]
                temp_reward = temp_reward[0] #this this reward also gets returned wrapped in list by vecenv
                self.terminal = self.terminal[0]
                cumulative_reward += temp_reward
                if i == self.stride - 1:
                    self.state = state_
                
                end_total_asset = state_[0]+ \
                sum(np.array(state_[1:(stockdim+1)])*np.array(state_[(stockdim+1):(stockdim*2+1)]))
                asset_memory.append(end_total_asset)

            sharpe = self.calc_sharpe(asset_memory)
            self.day += self.stride
            # self.reward = cumulative_reward    
            self.reward = sharpe
            self.env_trade.env_method("move_data_forward", steps=self.stride)
            _ = self.env_trade.reset()
            return self.state, self.reward, self.terminal, {}
        
    def get_action_meanings(self):
        return {0: "A2C", 1: "PPO", 2: "TD3"}

    def render(self, mode='human',close=False):
        return self.state


class ModelSelectionEnvTrade(gym.Env): #now made to evaluate single day. Better to do it monthly? If we incorporate LSTM it will be no problem anymore to do learning per day.
    def __init__(self, env_trade, df, a2c, ppo, td3, day = 0, initial = True, previous_state = [], model_name="Ensemble", iteration='', selection_period=62):
        self.day = day
        self.df = df
        self.action_space = spaces.Discrete(N_MODELS)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (182,))
        self.data = self.df.loc[self.day,:]
        self.terminal = False   
        self.reward = 0
        self.cost = 0
        self.balance_memory = pd.DataFrame(columns=['balance'])
        self.initial = initial
        self.previous_state = previous_state
        self.a2c = a2c
        self.ppo = ppo
        self.td3 = td3
        self.env_trade = env_trade
        self.state = self.env_trade.reset()[0]       # dummyvecenv returns state wrapped in list
        self.selection_period = selection_period

    def reset(self):
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.cost = 0
        self.terminal = False 
        self.balance_memory = pd.DataFrame(columns=['balance'])
        #initiate state
        self.env_trade = DummyVecEnv([lambda: StockEnvTrain(self.df)])
        self.state = self.env_trade.reset()[0]       # dummyvecenv returns state wrapped in list
        return self.state
    
    def return_chosen_model(self, action):
        if action == 0:
            return self.a2c
        elif action == 1:
            return self.ppo
        elif action == 2:
            return self.td3

    def step(self, action): #possible to add option to run part of a window. Something for later
        self.terminal = self.day >= len(self.df.index.unique())-self.selection_period
        

        if self.terminal:
            self.env_trade.env_method("set_terminal_true")
            state_, temp_reward, self.terminal, info_ = self.env_trade.step([np.zeros(30)])
            return self.state, self.reward, self.terminal, {'balance': self.balance_memory}

        else:
            model = self.return_chosen_model(action=action)
            cumulative_reward = 0
            state_ = self.state
            stockdim = EnvMultipleStock_train.STOCK_DIM
            for i in range(self.selection_period):
                model_action, _states = model.predict(state_)
                state_, temp_reward, self.terminal, info_ = self.env_trade.step([model_action])
                state_ = state_[0]
                temp_reward = temp_reward[0]
                self.terminal = self.terminal[0]
                cumulative_reward += temp_reward
                total_asset = state_[0]+ sum(np.array(state_[1:(stockdim+1)])*np.array(state_[(stockdim+1):(stockdim*2+1)]))
                self.balance_memory.loc[self.df['datadate'].unique()[self.day]] = total_asset
                self.day += 1

            self.env_trade.env_method("set_terminal_true")
            state__, temp_reward, self.terminal, info_ = self.env_trade.step([np.zeros(30)])

            self.reward = cumulative_reward    
            self.state = state_
            return self.state, self.reward, self.terminal, {'balance': self.balance_memory}
        
    def get_action_meanings(self):
        return {0: "A2C", 1: "PPO", 2: "TD3"}

    def render(self, mode='human',close=False):
        return self.state

