import os
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import TD3
from stable_baselines.common.vec_env import DummyVecEnv
import optuna
from model_selection.selection_env import ModelSelectionEnvTrain, ModelSelectionEnvTrade
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_trade import StockEnvTrade
from env import EnvMultipleStock_train

from preprocessing.preprocessors import *
import os
import time
from config.config import *

from pathlib import Path
import os
import pandas as pd
from pathlib import Path
VALIDATION_RANGE = 63

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def get_all_data():
    data = pd.read_csv('done_data.csv', index_col=0)
    unique_trade_date = data[(data.datadate > 20151001)&(data.datadate <= 20200707)].datadate.unique()
    train_data = data_split(data, start=20090000, end=unique_trade_date[0])
    env_train = DummyVecEnv([lambda: StockEnvTrain(train_data)])
    trade_data = data_split(data, start=unique_trade_date[0], end=unique_trade_date[63])
    stockdim = EnvMultipleStock_train.STOCK_DIM
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                    turbulence_threshold=120,
                                                    initial=True,
                                                    previous_state=[],
                                                    model_name="name",
                                                    iteration=63)])
    return data, unique_trade_date, train_data, env_train, trade_data,stockdim, env_trade

def load_trading_models(env, path):
    for name in os.listdir(path):
        if "A2C" in name:
            print("loading A2C")
            a2c = A2C.load(path.joinpath(name), env)
        elif "TD3" in name:
            print("loading TD3")
            td3 = TD3.load(path.joinpath(name), env)
        elif "PPO" in name:
            print("loading PPO")
            ppo = PPO2.load(path.joinpath(name), env)
    return a2c, ppo, td3

def train_selector(selector_env_train, model_name, timesteps, learning_rate):
    model = A2C('MlpLnLstmPolicy', selector_env_train, verbose=0, learning_rate=learning_rate)
    model.learn(total_timesteps=timesteps)

    model.save(Path(f"{config.TRAINED_MODEL_DIR}/{model_name}"))
    return model

def get_envs(selection_period, stride, env_train, env_trade, train_data, trade_data):
    path = Path("trained_models/SelectorTuning")
    a2c, ppo, td3 = load_trading_models(env_train, path)
    selection_env_train = DummyVecEnv([lambda: ModelSelectionEnvTrain(train_data, a2c, ppo, td3, selection_period=selection_period, stride=stride)])
    selection_env_trade = DummyVecEnv([lambda: ModelSelectionEnvTrade(env_trade, trade_data, a2c, ppo, td3, model_name="selector_tuning", selection_period=selection_period)])
    return selection_env_train, selection_env_trade
    

def objectiveSelector(trial):
    data, unique_trade_date, train_data, env_train, trade_data,stockdim, env_trade = get_all_data()
    lr_trial = trial.suggest_float('learning_rate', 0.0001, 0.002)
    timesteps_trial = 100000
    selection_period_trial = trial.suggest_int("selection_period", 1, 51, 5)
    stride_trial = trial.suggest_int("stride", 1, selection_period_trial)
    print(f"selection period: {selection_period_trial}, stride: {stride_trial}, lr: {lr_trial}, timesteps: {timesteps_trial}")
    selection_env_train, selection_env_trade = get_envs(selection_period_trial, stride_trial, env_train, env_trade, train_data, trade_data)
    model = train_selector(selection_env_train, f"Selector_LR{lr_trial}_TS{timesteps_trial}", timesteps=timesteps_trial, learning_rate=lr_trial)
    print("============================TRAINING DONE!!!!===============================")
    total_reward = 0      
    asset_memory = []
    done = True
    obs = selection_env_trade.reset()[0]
    i = 0
    while done:
        i += 1
        action, state = model.predict([obs])
        action = action[0]
        obs, reward, done, info = selection_env_trade.step([action])
        obs = obs[0]
        total_reward += reward
        average_reward = total_reward / i*selection_period_trial
  
        trial.report(average_reward, i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return average_reward

def calc_sharpe(asset_memory):
        df_total_value = pd.DataFrame(asset_memory)
        df_total_value.columns = ['account_value']
        df_total_value['daily_return']=df_total_value.pct_change(1)

        sharpe = (VALIDATION_RANGE**0.5)*df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
        return sharpe

def main():
    
    list_best_hyperparams = []

    print("======Selector=======")
    studySelector = optuna.create_study(direction='maximize', pruner=optuna.pruners.HyperbandPruner())
    studySelector.optimize(objectiveSelector, n_trials=30)

    list_best_hyperparams.append(studySelector.best_params)
    best_params = pd.DataFrame(list_best_hyperparams)
    best_params.to_csv("best_params_selector.csv")

if __name__ == "__main__":
    main()