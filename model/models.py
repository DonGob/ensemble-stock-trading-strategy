# common library
import re
import turtle
import pandas as pd
import numpy as np
import time
import gym
# RL models from stable-baselines

import matplotlib
import matplotlib.dates as mdates

from sklearn.metrics import balanced_accuracy_score
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines import GAIL, SAC
from stable_baselines import ACER
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import TD3
from stable_baselines import DDPG

from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv
from preprocessing.preprocessors import *
from config import config

# customized env
from model_selection.selection_env import ModelSelectionEnvTrain, ModelSelectionEnvTrade
from model_selection import selection_env

from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade

from pathlib import Path
import os

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

TRAINED_MODEL_DIR = Path("trained_models/optimized_trading_models")

def train_A2C(env_train, model_name, timesteps=790000, learning_rate=0.00035):
    """A2C model"""

    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0, learning_rate=learning_rate,)
    model.learn(total_timesteps=timesteps)
    end = time.time()
    print("the save name is", f"{config.TRAINED_MODEL_DIR}/{model_name}")
    model.save(Path(f"{config.TRAINED_MODEL_DIR}/{model_name}"))
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model

def train_TD3(env_train, model_name, timesteps=160000, learning_rate=0.00105):
    """TD3 model"""

    # add the noise objects for TD3
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = TD3('MlpPolicy', env_train, action_noise=action_noise, learning_rate=learning_rate)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(Path(f"{config.TRAINED_MODEL_DIR}/{model_name}"))
    print('Training time (TD3): ', (end-start)/60,' minutes')
    return model

def train_PPO(env_train, model_name, timesteps=85000, learning_rate=0.00177):
    """PPO model"""

    start = time.time()
    model = PPO2('MlpPolicy', env_train, ent_coef = 0.005, nminibatches = 8,  learning_rate=learning_rate)
    #model = PPO2('MlpPolicy', env_train, ent_coef = 0.005)

    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(Path(f"{config.TRAINED_MODEL_DIR}/{model_name}"))
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model

    
def train_selection_model(env_train, model_name, timesteps=100000, learning_rate=0.001):
    start = time.time()
    model = A2C('MlpLnLstmPolicy', env_train, verbose=0, learning_rate=learning_rate)
    model.learn(total_timesteps=timesteps, tb_log_name="selector")
    end = time.time()

    model.save(Path(f"{config.TRAINED_MODEL_DIR}/{model_name}"))
    print('Training time (Selection): ', (end - start) / 60, ' minutes')
    return model

def load_model(name, env):
    model_exists, model_path = find_model_path(name)
    print(name)
    if model_exists:
        if "A2C" in name:
            print(f"loading A2C, from path {model_path}")
            return A2C.load(model_path, env)
        elif "TD3" in name:
            print(f"loading TD3, from path {model_path}")
            return TD3.load(model_path, env)
        elif "PPO" in name:
            print(f"loading PPO, from path {model_path}")
            return PPO2.load(model_path, env)
        elif "Selector" in name:
            print(f"loading selector, from path {model_path}")
            return A2C.load(model_path, env, tensorboard_log= './selector_tb/')
        else:
            print("Error! model could not be loaded in load_model function! name does not match any of the models")
    else:
        print(f"model {name} still had to be trained")
        print(f"the path which is missing is {model_path}")
        return run_correct_train_function(name, env)

def run_correct_train_function(name, env):
    if "A2C" in name:
        return train_A2C(env, name)
    elif "TD3" in name:
        return train_TD3(env, name)
    elif "PPO" in name:
        return train_PPO(env, name)
    elif "Selector" in name:
        return train_selection_model(env, name)
    else:
        print("Error! none of the models match!! error in Run_correct_train_function")

def find_model_path(target_model):
    all_model_names = os.listdir(TRAINED_MODEL_DIR)
    for model_name in all_model_names:
        if target_model in model_name:
            path = os.path.join(TRAINED_MODEL_DIR, model_name)
            return (True, path)
    return (False, "Model does not exist!")

def DRL_prediction(df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   turbulence_threshold,
                   initial,
                   a2c,
                   ppo,
                   td3):
    ### make a prediction based on trained model###
    print("======Trading from: ", unique_trade_date[iter_num - rebalance_window], "to ", unique_trade_date[iter_num])

    ## trading env
    balance_df = pd.DataFrame(columns=['balance'])
    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)])

    env_selection = DummyVecEnv([lambda: ModelSelectionEnvTrade(env_trade, trade_data, a2c, ppo, td3, initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)]) 
    
    obs_trade = env_trade.reset()[0]    #dummy env returns the state wrapped in a list
    selection_period = selection_env.SELECTION_PERIOD
    for i in range(0, len(trade_data.index.unique()) , selection_period):
        action, _states = model.predict([obs_trade]) 
        action = action[0] #since action is returned wrapped in a list
        start_state, rewards, dones, info = env_selection.step([action]) #dummy vec env needs action wrapped in list
        info = info[0]

        if len(info) != 0:
            obs_trade = info['terminal_observation'] #dummyvecenv returns state wrapped in a list
        last_state = obs_trade
        balance_df = pd.concat([balance_df, info['balance']])
    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('results/last_state_{}_{}.csv'.format(name, iter_num), index=False)
    return last_state, balance_df

def get_yearly_sharpe():
    ###Calculate Sharpe ratio based on validation results###
    df = pd.read_csv('balance_data.csv')
    balances = df['balance']
    df['daily_return'] = balances.pct_change(1)
    sharpe = ((252 ** 0.5) * df['daily_return'].mean()) / df['daily_return'].std()
    return sharpe

def get_all_models(df, iteration, trade_dates, rebalance_window):
    print("======Model training from: ", 20090000, "to ",
            trade_dates[iteration - rebalance_window])
    
    train_data = data_split(df, start=20090000, end=trade_dates[iteration - rebalance_window])
    env_train = DummyVecEnv([lambda: StockEnvTrain(train_data, standalone=True)])

    model_a2c = load_model("A2C_190k_{}".format(iteration), env_train)
    model_ppo = load_model("PPO_85k_{}".format(iteration), env_train)
    model_td3 = load_model("TD3_160k_{}".format(iteration), env_train)
    
    env_selection_train = DummyVecEnv([lambda: ModelSelectionEnvTrain(train_data, model_a2c, model_ppo, model_td3)]) 
    selection_model = load_model("Selector_{}".format(iteration), env_selection_train)

    return model_a2c, model_ppo, model_td3, selection_model

def get_turbulence(insample_turbulence, insample_turbulence_threshold, iteration, trade_dates, df, rebalance_window):
    end_date_index = df.index[df["datadate"] == trade_dates[iteration - rebalance_window - rebalance_window]].to_list()[-1]
    start_date_index = end_date_index - rebalance_window*30 + 1

    historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
    #historical_turbulence = df[(df.datadate<unique_trade_date[i - rebalance_window - validation_window]) & (df.datadate>=(unique_trade_date[i - rebalance_window - validation_window - 63]))]


    historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])

    historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

    if historical_turbulence_mean > insample_turbulence_threshold:
        # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
        # then we assume that the current market is volatile,
        # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
        # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
        turbulence_threshold = insample_turbulence_threshold
    else:
        # if the mean of the historical data is less than the 90% quantile of insample turbulence data
        # then we tune up the turbulence_threshold, meaning we lower the risk
        turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)

    print(f'turb thresh is: {turbulence_threshold}')
    return turbulence_threshold


def check_if_initial(iteration, rebalance_window):
    if iteration - rebalance_window*2 == 0:
        # inital state
        initial = True
    else:
        # previous state
        initial = False
    return initial

def plot_balance(df):
    dates = pd.to_datetime(df['index'], format='%Y%m%d')
    fig, ax = plt.subplots()
    data = df['balance']
    ax.plot(dates, data)

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(True)
    ax.set_ylabel(r'balance [\$]')
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    plt.savefig("test_balance")

def run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window) -> None:
    """Ensemble Strategy that combines PPO, A2C and TD3"""
    print("============Start Ensemble Strategy============")
    start = time.time()
    last_state_ensemble = []
    balance_df = pd.DataFrame(columns=['balance'])

    
    insample_turbulence = df[(df.datadate<20151000) & (df.datadate>=20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)
    
    for i in range(rebalance_window*2, len(unique_trade_date), rebalance_window):

        turbulence_threshold = get_turbulence(insample_turbulence, insample_turbulence_threshold, i, unique_trade_date, df, rebalance_window)
        model_a2c, model_ppo, model_td3, selection_model = get_all_models(df, i, unique_trade_date, rebalance_window)

        ############## Trading starts ##############\
        last_state_ensemble, balance_memory = DRL_prediction(df=df, model=selection_model, name="ensemble",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=check_if_initial(i, rebalance_window), a2c = model_a2c, ppo = model_ppo, td3= model_td3)
        balance_df = pd.concat([balance_df, balance_memory])
        ############## Trading ends ##############

    balance_df = balance_df.reset_index()
    end = time.time()

    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")

    balance_df.to_csv('balance_data.csv')

    plot_balance(balance_df)
    
    print(f'sharpe from entire strategy is {get_yearly_sharpe()}')

    data_list = []
    for i in range(63, 1200, 63):
        test_path = f'results/account_value_trade_ensemble_{i}.csv'
        test_read = pd.read_csv(test_path, index_col=0)
        data_list.append(test_read)
    final_frame = pd.concat(data_list).reset_index(drop=True)

