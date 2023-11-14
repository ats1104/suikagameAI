import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple

from tqdm import tqdm

import random
from collections import namedtuple
from typing import Tuple

import pygame
import pygame.surfarray
import pymunk

import gym
from gym.spaces import Box, Discrete
from gym.wrappers import FrameStack
from gym.envs.registration import register
# 環境を登録するよう．
import src




EPISODES = 1000

if __name__ == "__main__":

    env = gym.make("myenv-v0")
    print("学習開始")

    for e in range(1000):
        # 初期状態：画像
    
        state = env.reset()
        # シミレーションのloop
        while True:
            
            action = env.controller.update(state)
            # 行動の実行
            next_state, reward, done, info = env.step(action)

            # キャッシュに記録
            env.controller.cache(state, next_state, env.controller.action_idx, reward, done)
            # 学習．
            q, loss = env.controller.learn()

            # ログ
            env.logger.log_step(reward, loss, q)

            if done:
                break

            state = next_state

        env.logger.log_episode()

        if e % 20:
            env.logger.record(episode=e, epsilon=env.controller.exploration_rate, step=env.controller.curr_step, model=env.controller.net)






