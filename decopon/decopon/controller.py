from abc import ABC, abstractmethod
from typing import Tuple

import pygame

import numpy as np
import random
import torch
import torch.nn as nn
import copy
from collections import deque

import time, datetime
import matplotlib.pyplot as plt

class Controller(ABC):
    @abstractmethod
    def update(self) -> Tuple[bool, bool, bool]:
        return True, True, True

class Human(Controller):
    def __init__(self) -> None:
        super().__init__()

    def update(self) -> Tuple[bool, bool, bool]:
        pressedKeys = pygame.key.get_pressed()
        return pressedKeys[pygame.K_LEFT], pressedKeys[pygame.K_RIGHT], pressedKeys[pygame.K_SPACE]

class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        self.target = copy.deepcopy(self.online)

    def forward(self, input, model):
   
        
        if model == "online":
            return self.online(input)

        elif model == "target":
            return self.target(input)

class AIPlayer(Controller):

    def __init__(self, state_dim, action_dim, save_dir, test) -> None:
        super().__init__()
        self.action_dict = {0:[True, False, False], 1:[False, True, False], 2:[False, False, True]}
        # init
        self.action_idx = 0

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.burnin =  1000 # 経験を訓練させるために最低限必要なステップ数
        self.learn_every = 5  # Q_onlineを更新するタイミングを示すステップ数
        self.sync_every = 100  # Q_target & Q_onlineを同期させるタイミングを示すステップ数
        self.test = test # bool ：学習かテストか
        self.use_cuda = torch.cuda.is_available()

        self.net = DQN(self.state_dim, self.action_dim).float()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        # 学習したモデルを試す場合
        if self.test:
            print("load train model")
            # 本来ならばベストなモデルをロードすることになる．
            self.net.load_state_dict(torch.load('./result/model_ep50.chkpt'))
            # self.net.load_state_dict(torch.load("./dqn_models/my_agent_v0.pth"))
        
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9995

        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 100

        # キャッシュ古いものは捨ててく．
        self.memory = deque(maxlen=10000)
        self.batch_size = 32

        # 割引率
        self.gamma = 0.9


    def act(self, state):

        # 探索
        # state.shape (h, w, c)
        if not self.test:

            if np.random.rand() < self.exploration_rate:
                self.action_idx = np.random.randint(self.action_dim)
            # 活用
            else:
                state = state.__array__()
                if self.use_cuda:
                    state = torch.tensor(state, dtype=torch.float64)
                else:
                    state = torch.tensor(state, dtype=torch.float64)
                state = state.float()
                # 次元の入れ替え
                state = state.permute(2, 0, 1)
                
                state = state.unsqueeze(0)
                action_values = self.net(state, model="online")
                self.action_idx = torch.argmax(action_values, axis=1).item()
            

            # 10 episode 分 は 更新 しない．
            if self.curr_step >= self.burnin:
                self.exploration_rate *= self.exploration_rate_decay
                self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

            # # ステップを+1
            self.curr_step += 1

            return self.action_idx

        # test時
        else:

            if np.random.rand() < 0.0:
                print("Random Action")
                self.action_idx = np.random.randint(self.action_dim)
                print(self.action_idx)
            else:
                print("AI Action")
                state = state.__array__()
                if self.use_cuda:
                    state = torch.tensor(state, dtype=torch.float64)
                else:
                    state = torch.tensor(state, dtype=torch.float64)
                state = state.float()
                state = state.permute(2, 0, 1)

                state = state.unsqueeze(0)
                action_values = self.net(state, model="online")
                self.action_idx = torch.argmax(action_values, axis=1).item()
                print(self.action_idx)

            self.curr_step += 1

            return self.action_idx

    def update(self, state) -> Tuple[bool, bool, bool]:
        self.action_idx = self.act(state)
        return self.action_dict[self.action_idx]

    def cache(self, state, next_state, action, reward, done):
        """経験をメモリに追加．"""
        # next_state, reward, done等はenvのstepで定義

        # state = state.__array__()
        # next_state = next_state.__array__()
      

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])
        state = state.permute(2, 0, 1)
        next_state = next_state.permute(2, 0, 1)

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        メモリから経験のバッチを取得
        """
        # batch_size分のデータを取得．
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def save(self):
        save_path = self.save_dir + f"/model_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"Model saved to {save_path} at step {self.curr_step}")


    def td_estimate(self, state, action):
        state = state.float()
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state = next_state.float()
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):

        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            # self.save()
            pass

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # メモリからサンプリング

        state, next_state, action, reward, done = self.recall()

        # TD Estimateの取得
        td_est = self.td_estimate(state, action)

        # TD Targetの取得
        td_tgt = self.td_target(reward, next_state, done)

        # 損失をQ_onlineに逆伝播させる
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


class MetricLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.save_log = save_dir + "/log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
            
        self.ep_rewards_plot = save_dir + "/reward_plot.jpg"
        self.ep_lengths_plot = save_dir + "/length_plot.jpg"
        self.ep_avg_losses_plot = save_dir + "/loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir + "/q_plot.jpg"

        # 指標の履歴
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # reacord()が呼び出されるたびに追加される移動平均
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # 現在のエピソードの指標
        self.init_episode()

        # 時間を記録
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "エピソード終了時の記録"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step, model):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()

        save_path = self.save_dir + f"/model_ep{episode}.chkpt"
        torch.save(model.state_dict(), save_path)

        print(f"Model saved to {save_path}")