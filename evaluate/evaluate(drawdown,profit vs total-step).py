import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Value, Lock, Process
import plotly.graph_objects as go
import plotly.express as px
import os
import time
import copy

import sys
sys.path.append('..')
from Env import Env
device = torch.device('cpu')
os.environ["OMP_NUM_THREADS"] = "1"


random_seed = 666
max_ep = 1000000
latent_num = 16
cnn_chanel_num = 16
img_shape = (64, 128)
stat_dim = 7
charge = 2
init_money = 1000000
threshold = 0.1
        
        
class AC(nn.Module):
    
    def __init__(self, latent_num, cnn_chanel_num, stat_dim):
        super(AC, self).__init__()
        
        # Encode
        self.encode_img = nn.Sequential(
            nn.Conv2d(1, cnn_chanel_num, 4, stride=2), nn.ReLU(), nn.MaxPool2d(3, stride=2),
            nn.Conv2d(cnn_chanel_num, 2*cnn_chanel_num, 4, stride=2), nn.ReLU(), nn.MaxPool2d(3, stride=2),
            nn.Conv2d(2*cnn_chanel_num, cnn_chanel_num, 1, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(192, latent_num), nn.ReLU())
        
        self.encode_stat = nn.Sequential(
            nn.Linear(stat_dim, latent_num), nn.ReLU(),
            nn.Linear(latent_num, latent_num), nn.ReLU())
        
        # Actor
        self.pi = nn.Sequential(
            nn.Linear(latent_num*2, latent_num), nn.ReLU(),
            nn.Linear(latent_num, latent_num//2), nn.ReLU())
        self.actor = nn.Sequential(
            nn.Linear(latent_num//2, 3), nn.Softmax(dim=-1))
        self.f = nn.Sequential(
            nn.Linear(latent_num//2, 1), nn.Sigmoid())
        
        # Critic
        self.V = nn.Sequential(
            nn.Linear(latent_num*2, latent_num), nn.ReLU(),
            nn.Linear(latent_num, latent_num//2), nn.ReLU(),
            nn.Linear(latent_num//2, 1))
    
    # Only used for visualization
    def forward(self, img, stat):
        encoded_img = self.encode_img(img)
        encoded_stat = self.encode_stat(stat)
        catted = torch.cat([encoded_img, encoded_stat], dim=1)
        hid = self.pi(catted)
        probs = self.actor(hid)
        f = self.f(hid)
        value = self.V(catted)
        return probs, f, value
    
    def act(self, img, stat):
        img = torch.from_numpy(img).float().unsqueeze(0)
        stat = torch.from_numpy(stat).float()
        encoded_img = self.encode_img(img.unsqueeze(0).to(device))
        encoded_stat = self.encode_stat(stat.unsqueeze(0).to(device))
        catted = torch.cat([encoded_img, encoded_stat], dim=1)
        hid = self.pi(catted).squeeze()
        p = Categorical(self.actor(hid))
        f = self.f(hid)
        value = self.V(catted).squeeze()
        action = p.sample()
        return action.item(), f.item()

    def act_max(self, img, stat):
        img = torch.from_numpy(img).float().unsqueeze(0)
        stat = torch.from_numpy(stat).float()
        encoded_img = self.encode_img(img.unsqueeze(0).to(device))
        encoded_stat = self.encode_stat(stat.unsqueeze(0).to(device))
        catted = torch.cat([encoded_img, encoded_stat], dim=1)
        hid = self.pi(catted).squeeze()
        p = self.actor(hid)
        f = self.f(hid)
        action = p.argmax()
        return action.item(), f.item()

class Worker(Process):
    def __init__(self, result_queue, total_step):
        super(Worker, self).__init__()
        self.result_queue = result_queue
        self.total_step = total_step
    
    def run(self):
        data = pd.read_csv(f"../data/M8888.XDCE_5m_test.csv")
        self.ac = AC(latent_num, cnn_chanel_num, stat_dim)
        self.ac.load_state_dict(torch.load('../stage3/S3_state_dict.pt'))
        self.env = Env(data, self.total_step, img_shape, charge, init_money, threshold)

        for _ in range(8):

            img, stat = self.env.reset()
            done = False
            actions = []
            fs = []
            f_thres = []
            moneys = []
            closes = []
            while not done:
                action, f = self.ac.act(img, stat)
                actions.append(action)
                fs.append(f)
                (img, stat), done = self.env.step(action, f)
                f_thres.append(self.env.f)
                moneys.append(self.env.money)
                closes.append(self.env.stats.iloc[self.env.cur_pos-1]['close'])
            rewards = self.env.reward_list

            profit = moneys[-1] - 1

            max_drawdown = 0
            for i in range(len(moneys)-1):
                for j in range(i, len(moneys)-1):
                    drawdown = (moneys[i] - moneys[j]) / moneys[i]
                    max_drawdown = drawdown if drawdown > max_drawdown else max_drawdown

            self.result_queue.put((profit/self.env.init_money-1, max_drawdown))

if __name__ == '__main__':
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    result_queue = Queue()

    x = []
    sample_num = 64
    for t_step in range(500, 20001, 500):
        workers = []
        for _ in range(8):
            worker = Worker(result_queue, t_step)
            worker.start()
            workers.append(worker)
        
        seen = 0
        while seen < sample_num:
            if not result_queue.empty():
                profit, max_drawdown = result_queue.get()
                x.append([t_step, profit, max_drawdown])
                print(t_step, profit, max_drawdown)
                seen += 1

        for worker in workers:
            worker.join()
                


    df = pd.DataFrame(np.array(x), columns=['total_step', 'profit', 'max_drawdown'])
    df.to_csv('draw-profit-vs-step.csv')