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
init_money = 1000000
total_step = 1000
start_pos = 40500

charge = 2
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
    
def run():
    data = pd.read_csv(f"../data/M8888.XDCE_5m_test.csv")
    ac = AC(latent_num, cnn_chanel_num, stat_dim)
    ac.load_state_dict(torch.load('../stage3/S3_state_dict.pt'))
    env = Env(data, total_step, img_shape, charge, init_money, threshold)
    env.start_pos = start_pos
    img, stat = env.reset()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    done = False
    actions = []
    fs = []
    f_thres = []
    moneys = []
    closes = []
    while not done:
        action, f = ac.act(img, stat)
        actions.append(action)
        fs.append(f)
        (img, stat), done = env.step(action, f)
        f_thres.append(env.f)
        moneys.append(env.money)
        closes.append(env.stats.iloc[env.cur_pos-1]['close'])
    rewards = env.reward_list

    temp = []
    for i in range(1000):
        temp.append(moneys[i])
    return temp
        
if __name__ == '__main__':    
    x = []
    for i in range(5,56,5):
        threshold = i/100
        profit = run()
        df = pd.DataFrame(np.array(profit)/init_money-1, columns=['profit'])
        df.to_csv(f"threshold/threshold={threshold}.csv")