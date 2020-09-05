import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Value, Lock, Process
from torch.optim import Adam
import stockstats
import os
import time
import copy

import sys
sys.path.append('..')
from Env import Env
from optimizer import GlobalAdam

device = torch.device('cpu')
os.environ["OMP_NUM_THREADS"] = "1"

random_seed = 666
latent_num = 16
cnn_chanel_num = 16
stat_dim = 7
img_shape = (64, 128)
save_every = 100
minibatch = 1024
        
        
class S2(nn.Module):
    
    def __init__(self, latent_num, cnn_chanel_num, stat_dim):
        super(S2, self).__init__()
        
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
    

    def forward(self, img, stat):
        encoded_img = self.encode_img(img)
        encoded_stat = self.encode_stat(stat)
        catted = torch.cat((encoded_img, encoded_stat), 1)
        hid = self.pi(catted)
        probs = self.actor(hid)
        return probs

    def act(self, img, stat):
        img = torch.from_numpy(img).float().unsqueeze(0)
        stat = torch.from_numpy(stat).float()
        encoded_img = self.encode_img(img.unsqueeze(0).to(device))
        encoded_stat = self.encode_stat(stat.unsqueeze(0).to(device))
        catted = torch.cat([encoded_img, encoded_stat], dim=1)
        hid = self.pi(catted).squeeze()
        p = Categorical(self.actor(hid))
        action = p.sample().squeeze()
        return action.item()


    
class Worker_Generator(Process):
    def __init__(self, no, data, pair_queue):
        super(Worker_Generator, self).__init__()
        self.no = no
        self.pair_queue = pair_queue

        self.data = data
        

        self.img_shape = img_shape
        self.threshold = 1
        self.k = 2

    def grey_scale_img(self, pos):
        end = pos + 1
        C = self.img_shape[1]
        sta = end - C
        R = self.img_shape[0]
        # Ohlc
        op = list(self.stats[sta:end]['open'])
        clz = list(self.stats[sta:end]['close'])
        high = list(self.stats[sta:end]['high'])
        low = list(self.stats[sta:end]['low'])
        minn = min(low)
        base = max(high)-minn
        # compose image matrix
        a = np.zeros([R,C])
        for i in range(C):
            tmph = R - int(((high[i]-minn)*(R-1)/base)+0.5) - 1
            tmpl = R - int(((low[i]-minn)*(R-1)/base)+0.5)
            a[tmph:tmpl, i] = 1
        return np.expand_dims(a, 0)

    def get_stats(self, pos):
        return np.array((self.wr[pos], self.kdjk[pos], self.kdjd[pos], self.kdjj[pos], self.rsi[pos], self.adx[pos], self.dma[pos]))

    def run(self):
        print(f"Process_{self.no}_started")

        self.stats = stockstats.StockDataFrame(self.data)
        def normalize(x):
            x = x.copy()
            return (x - np.mean(x)) / (np.std(x) + 1e-5)
        self.wr = normalize(self.stats['wr_14'])
        self.kdjk = normalize(self.stats['kdjk_20'])
        self.kdjd = normalize(self.stats['kdjd_20'])
        self.kdjj = normalize(self.stats['kdjj_20'])
        self.rsi = normalize(self.stats['rsi_18'])
        self.adx = normalize(self.stats['adx'])
        self.dma = normalize(self.stats['dma'])
        self.len = len(self.stats)

        def shirnk(x):
            if abs(x) > self.threshold:
                return x
            else:
                return 0
        sign2cate = lambda x: 1 - x

        while True:
            pos = np.random.randint(self.img_shape[1], self.len - 1 - self.k)
            img = self.grey_scale_img(pos)
            delta = self.stats.iloc[pos+self.k]['close'] - self.stats.iloc[pos]['close']
            t = shirnk(delta)
            sign = np.sign(t)
            cate = sign2cate(sign)

            stat = self.get_stats(pos)

            self.pair_queue.put((img, stat, cate))


class Validate(Process):
    def __init__(self,  shared_model, optimizer_lock, validate_queue):
        super(Validate, self).__init__()
        self.shared_model = shared_model
        self.optimizer_lock = optimizer_lock
        self.validate_queue = validate_queue
    
    def run(self):
        print("Process_Validate_started")
        self.local_model = copy.deepcopy(self.shared_model).to(device)
        data = pd.read_csv(f"../data/M8888.XDCE_5m_test.csv")
        self.env = Env(data, 990, img_shape, 2, 1000000, 0.1)
        while True:
            with self.optimizer_lock:
                self.local_model.load_state_dict(self.shared_model.state_dict())
            img, stat = self.env.reset()
            done = False
            while not done:
                action = self.local_model.act(img, stat)
                (img, stat), done = self.env.step(action, 0.6)
            self.validate_queue.put((self.env.reward_list.sum(), self.env.money/self.env.init_money, self.env.is_win.mean()))
    
def train():
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    writer = SummaryWriter()
    s2 = S2(latent_num, cnn_chanel_num, stat_dim).to(device).share_memory()

    writer.add_graph(s2, (torch.zeros([1, 1, img_shape[0], img_shape[1]]).to(device), torch.zeros([1, stat_dim]).to(device)))

    optim = GlobalAdam([
                {'params': s2.encode_img.parameters()},
                {'params': s2.encode_stat.parameters()},
                {'params': s2.pi.parameters()},
                {'params': s2.actor.parameters()}
            ], lr=1e-2, weight_decay=0.01)

    if os.path.exists('S2_state_dict.pt'):
        s2.load_state_dict(torch.load('S2_state_dict.pt'))
        optim.load_state_dict(torch.load('S2_Optim_state_dict.pt'))
    
    pair_queue = Queue(10000)
    validate_queue = Queue()
    optimizer_lock = Lock()

    process = []
    data_list = ['A8888.XDCE', 'AL8888.XSGE', 'AU8888.XSGE', 'C8888.XDCE', 'M8888.XDCE', 'RU8888.XSGE', 'SR8888.XZCE']
    for no in range(mp.cpu_count()-1):
        data = pd.read_csv(f"../data/{data_list[no]}_5m.csv")
        worker = Worker_Generator(no, data, pair_queue)
        worker.start()
        process.append(worker)
    validater = Validate(s2, optimizer_lock, validate_queue)
    validater.start()

    epochs = 0
    while True:
        imgs = []
        stats = []
        cates = []
        seen = 0
        while seen < minibatch:
            img, stat, cate = pair_queue.get()
            imgs.append(img)
            stats.append(stat)
            cates.append(cate)
            seen += 1

        imgs = torch.tensor(imgs).float().to(device)
        stats = torch.tensor(stats).float().to(device)
        g_t = torch.tensor(cates).long().to(device)
        pred = s2(imgs, stats)
        loss = F.cross_entropy(pred, g_t)
        accr = (pred.argmax(1) == g_t).sum().item() / minibatch

        with optimizer_lock:
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        if not validate_queue.empty():
            val_reward, val_money, val_win = validate_queue.get()
            writer.add_scalar('Validate/reward', val_reward, epochs)
            writer.add_scalar('Validate/money', val_money, epochs)
            writer.add_scalar('Validate/win_rate', val_win, epochs)

        writer.add_scalar('Train/Loss', loss.item(), epochs)
        writer.add_scalar('Train/Accr', accr, epochs)
        epochs += 1

        if epochs % save_every == 0:
            torch.save(s2.state_dict(), 'S2_state_dict.pt')
            torch.save(optim.state_dict(), 'S2_Optim_state_dict.pt')


    for worker in process: worker.join()

if __name__ == '__main__':
    train()