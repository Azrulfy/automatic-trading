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
render = False
max_ep = 1000000
latent_num = 16
cnn_chanel_num = 16
lr = 2.5e-5
betas = (0.9, 0.999)
weight_decay = 0.001
img_shape = (64, 128)
stat_dim = 7
total_step = 1000
charge = 2
init_money = 1000000
loss_critic_coef = 0.5
loss_position_coef = 0.2
threshold = 0.1
save_every = 100

class Memory:
    def __init__(self):
        self.state_img = []
        self.state_stat = []
        self.action = []
        self.critic = []
        self.log_prob = []
        self.f = []
        self.is_done = []
    
    def clear(self):
        del self.state_img[:]
        del self.state_stat[:]
        del self.action[:]
        del self.critic[:]
        del self.log_prob[:]
        del self.f[:]
        del self.is_done[:]
        
        
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
    
    def act(self, img, stat, memory):
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
        memory.state_img.append(img)
        memory.state_stat.append(stat)
        memory.action.append(action)
        memory.critic.append(value)
        memory.f.append(f)
        memory.log_prob.append(p.log_prob(action))
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
    def __init__(self, worker_num, shared_model, ep_cnt, optimizer_lock, result_queue, gradient_queue, loss_queue):
        super(Worker, self).__init__()
        self.worker_num = worker_num
        
        self.shared_model = shared_model
        
        
        self.ep_cnt = ep_cnt
        self.optimizer_lock = optimizer_lock
        self.ep = 0
        
        self.gradient_queue = gradient_queue
        self.result_queue = result_queue
        self.loss_queue = loss_queue
        
        self.memory = Memory()
        
        self.loss_critic_coef = loss_critic_coef
        self.loss_position_coef = loss_position_coef
        self.render = render
    
    def run(self):
        print(f"Process_{self.worker_num}_started")
        self.local_model = copy.deepcopy(self.shared_model).to(device)
        self.local_optim = Adam(self.local_model.parameters(), lr=lr, betas=betas)
        data = pd.read_csv(f"../data/M8888.XDCE_5m.csv")
        self.env = Env(data, total_step, img_shape, charge, init_money, threshold)
        while True:
            with self.optimizer_lock:
                self.local_model.load_state_dict(self.shared_model.state_dict())
            
            img, stat = self.env.reset()
            done = False
            while not done:
                try:
                    action, f = self.local_model.act(img, stat, self.memory)
                except RuntimeError:
                    action = 0
                    f = 0.5
                    print(self.env.cur_pos)
                (img, stat), done = self.env.step(action, f)
                self.memory.is_done.append(done)
                if self.render:
                    env.render()
            self.update()
            with self.ep_cnt.get_lock():
                self.ep = self.ep_cnt.value
                self.result_queue.put((self.env.reward_list.sum(), self.env.money/self.env.init_money, self.env.is_win.mean()))
                
    def update(self):
        torch.cuda.empty_cache()
        rewards = torch.from_numpy(self.env.reward_list).float().to(device).detach()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        value = torch.stack(self.memory.critic)
        loss_critic = F.mse_loss(value, rewards)
        
        f = torch.stack(self.memory.f).squeeze()
        is_win = torch.from_numpy(self.env.is_win).float().to(device).detach()
        loss_f = F.mse_loss(f, is_win)

        log_prob = torch.stack(self.memory.log_prob)
        advantages = rewards - value.detach()
        objective_actor = (advantages * torch.exp(log_prob)).mean()
        
        loss = - objective_actor +  self.loss_critic_coef*loss_critic + self.loss_position_coef*loss_f
        self.local_optim.zero_grad()
        loss.backward()
        gradients = [param.grad.clone().cpu() for param in self.local_model.parameters()]
        self.gradient_queue.put(gradients)
        self.loss_queue.put((objective_actor.item(), loss_critic.item(), loss_f.item()))
        self.memory.clear()

        


class Validate(Process):
    def __init__(self,  shared_model, ep_cnt, optimizer_lock, validate_queue):
        super(Validate, self).__init__()
        
        self.shared_model = shared_model
        
        self.optimizer_lock = optimizer_lock

        self.memory = Memory()
        
        self.validate_queue = validate_queue

        self.gamma = gamma
        self.render = render
    
    def run(self):
        print("Process_Validate_started")
        self.local_model = copy.deepcopy(self.shared_model).to(device)
        data = pd.read_csv(f"../data/M8888.XDCE_5m_test.csv")
        self.env = Env(data, 10000, img_shape, charge, init_money, threshold)
        while True:
            with self.optimizer_lock:
                self.local_model.load_state_dict(self.shared_model.state_dict())
            
            img, stat = self.env.reset()
            done = False
            while not done:
                action, f = self.local_model.act_max(img, stat)
                (img, stat), done = self.env.step(action, f)
            self.validate_queue.put((self.env.reward_list.sum(), self.env.money/self.env.init_money, self.env.is_win.mean()))
                
        


def update_shared_model(gradient_queue, optimizer_lock, optim, ac):
        while True:
            gradients = gradient_queue.get()
            with optimizer_lock:
                optim.zero_grad()
                for grads, params in zip(gradients, ac.parameters()):
                    params._grad = grads
                optim.step()


def train():
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    writer = SummaryWriter()
    ac = AC(latent_num, cnn_chanel_num, stat_dim)
    writer.add_graph(ac, (torch.zeros([1, 1, img_shape[0], img_shape[1]]), torch.zeros([1, stat_dim])))
    optim = GlobalAdam([
                {'params': ac.encode_img.parameters(), 'lr': 2.5e-5},
                {'params': ac.encode_stat.parameters(), 'lr': 2.5e-5},
                {'params': ac.pi.parameters(), 'lr': 2.5e-5},
                {'params': ac.actor.parameters(), 'lr': 2.5e-5},
                {'params': ac.f.parameters()},
                {'params': ac.V.parameters()}
            ], lr=5e-3, weight_decay=weight_decay)
    
    if os.path.exists('S3_state_dict.pt'):
        ac.load_state_dict(torch.load('S3_state_dict.pt'))
        optim.load_state_dict(torch.load('S3_Optim_state_dict.pt'))
    else:
        ac.load_state_dict(torch.load('../stage2/S2_state_dict.pt'), strict=False)

    result_queue = Queue()
    validate_queue = Queue()
    gradient_queue = Queue()
    loss_queue = Queue()
    ep_cnt = Value('i', 0)
    optimizer_lock = Lock()
    processes = []
    ac.share_memory()
        
    optimizer_worker = Process(target=update_shared_model, args=(gradient_queue, optimizer_lock, optim, ac))
    optimizer_worker.start()

    for no in range(mp.cpu_count()-3):
        worker = Worker(no, ac, ep_cnt, optimizer_lock, result_queue, gradient_queue, loss_queue)
        worker.start()
        processes.append(worker)
    validater = Validate(ac, ep_cnt, optimizer_lock, validate_queue)
    validater.start()

    best_reward = 0
    while True:
        with ep_cnt.get_lock():
            if not result_queue.empty():
                ep_cnt.value += 1
                reward, money, win_rate = result_queue.get()
                objective_actor, loss_critic, loss_f = loss_queue.get()


                writer.add_scalar('Interaction/Reward', reward, ep_cnt.value)
                writer.add_scalar('Interaction/Money', money, ep_cnt.value)
                writer.add_scalar('Interaction/win_rate', win_rate, ep_cnt.value)

                writer.add_scalar('Update/objective_actor', objective_actor, ep_cnt.value)
                writer.add_scalar('Update/loss_critic', loss_critic, ep_cnt.value)
                writer.add_scalar('Update/loss_f', loss_f, ep_cnt.value)

                
                with optimizer_lock:
                    if reward > best_reward:
                            best_reward = reward
                            torch.save(ac.state_dict(), 'S3_BEST_state_dict.pt')
                    if ep_cnt.value % save_every == 0:
                        torch.save(ac.state_dict(), 'S3_state_dict.pt')
                        torch.save(optim.state_dict(), 'S3_Optim_state_dict.pt')
            

            if not validate_queue.empty():
                val_reward, val_money, val_win_rate = validate_queue.get()

                writer.add_scalar('Validation/reward', val_reward, ep_cnt.value)
                writer.add_scalar('Validation/money', val_money, ep_cnt.value)
                writer.add_scalar('Validation/win_rate', val_win_rate, ep_cnt.value)

                        

    for worker in processes: worker.join()
    optimizer_worker.kill()

if __name__ == '__main__':
    train()