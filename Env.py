import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch import multiprocessing
from torch.multiprocessing import Queue, Value, Lock, Process
from torch.optim import Adam
import stockstats
import plotly.graph_objects as go
import plotly.express as px
import os
from gym.envs.classic_control import rendering
import pyglet
from pyglet.window import key
import pyglet.gl
from pyglet.gl import *
import time
import copy


class Env():
    
    def __init__(self, data, total_step, img_shape, charge, init_money, threshold):
        self.stats = stockstats.StockDataFrame(data)
        def normalize(x):
            x = x.copy()
            return (x - np.mean(x)) / (np.std(x) + 1e-5)
        self.wr = normalize(self.stats['wr_14'])
        self.kdjk = normalize(self.stats['kdjk_20'])
        self.kdjd = normalize(self.stats['kdjd_20'])
        self.kdjj = normalize(self.stats['kdjj_20'])
        self.rsi = normalize(self.stats['rsi_12'])
        self.adx = normalize(self.stats['adx'])
        self.dma = normalize(self.stats['dma'])
        
        self.len = len(self.stats)
        self.start_pos = None
        self.cur_pos = None
        self.total_step = total_step
        self.step_cnt = None
        self.in_cnt = None
        self.is_win = None
        self.reward_list = None
        self.img_shape = img_shape
        self.charge = charge
        self.action = None # 0:buy, 1:wait, 2:sell
        self.f = None
        self.threshold = threshold
        self.init_money = init_money
        self.money = None
        self.contract = None
        self.sign = None
        self.in_close = None
        self.window = None
        print("Env_started")
        
    def reset(self):
        self.step_cnt = 0
        self.action = 1
        self.money = self.init_money
        self.contract = 0
        self.in_close = None
        self.f = 0
        self.in_cnt = None
        self.is_win = np.zeros(self.total_step)
        self.reward_list = np.zeros(self.total_step)
        if self.start_pos is None:
            self.cur_pos = np.random.randint(self.img_shape[1] - 1, self.len - self.total_step)
            self.start_pos = self.cur_pos
        else:
            self.cur_pos = self.start_pos
        return self.grey_scale_img(self.cur_pos), self.get_stats(self.cur_pos)
        
    def step(self, action, f):
        
        self.step_cnt += 1

        done = False
        if self.step_cnt == self.total_step:
            done = True
            self.start_pos = None
        
        self.calc_r(action, f)

        self.action = action
        img = self.grey_scale_img(self.cur_pos)
        stats = self.get_stats(self.cur_pos)

        self.cur_pos += 1
        return (img, stats), done

    def calc_r(self, cur_action, f):
        act2sign = [1, 0, -1]
        sign = act2sign[self.action]
        cur_sign = act2sign[cur_action]
        cur_close = self.stats.loc[self.cur_pos].iat[2]
        next_close = self.stats.loc[self.cur_pos+1].iat[2]
        def kai_cang():
            self.sign = cur_sign
            self.contract = cur_sign * (self.money * f // cur_close)
            self.in_close = cur_close
            if self.in_cnt is None:
                self.in_cnt = self.step_cnt
        def ping_cang():
            if self.in_close:
                delta = next_close - self.in_close
                self.reward_list[self.in_cnt-1: self.step_cnt] =  80 * np.tanh(0.02 * np.linspace(delta * self.sign, 0, self.step_cnt-self.in_cnt+1))
                if delta * self.sign > 0:
                    self.is_win[self.in_cnt-1: self.step_cnt] = 1
                self.in_cnt = None
                self.money = self.money + (delta - self.charge) * self.contract
                self.contract = 0
                self.sign = None
                self.in_close = None
            
        if sign == 0 and cur_sign != 0:
            kai_cang()
        elif cur_sign == 0:
            ping_cang()
        elif sign == cur_sign:
            if abs(f - self.f) >= self.threshold:
                self.f = f
                ping_cang()
                kai_cang()
        else:
            ping_cang()
            kai_cang()
        
    def render(self):
        scale = 2
        if self.window == None:
            self.window = pyglet.window.Window(width=self.img_shape[1]*2*scale, height=self.img_shape[0]*scale)
        img = np.expand_dims(self.grey_scale_img(self.cur_pos), -1).repeat(3,-1) * 255
        img = np.array(img, dtype=np.uint8)
        image = pyglet.image.ImageData(img.shape[1], img.shape[0], 'RGB', img.tobytes(), pitch=img.shape[1]*-3)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        texture = image.get_texture()
        texture.width = self.img_shape[1]*scale
        texture.height = self.img_shape[0]*scale
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        label = pyglet.text.Label(str(self.money), font_name='Times New Roman', font_size=10, x=self.img_shape[1]*scale, y=self.img_shape[0])
        label.draw()
        texture.blit(0, 0)
        self.window.flip()
        
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
        return a
        
    def get_stats(self, pos):
        return np.array((self.wr[pos], self.kdjk[pos], self.kdjd[pos], self.kdjj[pos], self.rsi[pos], self.adx[pos], self.dma[pos]))