import numpy as np
import numba as nb
import math
from collections import Counter
import time
from scipy.optimize import minimize
# import time
# import os
# from global_config import GLOBAL_PATH

class HPP:
    def __init__(
        self,
        #events,
        cfg,
        content_num,
        #init_para_vector,
        if_print   = False
    ):
        #超参数
        self.PPF_delta           = cfg['PPF_delta']
        #self.slot_interval   = cfg['slot_interval']
        #self.update_interval = cfg['update_interval']
        #self.PAC_D           = cfg['PAC_D']

        #数据集相关参数
        self.all_t       = cfg['all_t'] #最大时间
        # self.t_o         = 0 #更新时间点
        #self.events     = events
        self.content_num = content_num
        # self.ED_index    = int(events['ed'].drop_duplicates().values)
        self.sumKernel  = np.zeros((self.content_num,self.all_t), dtype = np.float64) #核函数表
        self.sumI1      = np.zeros((self.content_num,self.all_t), dtype = np.float64) #部分核函数积分表
        self.eventNum_t = np.zeros((self.content_num,self.all_t), dtype = np.int32) #历史事件请求次数表


    def setKernel(self,kerenls):
        self.eventNum_t = kerenls[0]
        self.sumKernel  = kerenls[1]
        self.sumI1      = kerenls[2]
        