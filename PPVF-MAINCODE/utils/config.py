import argparse
import os
import time

import torch
from global_config import GLOBAL_PATH, DATA_PATH
from torch.utils.tensorboard import SummaryWriter

class Config:
    def __init__(self):
        def float_list(value):
            try:
                # 将逗号分隔的字符串转换为浮点数列表
                float_values = [float(x) for x in value.split(',')]
                return float_values
            except ValueError:
                raise argparse.ArgumentTypeError("Invalid float list format")
        def int_list(value):
            try:
                # 将逗号分隔的字符串转换为浮点数列表
                int_values = [int(x) for x in value.split(',')]
                return int_values
            except ValueError:
                raise argparse.ArgumentTypeError("Invalid int list format")

        self.current_time = time.strftime(
            '%m-%d-%H-%M-%S', time.localtime())
        parser = argparse.ArgumentParser(description=self.current_time)

        #运行相关参数
        parser.add_argument("--fetch_policy", type=str, default="PPVF") # Edge的缓存策略 
        parser.add_argument("--predict_policy", type=str, default="HPP_PAC") # prediction model
        parser.add_argument("--noise", action="store_true") 
        parser.add_argument("--train_level", type=str, default="edge") # province / city / isp / edge
        parser.add_argument("--run_way", type=str, default="training") # training / testing
        parser.add_argument("--ifLoadKernel", action="store_true") 
        parser.add_argument("--maxProgressNum", type=int, default=25) # 

        #数据集相关参数
        parser.add_argument("--PPF_update_interval_day", type=float, default=10) # 在线更新间隔/天 
        parser.add_argument("--slot_interval", type=int, default=60) # 总time slot间隔/min, 60min更新cache
        parser.add_argument('--spilt_day', type=int_list, default="10,30") # 前12天初始化环境, 30天是总的环境
        parser.add_argument('--dataset_percent',type=float,default=0.005) # 0.5 => 视频平台的视频总量比例

        #在线算法相关参数
        # parser.add_argument("--c_e", type=int, default=100) 0.001,0.002,0.005,0.01,0.02,0.05,0.1
        # parser.add_argument("--f_e", type=int, default=2) 
        parser.add_argument("--f_e", type=int_list, default="3") 
        parser.add_argument("--c_e_ratio", type=float_list, default="0.001,0.002,0.005,0.01,0.02,0.05,0.1") 
        parser.add_argument("--epsilon", type=float_list, default="1")
        parser.add_argument("--xi", type=float_list, default="10.0")

        #HPP模型相关参数
        parser.add_argument("--PPF_delta", type=float, default=0.1) #HPP_self：0.05-0.1
        parser.add_argument("--penalty", type=float_list, default="10000.0,10000.0,10000.0")
        parser.add_argument("--para_init", type=float_list, default="1.0,1.0,1.0")
        parser.add_argument("--maxiter", type=int, default=20) 
        parser.add_argument("--PAC_D", type=int, default=10) 
        parser.add_argument("--ftol", type=float, default=1e-9) 
        
        # In config.py
        parser.add_argument("--verbose", action="store_true", help="Enable detailed simulation logs")
        self.verbose = log_parser.verbose
        


        log_parser = parser.parse_args()

        #实验方式设置
        self.if_print       = True
        self.run_way        = log_parser.run_way
        self.fetch_policy   = log_parser.fetch_policy
        self.predict_policy = log_parser.predict_policy
        self.if_noise       = log_parser.noise
        self.train_level    = log_parser.train_level
        self.ifLoadKernel   = log_parser.ifLoadKernel
        self.maxProgressNum = log_parser.maxProgressNum

        # 调谐一些参数
        if self.fetch_policy in ["PPVF"]:
            self.if_noise = True
        else:
            raise  ValueError(f"not this fetch_policy {self.fetch_policy}")

        #数据集相关参数
        self.dataset_percent     = log_parser.dataset_percent
        self.test_day            = log_parser.spilt_day[0] # 单位：day
        self.all_day             = log_parser.spilt_day[1]
        self.slot_interval       = log_parser.slot_interval  # 单位：min
        self.PPF_update_interval_day = log_parser.PPF_update_interval_day
        self.day_2_time          = 1440 // self.slot_interval
        self.update_interval     = int(self.day_2_time * self.PPF_update_interval_day)
        self.all_t               = self.all_day * self.day_2_time

        #在线算法相关参数
        f_e_tmp = log_parser.f_e
        epsilon_tmp = log_parser.epsilon
        c_e_ratio_tmp  = log_parser.c_e_ratio   #Ce ratio is doubtful not defined in paper
        xi_tmp = log_parser.xi


        if len(f_e_tmp) == 1 and len(epsilon_tmp) ==1 and len(xi_tmp) ==1:
            self.test_hyper_paras = "c_e"
            self.f_e              = f_e_tmp[0]
            self.c_e_ratio        = c_e_ratio_tmp
            self.epsilon          = epsilon_tmp[0]
            self.xi               = xi_tmp[0]
        elif len(f_e_tmp) > 1 and len(c_e_ratio_tmp)==1 and len(epsilon_tmp) ==1 and len(xi_tmp)==1:
            self.test_hyper_paras = "f_e"
            self.f_e              = f_e_tmp
            self.c_e_ratio        = c_e_ratio_tmp[0]
            self.epsilon          = epsilon_tmp[0]
            self.xi               = xi_tmp[0]
        elif len(f_e_tmp) == 1 and len(c_e_ratio_tmp)==1 and len(epsilon_tmp) > 1 and len(xi_tmp)==1 and self.if_noise:
            self.test_hyper_paras = "epsilon"
            self.f_e              = f_e_tmp[0]
            self.c_e_ratio        = c_e_ratio_tmp[0]
            self.epsilon          = epsilon_tmp
            self.xi               = xi_tmp[0]
        elif len(f_e_tmp) == 1 and len(c_e_ratio_tmp)==1 and len(epsilon_tmp) == 1 and len(xi_tmp)>1:
            self.test_hyper_paras = "xi"
            self.f_e              = f_e_tmp[0]
            self.c_e_ratio        = c_e_ratio_tmp[0]
            self.epsilon          = epsilon_tmp[0]
            self.xi               = xi_tmp
        else:
            raise ValueError("can not test c_e, f_e, epsilon or xi together")
        
        # if self.if_noise :
        #     self.epsilon = log_parser.epsilon
        # else:
        #     self.epsilon = 0
        # self.epsilon = log_parser.epsilon

        #HPP模型训练及预测相关参数
        self.PPF_delta = log_parser.PPF_delta
        self.penalty   = log_parser.penalty
        self.para_init = log_parser.para_init
        self.maxiter   = log_parser.maxiter
        self.PAC_D     = log_parser.PAC_D
        self.ftol      = log_parser.ftol

        # 配置参数
        self.path           = GLOBAL_PATH
        self.data_path      = DATA_PATH
        self.model_path     = os.path.join(self.path, "HPPModel/")
        self.device         = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        if self.if_noise:
            self.parameter_name = f"Noise:{self.if_noise}-fe:{self.f_e}-ce:{self.c_e_ratio}-xi:{self.xi}-epsilon:{self.epsilon}-dataset:{self.dataset_percent}"
        else:
            self.parameter_name = f"Noise:{self.if_noise}-fe:{self.f_e}-ce:{self.c_e_ratio}-dataset:{self.dataset_percent}"


        self.summary_path   =  os.path.join(self.path, f'tensorboard/202401/{self.current_time}-{self.fetch_policy}-{self.predict_policy}-{self.run_way}-{self.parameter_name}/')
        self.actions_path = os.path.join(self.path, f'final_state/final_actions/{self.current_time}-{self.fetch_policy}-{self.predict_policy}-{self.run_way}-{self.parameter_name}.npy')
        self.hitrate_path = os.path.join(self.path, f'final_state/final_hitrate/{self.current_time}-{self.fetch_policy}-{self.predict_policy}-{self.run_way}-{self.parameter_name}.npy')
        self.remained_b_path = os.path.join(self.path, f'final_state/final_remained_b/{self.current_time}-{self.fetch_policy}-{self.predict_policy}-{self.run_way}-{self.parameter_name}.npy')
        self.summary_writer = SummaryWriter(self.summary_path)
        self.log_h_parameters()
        print("Current time: ", self.current_time)


    def log_h_parameters(self):
        # print(f'test_day: {self.test_day}')
        # print(f'all_day: {self.all_day}')
        # print(f'slot_interval: {self.slot_interval}')
        # print(f'update_interval: {self.update_interval}')
        # print(f'c_e: {self.c_e}')
        # print(f'xi: {self.xi}')
        # print(f'PPF_delta: {self.PPF_delta}')
        # print(f'penalty: {self.penalty}')
        # print(f'para_init: {self.para_init}')

        self.summary_writer.add_text(
            'test_day', str(self.test_day), 0)
        self.summary_writer.add_text(
            'all_day', str(self.all_day), 0)
        self.summary_writer.add_text(
            'slot_interval', str(self.slot_interval), 0)
        self.summary_writer.add_text(
            'update_interval', str(self.update_interval), 0)
        self.summary_writer.add_text(
            'c_e_ratio', str(self.c_e_ratio), 0)
        self.summary_writer.add_text(
            'f_e', str(self.f_e), 0)
        self.summary_writer.add_text(
            'xi', str(self.xi), 0)
        self.summary_writer.add_text(
            'PPF_delta', str(self.PPF_delta), 0)
        self.summary_writer.add_text(
            'penalty', str(self.penalty), 0)
        self.summary_writer.add_text(
            'para_init', str(self.para_init), 0)
