import numpy as np
import time
from scipy.optimize import minimize
from multiprocessing import Pool, Manager
from utils.parallel_cal_functions import get_ED_likehood_gradient,get_ED_likehood_gradient_HPPSELF
import copy

class PS(object):
    def __init__(self, cfg, trace, index2eds):
        self.index2eds       = index2eds #所有的ed字典
        self.iteration       = [0,0,0] # 训练轮数
        self.para_init       = cfg.para_init # HPP初始参数
        self.content_num     = trace.content_num # 视频总数
        self.update_interval = cfg.update_interval #模型在线更新间隔
        self.trace           = trace #数据集
        self.predict_policy  = cfg.predict_policy
        
        if self.predict_policy in ["HPP_PAC"]:
            self.penalty_beta = cfg.penalty[0]# 惩罚项  
            self.penalty_p    = cfg.penalty[1]# 惩罚项
            self.penalty_q    = cfg.penalty[2]# 惩罚项
            self.PAC_D        = cfg.PAC_D #PAC降维维数
            self.beta         = np.full((self.trace.ED_num,self.content_num), self.para_init[0]) #HPP参数
            #self.p           = np.random.random((self.content_num, cfg.PAC_D)) * self.para_init[1]
            #self.q           = np.random.random((self.content_num, cfg.PAC_D)) * self.para_init[2]
            self.p            = np.full((self.content_num, self.PAC_D), self.para_init[1]) #HPP参数
            self.q            = np.full((self.content_num, self.PAC_D), self.para_init[2]) #HPP参数
            self.beta_zero    = np.ones_like(self.beta) * 1e-9 #HPP参数优化后最小值
            self.p_zero       = np.ones_like(self.p) * 1e-9 #HPP参数优化后最小值
            self.q_zero       = np.ones_like(self.q) * 1e-9 #HPP参数优化后最小值
        elif self.predict_policy in ["HPP_SELF"]:
            self.penalty_beta  = cfg.penalty[0]# 惩罚项
            self.penalty_omega = cfg.penalty[1]# 惩罚项
            self.beta          = np.full(self.content_num, self.para_init[0]) #HPP参数
            self.omega         = np.full(self.content_num, self.para_init[1]) #HPP参数
            self.beta_zero     = np.ones_like(self.beta) * 1e-9 #HPP参数优化后最小值
            self.omega_zero    = np.ones_like(self.omega) * 1e-9 #HPP参数优化后最小值
        elif self.predict_policy in ["HRS","NO_MODEL","HIS","FUT","HIS_ONE"]:
            pass
        else:
            raise ValueError(f"not this predict_policy {self.predict_policy}.")
    
        self.maxProgressNum = cfg.maxProgressNum #最大并行进程数
        self.if_print       = cfg.if_print #是否打印log
        self.ftol           = cfg.ftol     #训练精度目标
        self.maxiter        = cfg.maxiter  #最大训练轮次
        self.summary_writer = cfg.summary_writer # tensorboard相关参数


    def global_Likelihood(self,para_vector):
        self.begintime = time.time() 
        self.vector_to_para(para_vector) #将最新的参数值更新到PS以计算惩罚项
        
        # 串行计算
        # self.results = {}
        # for ed_index in self.index2eds.keys():
        #     ed = self.index2eds[ed_index]
        #     self.results[ed_index] = ed.HPP.likelihood(para_vector)

        # 并行计算所有edge的likelihood和梯度
        self.results =  Manager().dict()
        parallel_para = []
        for ed_index in self.index2eds.keys():
            ed_events = self.trace.get_ed_trace(ed_index)
            ed = self.index2eds[ed_index]
            edEventArray = ed_events[(ed_events['time'] >=self.t_o-self.update_interval) & (ed_events['time'] < self.t_o)].to_numpy(dtype=np.int32)
            if self.predict_policy in ["HPP_PAC"]:
                parallel_para.append([ 
                    ed_index, self.results, \
                    edEventArray, \
                    self.beta[ed_index,:], self.p, self.q, \
                    ed.HPP.eventNum_t, ed.HPP.sumKernel, ed.HPP.sumI1, ed.HPP.PPF_delta,\
                    self.content_num, self.PAC_D, self.update_interval, self.t_o\
                    ])
            elif self.predict_policy in ["HPP_SELF"]:
                parallel_para.append([ 
                    ed_index, self.results, \
                    edEventArray, \
                    self.beta, self.omega, \
                    ed.HPP.eventNum_t, ed.HPP.sumKernel, ed.HPP.sumI1, ed.HPP.PPF_delta,\
                    self.content_num, self.update_interval, self.t_o\
                    ])
            else:
                raise ValueError(f"This predict_policy {self.predict_policy} can not training by FL.")
        with Pool(min(len(parallel_para),self.maxProgressNum)) as pool:
            if self.predict_policy in ["HPP_PAC"]:
                pool.starmap(get_ED_likehood_gradient, parallel_para) #同时计算likehood和gradient
            elif self.predict_policy in ["HPP_SELF"]:
                pool.starmap(get_ED_likehood_gradient_HPPSELF, parallel_para) #同时计算likehood和gradient
            else:
                raise ValueError(f"This predict_policy {self.predict_policy} can not training by FL.")
            pool.close()
            # print('--pool close--')
            pool.join()
            # print('--clean pool --')

        loss_global = sum(self.results[ed_index][0] for ed_index in self.results.keys())
        
        #计算惩罚项
        if self.predict_policy in ["HPP_PAC"]:
            self.loss_global = loss_global \
                + 0.5 * self.penalty_beta  * np.sum(self.beta  * self.beta)\
                + 0.5 * self.penalty_p * np.sum(self.p * self.p) \
                + 0.5 * self.penalty_q * np.sum(self.q * self.q)
        elif self.predict_policy in ["HPP_SELF"]:
            self.loss_global = loss_global \
                + 0.5 * self.penalty_beta  * np.sum(self.beta  * self.beta)\
                + 0.5 * self.penalty_omega * np.sum(self.omega * self.omega)
        else:
            raise ValueError(f"not this predict_policy {self.predict_policy}.")
        
        if self.if_print:
            #print(self.iteration,'Likelihood use time:', time.time() - self.begintime, 's')
            print(self.iteration,'Likelihood:', self.loss_global)
        return self.loss_global
    
    def global_Gradient(self,para_vector):
        begintime = time.time()
        if self.predict_policy in ["HPP_PAC"]:
            grad_beta = np.zeros_like(self.beta)
            grad_p    = np.zeros_like(self.p)
            grad_q    = np.zeros_like(self.q)

            for ed_index in self.results.keys():
                grad_beta[ed_index,:] = self.results[ed_index][1].copy()
                grad_p    += self.results[ed_index][2]
                grad_q    += self.results[ed_index][3]
            #计算惩罚项
            grad_beta += self.penalty_beta  * self.beta
            grad_p    += self.penalty_p * self.p
            grad_q    += self.penalty_q * self.q

            all_grad = np.concatenate([grad_beta.ravel(), grad_p.ravel(),grad_q.ravel()])
        elif self.predict_policy in ["HPP_SELF"]:
            grad_beta  = np.zeros_like(self.beta)
            grad_omega = np.zeros_like(self.omega)

            for ed_index in self.results.keys():
                grad_beta  += self.results[ed_index][1]
                grad_omega += self.results[ed_index][2]
            
            #计算惩罚项
            grad_beta  += self.penalty_beta  * self.beta
            grad_omega += self.penalty_omega * self.omega

            all_grad = np.concatenate([grad_beta.ravel(), grad_omega.ravel()])
        else:
            raise ValueError(f"not this predict_policy {self.predict_policy}.")
        if self.if_print:
            #print(self.iteration,'Gradient use time:', time.time() - begintime, 's',end=' ')
            print(self.iteration,'Use time:', time.time() - self.begintime, 's')
            self.iteration[2] += 1
        return all_grad

    def vector_to_para(self, para_vector):
        if self.predict_policy in ["HPP_PAC"]:
            begin = 0 
            end = begin + np.cumprod(self.beta.shape)[-1]
            self.beta = para_vector[begin : end].reshape(self.beta.shape)
            self.beta_zero [self.beta <= 0] = self.beta_zero[self.beta <= 0] / 10
            self.beta[self.beta <= 0] = self.beta_zero[self.beta <= 0]
            begin = end 
            end = begin + np.cumprod(self.p.shape)[-1]
            self.p = para_vector[begin : end].reshape(self.p.shape)
            self.p_zero[self.p <= 0] = self.p_zero[self.p <= 0] / 10
            self.p[self.p <= 0] = self.p_zero[self.p <= 0]
            begin = end 
            end = begin + np.cumprod(self.q.shape)[-1]
            self.q = para_vector[begin : end].reshape(self.q.shape)
            self.q_zero[self.q <= 0] = self.q_zero[self.q <= 0] / 10
            self.q[self.q <= 0] = self.q_zero[self.q <= 0]
        elif self.predict_policy in ["HPP_SELF"]:
            begin = 0 
            end = begin + np.cumprod(self.beta.shape)[-1]
            self.beta = para_vector[begin : end].reshape(self.beta.shape)
            self.beta_zero [self.beta <= 0] = self.beta_zero[self.beta <= 0] / 10
            self.beta[self.beta <= 0] = self.beta_zero[self.beta <= 0]
            begin = end 
            end = begin + np.cumprod(self.omega.shape)[-1]
            self.omega = para_vector[begin : end].reshape(self.omega.shape)
            self.omega_zero[self.omega <= 0] = self.omega_zero[self.omega <= 0] / 10
            self.omega[self.omega <= 0] = self.omega_zero[self.omega <= 0]
        else:
            raise ValueError(f"not this predict_policy {self.predict_policy}.")

    def para_to_vector(self):
        if self.predict_policy in ["HPP_PAC"]:
            return np.concatenate([self.beta.ravel(), self.p.ravel(), self.q.ravel()])
        elif self.predict_policy in ["HPP_SELF"]:
            return np.concatenate([self.beta.ravel(), self.omega.ravel()])
        else:
            raise ValueError(f"not this predict_policy {self.predict_policy}.")

    def update_eds_index(self,index2eds):
        self.index2eds = index2eds

    def set_t_o(self,t_o):
        self.t_o = t_o

    def global_fit_para(self):
        #开始训练
        self.first_time = time.time()
        self.last_time =  time.time()

        def call_back(x):
            if self.if_print:
                print(self.iteration[0:2],'Iteration callback, this iteration use time:', time.time() - self.last_time, 's')
                print(self.iteration[0:2],'Iteration callback, totally use time:', time.time() - self.first_time, 's') 
                print(self.iteration[0:2],'Iteration callback, likehood: ' + str(self.loss_global))
            self.summary_writer.add_scalar("loss_global", self.loss_global, self.iteration[1])
            self.iteration[1] += 1
            self.iteration[2]  = 0
            self.last_time = time.time()
            self.para_vector_backup = copy.deepcopy(x)

        # 训练参数约束：均无约束条件

        if self.predict_policy in ["HPP_PAC"]:
            bnd_beta  = np.cumprod(self.beta.shape) [-1]
            bnd_p = np.cumprod(self.p.shape)[-1] + bnd_beta
            bnd_q = np.cumprod(self.q.shape)[-1] + bnd_p
            bnds      = [(0,None) for i in range(bnd_beta)] + \
                        [(0,None) for i in range(bnd_beta , bnd_p)] + \
                        [(0,None) for i in range(bnd_p , bnd_q)] 
            self.beta = np.full((self.trace.ED_num,self.content_num), self.para_init[0]) #HPP参数
            self.p    = np.full((self.content_num, self.PAC_D), self.para_init[1]) #HPP参数
            self.q    = np.full((self.content_num, self.PAC_D), self.para_init[2]) #HPP参数
            self.beta_zero    = np.ones_like(self.beta) * 1e-9 #HPP参数优化后最小值
            self.p_zero       = np.ones_like(self.p) * 1e-9 #HPP参数优化后最小值
            self.q_zero       = np.ones_like(self.q) * 1e-9 #HPP参数优化后最小值
        elif self.predict_policy in ["HPP_SELF"]:
            self.beta  = np.full(self.content_num, self.para_init[0]) #HPP参数
            self.omega = np.full(self.content_num, self.para_init[1]) #HPP参数
            self.beta_zero     = np.ones_like(self.beta) * 1e-9 #HPP参数优化后最小值
            self.omega_zero    = np.ones_like(self.omega) * 1e-9 #HPP参数优化后最小值
            bnd_beta  = np.cumprod(self.beta.shape) [-1]
            bnd_omega = np.cumprod(self.omega.shape)[-1] + bnd_beta
            bnds      = [(0,None) for i in range(bnd_beta)] + \
                        [(0,None) for i in range(bnd_beta , bnd_omega)] 
        else:
            raise ValueError(f"not this predict_policy {self.predict_policy}.")

        init_para_vector = self.para_to_vector()
        self.para_vector_backup = copy.deepcopy(init_para_vector)

        res = minimize(self.global_Likelihood,
                        init_para_vector,
                        method = 'L-BFGS-B',
                        jac = self.global_Gradient,
                        bounds = bnds,
                        options = {'ftol': self.ftol, 'maxiter': self.maxiter, 'maxls':10},
                        callback = call_back
                        )

        if self.if_print:
            print('Likelihood '+str(self.loss_global))
            print('Totally use time: ', time.time() - self.first_time, 's') 
        
        if res.success:
            self.vector_to_para(res.x) #将最新的参数值更新到PS以计算惩罚项
        else:
            self.vector_to_para(self.para_vector_backup)
            print(res.message)
        self.iteration[0] += 1
        self.iteration[2]  = 0
        return res

