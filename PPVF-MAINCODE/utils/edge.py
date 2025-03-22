import numpy as np
from utils.hpp import HPP
import random
from scipy.stats import pearsonr
from collections import OrderedDict
import copy

class ED(object):
    def __init__(self, index, cfg, content_num, events):   

        # 公共配置
        self.content_num = int(content_num) #视频总数
        self.ed_index = index

        self.predict_policy = cfg["predict_policy"] # 效用的预测模型
        self.fetch_policy   = cfg['fetch_policy'] # 预请求策略
        self.if_noise       = cfg['if_noise'] #是否加噪

        self.c_e        = cfg["c_e"]
        self.cache_list = [OrderedDict() for _ in range(cfg["hyper_paras_num"])] #有序缓存字典
        # self.epsilon    = cfg["epsilon"]#单轮请求决策消耗的隐私预算
        # self.xi         = cfg["xi"] #总的隐私预算
        self.remained_b = np.zeros((len(self.c_e),self.content_num),dtype=np.float32) # 目前隐私预算消耗的比例
        # self.remained_action_times = np.full(fill_value=int(cfg['xi']/),shape=(len(self.c_e),self.content_num),dtype=np.int32) # 目前隐私预算消耗的比例

        self.n          = 1 # 请求的次数
        self.actions    = np.zeros((len(self.c_e),self.content_num),dtype=np.int64) 

        if self.if_noise:
            self.sum_XY = np.zeros((self.content_num, self.content_num))  
            self.sum_X = np.zeros(shape=self.content_num)
            self.sum_X_square = np.zeros(shape=self.content_num)
            # self.sum_X = np.zeros((self.content_num, 1))  
            # self.sum_X_square = np.zeros((self.content_num, 1)) 

        if self.fetch_policy in ["PPVF"]:# 缩放函数所需
            self.Up_bound  = 1 # 效用上限
            self.Low_bound = 0.1 # 效用下限
            self.B0        = 1 / (1 + np.log(self.Up_bound / self.Low_bound)) #最低阈值
        else:
            raise ValueError(f"not this fetch policy: {self.fetch_policy}")
        
        self.density = np.ones(self.content_num) / self.content_num
        if self.predict_policy in ["HPP_PAC","HPP_SELF"]:
            self.HPP = HPP(content_num = self.content_num, cfg = cfg, if_print = False)
        elif self.predict_policy in ["HRS"]:
            self.HPP = HRS(ed_index = self.ed_index, events = events, content_num = self.content_num, cfg = cfg)
        elif self.predict_policy in ["HIS"]:
            self.content_count = np.ones(shape=self.content_num,dtype=np.int32)
        elif self.predict_policy in ["NO_MODEL","FUT","HIS_ONE"]:
            pass
        else:
            raise ValueError(f"not this predict policy: {self.predict_policy}")

    def get_hitrate(self, ed_slot_trace):
        """
        Calculate the hit rate for the edge device (ED) in the given time slot trace.

        Args:
            ed_slot_trace (pd.DataFrame): Request trace for this ED in current time slot.

        Returns:
            ed_click (int): Total number of video requests (clicks).
            ed_hit (np.ndarray): Array of hit counts for each cache size.
        """

        def if_success(x, cachingSet):
            # Check if the requested video is present in cache
            return 1 if x['i'] in cachingSet else 0

        ed_click = ed_slot_trace.shape[0]  # Total number of video requests (clicks)
        print(f"[ED {self.ed_index}] Total video requests (clicks) in this slot: {ed_click}")
        print(f"[ED {self.ed_index}] Request details: {ed_slot_trace[['i', 'time']].to_dict('records')}")

        if ed_click != 0:
            ed_hit = []
            for idx, cache in enumerate(self.cache_list):  # Iterate over each cache configuration
                hit_count = int(np.sum(ed_slot_trace.apply(if_success, axis=1, cachingSet=set(cache.keys()))))  # Number of hits
                print(f"[ED {self.ed_index}, Cache {idx}] Cache contents: {list(cache.keys())}")
                print(f"[ED {self.ed_index}, Cache {idx}] Hit count: {hit_count}, Cache size: {len(cache)}")
                ed_hit.append(hit_count)
            ed_hit = np.array(ed_hit, dtype=np.int64)
        else:
            # No requests were made in this time slot
            ed_hit = np.zeros(len(self.cache_list), dtype=np.int64)
            print(f"[ED {self.ed_index}] No requests in this slot, all hit counts set to zero.")

        print(f"[ED {self.ed_index}] Summary - Total clicks: {ed_click}, Hits across caches: {ed_hit}")
        return ed_click, ed_hit



    def update_density(self, density):
        self.density = density

    def calculate_pearson_correlation(self):
        #data         = data[:,np.newaxis]
        self.n            += 1
        self.sum_XY       += self.density @ self.density.T
        self.sum_X        += self.density
        self.sum_X_square += self.density * self.density
        print(f"[ED {self.ed_index}] Updating correlation stats - Iteration: {self.n}")
        print(f"[ED {self.ed_index}] Sum of density products (sum_XY sample): {self.sum_XY[:5, :5]}")
        print(f"[ED {self.ed_index}] Sum of densities (sum_X sample): {self.sum_X[:5]}")
        print(f"[ED {self.ed_index}] Sum of squared densities (sum_X_square sample): {self.sum_X_square[:5]}")

        if self.n > 2:
            assert (self.n*self.sum_X_square>=self.sum_X * self.sum_X).all(), [self.n,self.ed_index]
            sigma_X = np.sqrt(self.n * self.sum_X_square - (self.sum_X * self.sum_X))
            denominator =  sigma_X @ sigma_X.T
            numerator = self.n * self.sum_XY - (self.sum_X @ self.sum_X.T) 
            psi = numerator / denominator
            print(f"[ED {self.ed_index}] Computed sigma_X (sample): {sigma_X[:5]}")
            print(f"[ED {self.ed_index}] Correlation matrix (psi sample): {psi[:5, :5]}")
        else:
            psi = np.eye(self.content_num,dtype=np.float64)
            print(f"[ED {self.ed_index}] Initial correlation matrix (identity) due to n <= 2")
        return psi

    def get_redundant_request(self, cache_index, f_e_total, epsilon, xi, psi): 
        #归一化
        def scale_value(value, input_min, input_max, output_min, output_max):
            input_range = input_max - input_min
            if input_range == 0:
                return value
            output_range = output_max - output_min
            scaled_value = ((value - input_min) / input_range) * output_range + output_min
            return scaled_value

        def get_th(remained_b): 
            if remained_b < self.B0:
                return self.Low_bound
            else:
                return (((self.Up_bound * np.e) / self.Low_bound)**remained_b) * (self.Low_bound / np.e)

        np.random.seed(self.ed_index)
        scaled_density = scale_value(self.density, self.density.min(), self.density.max(), self.Low_bound, self.Up_bound) 
        print(f"[ED {self.ed_index}, Cache {cache_index}] Scaled density (sample): {scaled_density[:5]}")
        print(f"[ED {self.ed_index}, Cache {cache_index}] Privacy params - epsilon: {epsilon}, xi: {xi}, f_e_total: {f_e_total}")

        cache = self.cache_list[cache_index]
        action = np.zeros(shape=self.content_num,dtype=np.int8)
        f_k = 0 
        if self.fetch_policy == "PPVF": 
            sampled_indexes = np.random.choice(range(self.content_num), size = self.content_num, replace = False)
            print("Size of sampled_indexes:",len(sampled_indexes))
            print(f"[ED {self.ed_index}, Cache {cache_index}] Sampled video indices (sample): {sampled_indexes[:5]}")
            print("Sampled Videos Size:",len(sampled_indexes))
            for i in sampled_indexes:
                if f_k + 1 > f_e_total:
                    print(f"[ED {self.ed_index}, Cache {cache_index}] Stopping - Reached f_e_total limit: {f_e_total}")
                    break
                elif (self.remained_b[cache_index,i] / xi) <= self.B0:     #condition unidentified
                    f_k += 1
                    action[i] = 1
                    self.remained_b[cache_index,i] = self.remained_b[cache_index,i] + epsilon  
                    print(f"[ED {self.ed_index}, Cache {cache_index}] Video {i} selected (below B0), f_k: {f_k}, remaining budget: {self.remained_b[cache_index,i]}")
                elif (scaled_density[i] / epsilon > get_th(self.remained_b[cache_index,i] / xi )) and epsilon <= xi - self.remained_b[cache_index,i]:
                    f_k += 1
                    action[i] = 1
                    self.remained_b[cache_index,i] = self.remained_b[cache_index,i] + epsilon  
                    print(f"[ED {self.ed_index}, Cache {cache_index}] Video {i} selected (above threshold), Threshold: {get_th(self.remained_b[cache_index,i] / xi)}, f_k: {f_k}, remaining budget: {self.remained_b[cache_index,i]}")
                else:
                    action[i] = 0
                    print(f"[ED {self.ed_index}, Cache {cache_index}] Video {i} not selected, Threshold: {get_th(self.remained_b[cache_index,i] / xi)}")

        if self.if_noise:
            action_indices  = np.argwhere(action==1)[:,0]
            redundant_action = np.zeros(shape=self.content_num,dtype=np.int8)
            print(f"[ED {self.ed_index}, Cache {cache_index}] Action indices for noise: {action_indices}")

            psi[np.abs(psi) < 0.95] = 0
            print(f"[ED {self.ed_index}, Cache {cache_index}] Adjusted psi (sample, thresholded at 0.95): {psi[:5, :5]}")

            if len(action_indices) > 0:
                action_density  = self.density[action_indices]
                sensitivity = psi[action_indices,action_indices] @ action_density.T
                sen_c = sensitivity.max()
                probabilities   = np.exp(epsilon * action_density / (2 * sen_c))
                probabilities  /= np.sum(probabilities)
                random_videos_indices = np.random.choice(action_indices, size=len(action_indices), p=probabilities, replace=True)
                redundant_action[random_videos_indices] = 1
                if np.isscalar(sensitivity):
                    print(f"[ED {self.ed_index}, Cache {cache_index}] CDP - Sensitivity: {sensitivity}, Global sensitivity: {sen_c}")
                else:
                    print(f"[ED {self.ed_index}, Cache {cache_index}] CDP - Sensitivity (sample): {sensitivity[:5]}, Global sensitivity: {sen_c}")
                print(f"[ED {self.ed_index}, Cache {cache_index}] CDP - Probabilities (sample): {probabilities[:5]}")
                print(f"[ED {self.ed_index}, Cache {cache_index}] CDP - Redundant video indices: {random_videos_indices}")
                print(f"[ED {self.ed_index}, Cache {cache_index}] Redundant action (sample): {redundant_action[:5]}")
        else:
            redundant_action = action
            print(f"[ED {self.ed_index}, Cache {cache_index}] No noise added, redundant_action equals action: {redundant_action[:5]}")

        return redundant_action

    def update_cache(self, cache_index, redundant_request, ed_slot_trace):
        action = np.zeros(shape=self.content_num,dtype=np.int8)
        c_k = 0
        hits = 0
        cache = self.cache_list[cache_index]
        print(f"[ED {self.ed_index}, Cache {cache_index}] Initial cache contents: {list(cache.keys())}")

        if self.fetch_policy in ["PPVF"]:
            action[list(set(ed_slot_trace['i']))] = 1
            redundant_request_indices = np.argwhere(redundant_request==1)[:,0]
            action[redundant_request_indices] = 1
            print(f"[ED {self.ed_index}, Cache {cache_index}] Real requests: {list(set(ed_slot_trace['i']))}")
            print(f"[ED {self.ed_index}, Cache {cache_index}] Redundant requests: {redundant_request_indices}")
            print(f"[ED {self.ed_index}, Cache {cache_index}] Combined action (sample): {action[:5]}")

            for video in cache.keys():
                action[video] = 0
                cache[video] = self.density[video]
            request_indices = np.argwhere(action==1)[:,0]
            for video,den_i in zip(request_indices,self.density[request_indices]):
                if len(cache) >= self.c_e[cache_index]:
                    min_index = min(cache, key=cache.get)
                    print(f"[ED {self.ed_index}, Cache {cache_index}] Cache full, min density video: {min_index} (density: {cache[min_index]})")
                    if den_i > cache[min_index]:
                        del cache[min_index]
                        cache[video] = den_i
                        print(f"[ED {self.ed_index}, Cache {cache_index}] Replaced {min_index} with {video} (density: {den_i})")
                    else:
                        print(f"[ED {self.ed_index}, Cache {cache_index}] Video {video} (density: {den_i}) not added, below min density")
                else:
                    cache[video] = den_i
                    print(f"[ED {self.ed_index}, Cache {cache_index}] Added video {video} (density: {den_i}) to cache")
        else:
            raise ValueError(f"not such fetch policy: {self.fetch_policy}")
        self.actions[cache_index] += action
        print(f"[ED {self.ed_index}, Cache {cache_index}] Updated cache contents: {list(cache.keys())}")
        return hits

    def get_noise_paras(self):
        if self.fetch_policy in ["PPVF"]:
            return (self.n, self.sum_XY, self.sum_X,self.sum_X_square)
        else:
            raise ValueError("not this fetch_policy {}.".format({self.fetch_policy}))

    def update_b(self, cache_index, remained_b):
        self.remained_b[cache_index,:] = remained_b

    def update_noise_paras(self, updated_buffer):
        if self.if_noise == True:
            if  self.fetch_policy in ["PPVF"]:
                self.n = updated_buffer[0]
                self.sum_XY = updated_buffer[1]
                self.sum_X = updated_buffer[2]
                self.sum_X_square = updated_buffer[3]
            else:
                raise ValueError(f"not this fetch_policy {self.fetch_policy} or {self.fetch_policy} can not add noise")
        else:
            raise ValueError("No noise is added, so no need to update the noise parameters")


