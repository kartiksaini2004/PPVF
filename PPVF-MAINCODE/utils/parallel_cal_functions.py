import os
import numpy as np
from global_config import GLOBAL_PATH, DATA_PATH

#Parallel initialization of commonly used tables in hpp prediction models
def init_ed_kernel(ed_index, cfg_dict, eventArray):
    sumKernel  = np.zeros((cfg_dict['content_num'],cfg_dict['all_t']), dtype = np.float64)
    sumI1      = np.zeros((cfg_dict['content_num'],cfg_dict['all_t']), dtype = np.float64)
    eventNum_t = np.zeros((cfg_dict['content_num'],cfg_dict['all_t']), dtype = np.int32)

    items = eventArray[:, 1]  # Get a list of items for all events
    taus = eventArray[:, 3]   # Get tau list of all events
    np.add.at(eventNum_t, (items, taus), 1) # 使用numpy的bincount函数来一次性增加事件数量

    delta = cfg_dict['PPF_delta']

    # kernel_name = f"tmp_data/kernel_data/{cfg_dict['train_level']}-ed_{ed_index}-timeSlot_{cfg_dict['slot_interval']}-delta_{delta}-content_{cfg_dict['content_num']}.npy"
    # integral_name = f"tmp_data/integral_data/{cfg_dict['train_level']}-ed_{ed_index}-timeSlot_{cfg_dict['slot_interval']}-delta_{delta}-content_{cfg_dict['content_num']}.npy"
    # kernel_path = os.path.join(GLOBAL_PATH,kernel_name)
    # integral_path = os.path.join(GLOBAL_PATH,integral_name)

    # # Get the folder path
    # folder_path = os.path.dirname (kernel_path)
    # # If the path of the folder does not exist, create it
    # if not os.path.exists (folder_path):
    #     os.makedirs (folder_path)

    # # Get the folder path
    # folder_path = os.path.dirname(integral_path)
    # # If the folder path does not exist, create it
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)

    # if cfg_dict["ifLoadKernel"] and os.path.exists(kernel_path) and os.path.exists(integral_path):
    #     sumKernel = np.load(kernel_path)
    #     sumI1 = np.load(integral_path)
    # else:
    #     #增量计算核函数对应表
    #     for tau in range(1,cfg_dict['all_t']):
    #         sumKernel[:,tau] = (eventNum_t[:,tau-1] + sumKernel[:,tau-1]) * np.exp(-delta)
    #         new_event_integral = eventNum_t[:,tau-1] * ((1 - np.exp(- delta * cfg_dict['update_interval'])) / delta)
    #         sumI1[:,tau] = (new_event_integral + sumI1[:,tau-1]) * np.exp(-delta) 
    #     if not os.path.exists(kernel_path):
    #         np.save(kernel_path,sumKernel)
    #     if not os.path.exists(integral_path):
    #         np.save(integral_path,sumI1)
    #增量计算核函数对应表
    for tau in range(1,cfg_dict['all_t']):
        sumKernel[:,tau] = (eventNum_t[:,tau-1] + sumKernel[:,tau-1]) * np.exp(-delta)
        new_event_integral = eventNum_t[:,tau-1] * ((1 - np.exp(- delta * cfg_dict['update_interval'])) / delta)
        sumI1[:,tau] = (new_event_integral + sumI1[:,tau-1]) * np.exp(-delta) 
    return ed_index, (eventNum_t, sumKernel, sumI1)







def get_redundant_request(ed_index, cache_index, origin_density,noise_paras, online_paras, cfg_dict):
    #normalization
    def scale_value(value, input_min, input_max, output_min, output_max):
        # Enter the interval range
        input_range = input_max - input_min
        if input_range == 0:
            return value
        # 输出区间范围
        output_range = output_max - output_min
        # 计算映射后的值
        scaled_value = ((value - input_min) / input_range) * output_range + output_min
        return scaled_value

    def get_th(remained_b): # 计算阈值（基于阈值的在线算法相关）
        if remained_b < cfg_dict["B0"]:
            return cfg_dict["Low_bound"]
        else:
            return (((cfg_dict["Up_bound"] * np.e) / cfg_dict["Low_bound"])**remained_b) * (cfg_dict["Low_bound"] / np.e)
    # 在线计算皮尔逊相关系数
    def calculate_pearson_correlation(data,noise_paras,content_num): 
        #data         = data[:,np.newaxis]
        n            = noise_paras[0]
        sum_XY       = noise_paras[1]
        sum_X        = noise_paras[2]
        sum_X_square = noise_paras[3]
        n            += 1
        sum_XY       += data @ data.T
        sum_X        += data
        sum_X_square += data * data
        if n>1:
            assert (n*sum_X_square>=sum_X * sum_X).all(), [n]
            sigma_X = np.sqrt(n * sum_X_square - (sum_X * sum_X))
            denominator =  sigma_X @ sigma_X.T
            numerator = n * sum_XY - (sum_X @ sum_X.T) 
            psi = numerator / denominator
        else:
            psi = np.eye(content_num,dtype=np.float64)
        return psi, (n,sum_XY,sum_X,sum_X_square)
    SEED = 0
    np.random.seed(SEED)
    #scaled_density = origin_density
    #先进行归一化

    f_e        = online_paras[0]
    remained_b = online_paras[1]

    scaled_density = scale_value(origin_density, origin_density.min(), origin_density.max(), cfg_dict["Low_bound"] * cfg_dict["epsilon"], cfg_dict["Up_bound"] * cfg_dict["epsilon"]) 

    action = np.zeros(shape=cfg_dict["content_num"],dtype=np.int8) #当前slot的缓存action
    f_k = 0 
    if cfg_dict["fetch_policy"] == "PPVF": #infocom2022 + HPP_PAC
        sampled_indexes = np.random.choice(range(cfg_dict["content_num"]), size = cfg_dict["content_num"], replace = False) # 无放回随机采样
        for i in sampled_indexes:
            if f_k + 1 > f_e:
                break
            # elif i in cache: #在缓存中就不请求
            #     pass
            #     f_k += 1
            #     action[i] = 1
            elif remained_b[i] <= cfg_dict["B0"]: #低于阈值，必然请求
                f_k += 1
                action[i] = 1
                remained_b[i] = remained_b[i] + cfg_dict["epsilon"] / cfg_dict["xi"]  
            elif (scaled_density[i] / cfg_dict["epsilon"] > get_th(remained_b[i])) and (cfg_dict["epsilon"] <= (1- remained_b[i]) * cfg_dict["xi"]):
                f_k += 1
                action[i] = 1
                remained_b[i] = remained_b[i] + cfg_dict["epsilon"] / cfg_dict["xi"] 
            else:
                action[i] = 0
    else:
        raise ValueError(f"not such fetch policy: {cfg_dict['fetch_policy']} or {cfg_dict['fetch_policy']} not privacy budget requirement")
    

    action_indices  = np.argwhere(action==1)[:,0] #获取有隐私预算分配的视频index
    redundant_action = np.zeros(shape=cfg_dict["content_num"],dtype=np.int8) # 冗余请求决策

    psi,updated_noise_paras = calculate_pearson_correlation(origin_density,noise_paras,cfg_dict["content_num"]) #根据计算相关性
        
    psi[np.abs(psi) < 0.95] = 0 #根据阈值修改相关性

    if len(action_indices)>0:
        action_density  = origin_density[action_indices]
        #计算每个视频的敏感度
        sensitivity = psi[action_indices,action_indices] @ action_density.T
        #全局敏感度
        sen_c = sensitivity.max()
        # 计算指数机制的概率
        probabilities   = np.exp(cfg_dict["epsilon"] * action_density / (2 * sen_c))
        probabilities  /= np.sum(probabilities)  # 归一化到概率分布
        random_videos_indices = np.random.choice(action_indices, size=len(action_indices), p=probabilities, replace=True)
        redundant_action[random_videos_indices] = 1

    return ed_index, cache_index, redundant_action, remained_b, updated_noise_paras

#并行获取最新的likelihood and gradient
def get_ED_likehood_gradient(ed_index, results, onEventArray, beta, p, q, eventNum_t, sumKernel, sumI1, delta, content_num, PAC_D, update_interval, t_o):
    omega = p @ q.T

    #计算强度函数
    def calIntensity(eventArray, beta, omega, p, q, sumKernel, content_num, PAC_D):
        content_indices = eventArray[:, 1]
        taus = eventArray[:, 3]
        # 计算强度函数
        Inten_tau = beta[content_indices] + np.sum(omega[content_indices, :] * sumKernel[:,taus].T, axis=1)
        sumIntensityTerm = np.sum(np.log(Inten_tau))
        betaGrad1 = np.zeros_like(beta)
        pGrad1 = np.zeros_like(p)
        qGrad1 = np.zeros_like(q)
        # 使用NumPy的矩阵操作计算梯度
        inv_Inten_tau = 1 / Inten_tau
        #先计算历史记录中item索引出现的次数，然后加权
        betaGrad1 = np.bincount(content_indices, weights=inv_Inten_tau, minlength=content_num)

        for d in range(PAC_D):
            tmp_pGrad1 = (q[:,d].T @ sumKernel[:,taus]).T / Inten_tau
            pGrad1[:,d] = np.bincount(content_indices, weights=tmp_pGrad1, minlength=content_num)
            #qGrad1[:,d] = sumKernel[:,taus] @ (p[content_indices,d] / Inten_tau)
            qGrad1[:,d] = np.sum((p[content_indices,d] * inv_Inten_tau) * sumKernel[:,taus] ,axis=1)
        return sumIntensityTerm, betaGrad1, pGrad1, qGrad1

    sumIntensityTerm, betaGrad1, pGrad1, qGrad1  = \
        calIntensity(onEventArray, beta, omega, p, q, sumKernel, content_num, PAC_D)

    #积分项第一项
    sumBetaTerm = update_interval * np.sum(beta) 

    #积分项第二项
    sumI2 = np.zeros(content_num, dtype = np.float64)
    for tau in range (t_o-update_interval, t_o):
        sumI2 += eventNum_t[:,tau] * (1 - np.exp(-delta*(t_o - tau))) / delta
    sumI_to = sumI1[:,t_o-update_interval] + sumI2
    sumOmegaTerm = np.sum( omega @ sumI_to )

    #总积分项
    sumIntegralTerm = sumBetaTerm + sumOmegaTerm

    #总目标
    obj = sumIntensityTerm - sumIntegralTerm

    ###-----以下计算梯度----###
    betaGrad = np.zeros_like(beta)
    pGrad    = np.zeros_like(p)
    qGrad    = np.zeros_like(q)

    #计算 beta 偏导数
    betaGrad = betaGrad1 - update_interval

    #计算 p q 偏导数
    expanded_sumI_to = np.repeat(sumI_to[:, np.newaxis], PAC_D, axis=1)
    pGrad2           = np.sum(q * expanded_sumI_to, axis=0)
    qGrad2           = np.zeros_like(q)
    sum_p            = np.sum(p,axis=0)
    for d in range(PAC_D):
        pGrad[:,d]  = pGrad1[:,d] - pGrad2[d]
        qGrad2[:,d] = sum_p[d] * sumI_to

    qGrad = qGrad1 - qGrad2

    results[ed_index] = -obj, -betaGrad, -pGrad, -qGrad

def get_ED_likehood_gradient_HPPSELF(ed_index, results, \
    onEventArray,\
    beta, omega, \
    eventNum_t, sumKernel, sumI1, delta, \
    content_num, update_interval, t_o):

    #计算强度函数
    content_indices = onEventArray[:, 1]
    taus = onEventArray[:, 3]
    # 计算强度函数
    Inten_tau = beta[content_indices] + omega[content_indices] * sumKernel[content_indices,taus]
    sumIntensityTerm = np.sum(np.log(Inten_tau))

    #积分项第一项
    sumBetaTerm = update_interval * np.sum(beta) 

    #积分项第二项
    sumI2 = np.zeros(content_num, dtype = np.float64)
    for tau in range (t_o-update_interval, t_o):
        sumI2 += eventNum_t[:,tau] * (1 - np.exp(-delta*(t_o - tau))) / delta
    sumI_to = sumI1[:,t_o-update_interval] + sumI2
    sumOmegaTerm = np.sum( omega * sumI_to )

    #总积分项
    sumIntegralTerm = sumBetaTerm + sumOmegaTerm

    #总目标
    obj = sumIntensityTerm - sumIntegralTerm

    ###-----以下计算梯度----###
    betaGrad   = np.zeros_like(beta)
    omegaGrad  = np.zeros_like(omega)
    betaGrad1  = np.zeros_like(beta)
    omegaGrad1 = np.zeros_like(omega)

    # 使用NumPy的矩阵操作计算梯度
    inv_Inten_tau = 1 / Inten_tau
    #先计算历史记录中item索引出现的次数，然后加权
    betaGrad1 = np.bincount(content_indices, weights=inv_Inten_tau, minlength=content_num)

    omegaGrad1_tmp = sumKernel[content_indices,taus] * inv_Inten_tau
    #先计算历史记录中item索引出现的次数，然后加权
    omegaGrad1 = np.bincount(content_indices, weights=omegaGrad1_tmp, minlength=content_num)

    #计算 beta 偏导数
    betaGrad = betaGrad1 - update_interval

    #计算 omega 偏导数
    omegaGrad = omegaGrad1 - sumI_to

    results[ed_index] = -obj, -betaGrad, -omegaGrad

def HRS_online_traning(ed_index,HPP,onlineBeginT,maxiter):
    HPP.onlineFit(onlineBeginT=onlineBeginT,maxiter=maxiter)
    return ed_index,HPP