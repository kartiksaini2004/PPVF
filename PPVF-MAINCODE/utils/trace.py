import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import os

class Trace(object):
    def __init__(self, cfg):
        # UIT_header = pd.read_csv(os.path.join(cfg.data_path,'UIT_ED.csv'), header = None)
        # UIT_header.columns = ['ed', 'i', 'day', 'type', 'country','province', 'city', 'isp', 'time']
        UIT_header = pd.read_csv(os.path.join(cfg.data_path,'UIT_ED_25_I_10373.csv'), header = None)
        UIT_header.columns = ['u', 'i', 'day', 'type', 'ed', 'time']

        # Determine the level
        if cfg.train_level in ['centralized','province','country','city','isp']:
            UIT_header.loc[:,'ed'] = -1
            
        elif cfg.train_level == 'ed':
            pass

        # UIT_header = UIT_header.drop(['type', 'country','province', 'city', 'isp'], axis=1)
        UIT_header = UIT_header.drop(['type','u'], axis=1)
        UIT_header = UIT_header[['ed', 'i', 'day', 'time']]

        #self.UIT_header = UIT_header.drop(['type', 'country','province','isp'], axis=1)

        SEED = 0
        random.seed(SEED)
        # Delete it according to the columns
        # Randomly select the specified element from the set
        origin_i = list(set(UIT_header['i']))
        num_filter_i = int(cfg.dataset_percent * len(set(UIT_header['i'])))  # Specify the number of elements to be selected
        filter_elements = set(random.sample(origin_i, num_filter_i))
        self.UIT_filtered = UIT_header[UIT_header['i'].isin(filter_elements)].copy()
        self.UIT_filtered.loc[:,'time'] =  self.UIT_filtered['time'] // cfg.slot_interval
        encoder_label_content = LabelEncoder()
        self.UIT_filtered.loc[:,'i'] = encoder_label_content.fit_transform(self.UIT_filtered['i'].values)

        #Divide the data set
        self.ED_index = set(self.UIT_filtered['ed'])
        self.ED_trace = {}
        self.ED_trace_count ={}
        UIT_group = self.UIT_filtered.groupby(self.UIT_filtered['ed'])

        for index in self.ED_index:
            self.ED_trace[index] = UIT_group.get_group(index).reset_index(drop=True)
            self.ED_trace_count[index] = len(self.ED_trace[index])
        
        self.content_num = len(set(self.UIT_filtered['i']))
        self.ED_num = len(self.ED_index)

        print("data process success, content number: ",self.content_num, "ed number: ",self.ED_num)

    def get_ed_trace(self, ed_index):
        return self.ED_trace[ed_index]
    
    def get_all_trace(self):
        return self.UIT_filtered