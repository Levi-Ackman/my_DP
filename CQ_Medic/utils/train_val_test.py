from torch.utils.data import DataLoader,Dataset,random_split
import numpy as np
import pandas as pd
import torch
from utils.process import ScaleData

def get_h_f_dataloader(dataset,batch_size,num_workers):

    h_f_dataloader= DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        )
    return h_f_dataloader

def get_data(root_dir_1='./data/case_inf.csv',
            root_dir_2='./data/wea_proed_inf.csv'):
    
    case_data=pd.read_csv(root_dir_1)
    weather_data=pd.read_csv(root_dir_2)

    n_cases_1=torch.tensor(np.array(case_data['cases'][:1827]),dtype=torch.float32).unsqueeze(1)
    weather_inf_1=torch.tensor(np.array(weather_data.iloc[:1827,1:2]),dtype=torch.float32)

    n_cases_2=torch.tensor(np.array(case_data['cases'][2193:2152]),dtype=torch.float32).unsqueeze(1)
    weather_inf_2=torch.tensor(np.array(weather_data.iloc[2193:2152,1:2]),dtype=torch.float32)

    case_data=torch.cat([n_cases_1, n_cases_2],dim=0)
    weather_data=torch.cat([weather_inf_1,weather_inf_2],dim=0)

    data=torch.cat([case_data, weather_data],dim=-1)

    targets=case_data.squeeze(1)

    return  data,targets

class h_d_train(Dataset):
    def __init__(
            self,  
            cut_len=365,  # use previous data to predict future data
            slide_win_size=0,  # the len of the sequence we try to pred
            split_ratio=0.8, # decide the len of train set
            normalize_type='quantile',
            normalize_dire='both',
            normalize_seed=42,
            transform=True,
            ):
        data,targets=get_data()

        ori_len=len(data)
        data_len=int(split_ratio*ori_len)

        data=data[:data_len,:]
        # targets=targets[:data_len]
        if transform:
            data=ScaleData(data.unsqueeze(1),normalize_type,normalize_dire,normalize_seed)

        self.features=torch.tensor(data,dtype=torch.float32).squeeze(1)
        self.targets=targets
        
        self.data_len=data_len
        self.slide_win_size=slide_win_size
        self.cut_len=cut_len
    
    def __len__(self):
        return int(self.data_len-self.cut_len-1)
    
    def __getitem__(self, index) :
        features=self.features[index:self.cut_len+index, :]
        if self.slide_win_size!=1:
            targets=self.targets[self.cut_len+index :  self.cut_len+index+self.slide_win_size]
        else:
            targets=self.targets[self.cut_len+index-1]
            # targets=self.targets[self.cut_len+index]

        return features,targets


class h_d_val(Dataset):
    def __init__(
            self,  
            cut_len=365,  # use previous data to predict future data
            slide_win_size=0,  # the len of the sequence we try to pred
            train_split_ratio=0.8, # decide the len of train set
            val_split_ratio=0.1,
            normalize_type='quantile',
            normalize_dire='both',
            normalize_seed=42,
            transform=True,
            ):
        data,targets=get_data()

        ori_len=len(data)
        data_len=int(val_split_ratio*ori_len)
        start_point=int(train_split_ratio*ori_len)

        data=data[start_point:start_point+data_len,:]
        # targets=targets[start_point:start_point+data_len]

        if transform:
           data=ScaleData(data.unsqueeze(1),normalize_type,normalize_dire,normalize_seed)

        self.features=torch.tensor(data,dtype=torch.float32).squeeze(1)
        self.targets=targets
        
        self.data_len=data_len
        self.slide_win_size=slide_win_size
        self.cut_len=cut_len
    
    def __len__(self):
        return int(self.data_len-self.cut_len-1)
    
    def __getitem__(self, index) :
        features=self.features[index:self.cut_len+index, :]
        if self.slide_win_size!=1:
            targets=self.targets[self.cut_len+index :  self.cut_len+index+self.slide_win_size]
        else:
            targets=self.targets[self.cut_len+index]

        return features,targets
    
class h_d_test(Dataset):
    def __init__(
            self,  
            cut_len=365,  # use previous data to predict future data
            slide_win_size=0,  # the len of the sequence we try to pred
            train_split_ratio=0.8, # decide the len of train set
            val_split_ratio=0.1,
            test_split_ratio=0.1,
            normalize_type='quantile',
            normalize_dire='both',
            normalize_seed=42,
            transform=True,
            ):
        data,targets=get_data()

        ori_len=len(data)
        data_len=int(test_split_ratio*ori_len)
        start_point=int(train_split_ratio*ori_len)+int(val_split_ratio*ori_len)

        data=data[start_point:start_point+data_len,:]
        
        # targets=targets[start_point:start_point+data_len]

        if transform:
           data=ScaleData(data.unsqueeze(1),normalize_type,normalize_dire,normalize_seed)

        self.features=torch.tensor(data,dtype=torch.float32).squeeze(1)
        self.targets=targets
        
        self.data_len=data_len
        self.slide_win_size=slide_win_size
        self.cut_len=cut_len
    
    def __len__(self):
        return int(self.data_len-self.cut_len-1)
    
    def __getitem__(self, index) :
        features=self.features[index:self.cut_len+index, :]
        if self.slide_win_size!=1:
            targets=self.targets[self.cut_len+index :  self.cut_len+index+self.slide_win_size]
        else:
            targets=self.targets[self.cut_len+index]

        return features,targets
    
from utils import augmantation as aug
class moco_h_d_train(Dataset):
    def __init__(
            self,  
            cut_len=365,  # use previous data to predict future data
            slide_win_size=0,  # the len of the sequence we try to pred
            split_ratio=0.8, # decide the len of train set
            normalize_type='quantile',
            normalize_dire='both',
            normalize_seed=42,
            transform=True,
            aug=aug
            ):
        data,targets=get_data()

        ori_len=len(data)
        data_len=int(split_ratio*ori_len)

        data=data[:data_len,:]
        # targets=targets[:data_len]
        if transform:
            data=ScaleData(data.unsqueeze(1),normalize_type,normalize_dire,normalize_seed)

        self.features=torch.tensor(data,dtype=torch.float32).squeeze(1)
        self.targets=targets
        
        self.data_len=data_len
        self.slide_win_size=slide_win_size
        self.cut_len=cut_len
        self.aug=aug
    
    def __len__(self):
        return int(self.data_len-self.cut_len-1)
    
    def __getitem__(self, index) :
        features=self.features[index:self.cut_len+index, :]
        if self.slide_win_size!=1:
            targets=self.targets[self.cut_len+index :  self.cut_len+index+self.slide_win_size]
        else:
            targets=self.targets[self.cut_len+index]
            
        aug_feature=self.aug.window_warp(features.unsqueeze(0))
        aug_feature=aug_feature.squeeze(0)
        
        return features, aug_feature, targets
    
class moco_h_d_val(Dataset):
    def __init__(
            self,  
            cut_len=365,  # use previous data to predict future data
            slide_win_size=0,  # the len of the sequence we try to pred
            train_split_ratio=0.8, # decide the len of train set
            val_split_ratio=0.1,
            normalize_type='quantile',
            normalize_dire='both',
            normalize_seed=42,
            transform=True,
            aug=aug
            ):
        data,targets=get_data()

        ori_len=len(data)
        data_len=int(val_split_ratio*ori_len)
        start_point=int(train_split_ratio*ori_len)

        data=data[start_point:start_point+data_len,:]
        # targets=targets[start_point:start_point+data_len]

        if transform:
           data=ScaleData(data.unsqueeze(1),normalize_type,normalize_dire,normalize_seed)

        self.features=torch.tensor(data,dtype=torch.float32).squeeze(1)
        self.targets=targets
        
        self.data_len=data_len
        self.slide_win_size=slide_win_size
        self.cut_len=cut_len

        self.aug=aug
    
    def __len__(self):
        return int(self.data_len-self.cut_len-1)
    
    def __getitem__(self, index) :
        features=self.features[index:self.cut_len+index, :]
        if self.slide_win_size!=1:
            targets=self.targets[self.cut_len+index :  self.cut_len+index+self.slide_win_size]
        else:
            targets=self.targets[self.cut_len+index]

        aug_feature=self.aug.window_warp(features.unsqueeze(0))
        aug_feature=aug_feature.squeeze(0)

        return features,aug_feature,targets
    
class moco_h_d_test(Dataset):
    def __init__(
            self,  
            cut_len=365,  # use previous data to predict future data
            slide_win_size=0,  # the len of the sequence we try to pred
            train_split_ratio=0.8, # decide the len of train set
            val_split_ratio=0.1,
            test_split_ratio=0.1,
            normalize_type='quantile',
            normalize_dire='both',
            normalize_seed=42,
            transform=True,
            aug=aug,
            ):
        data,targets=get_data()

        ori_len=len(data)
        data_len=int(test_split_ratio*ori_len)
        start_point=int(train_split_ratio*ori_len)+int(val_split_ratio*ori_len)

        data=data[start_point:start_point+data_len,:]
        
        # targets=targets[start_point:start_point+data_len]

        if transform:
           data=ScaleData(data.unsqueeze(1),normalize_type,normalize_dire,normalize_seed)

        self.features=torch.tensor(data,dtype=torch.float32).squeeze(1)
        self.targets=targets
        
        self.data_len=data_len
        self.slide_win_size=slide_win_size
        self.cut_len=cut_len

        self.aug=aug
    
    def __len__(self):
        return int(self.data_len-self.cut_len-1)
    
    def __getitem__(self, index) :
        features=self.features[index:self.cut_len+index, :]
        if self.slide_win_size!=1:
            targets=self.targets[self.cut_len+index :  self.cut_len+index+self.slide_win_size]
        else:
            targets=self.targets[self.cut_len+index]
        
        aug_feature=self.aug.window_warp(features.unsqueeze(0))
        aug_feature=aug_feature.squeeze(0)

        return features,aug_feature,targets