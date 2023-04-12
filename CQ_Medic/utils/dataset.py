from torch.utils.data import DataLoader,Dataset,random_split
import numpy as np
import pandas as pd
import torch
from utils.process import ScaleData

class Hand_foot_data_with_weath(Dataset):
    def __init__(
            self,
            root_dir_1='./data/case_inf.csv',
            root_dir_2='./data/wea_proed_inf.csv',
            data_len=2512,
            
            cut_len=365,  # use previous data to predict future data
            slide_win_size=0,  # the len of the sequence we try to pred
            transform=None
            ):
        super(Hand_foot_data_with_weath,self).__init__()
        case_data=pd.read_csv(root_dir_1)
        weather_data=pd.read_csv(root_dir_2)

        self.n_cases=torch.tensor(np.array(case_data['cases'][:]),dtype=torch.float32)

        n_cases=torch.tensor(np.array(case_data['cases'][:data_len]),dtype=torch.float32).unsqueeze(1)
        weather_inf=torch.tensor(np.array(weather_data.iloc[:data_len,1:7]),dtype=torch.float32)

        data=torch.cat([n_cases, weather_inf],dim=-1)
        print(data.shape)
        data=ScaleData(data.unsqueeze(1),'quantile','both',42)

        
        self.features=torch.tensor(data,dtype=torch.float32).squeeze(1)
        print(self.features.shape)
        
        self.data_len=data_len
        self.slide_win_size=slide_win_size
        self.cut_len=cut_len
    
    def __len__(self):
        return int(self.data_len-self.cut_len-1)
    
    def __getitem__(self, index) :
        features=self.features[index:self.cut_len+index, :]
        if self.slide_win_size!=0:
            labels=self.n_cases[self.cut_len+index+1 :  self.cut_len+index+1+self.slide_win_size]
        else:
            labels=self.n_cases[self.cut_len+index+1]

        return features,labels
    

def get_h_f_dataloader(dataset,batch_size,num_workers):

    h_f_dataloader= DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        )
    return h_f_dataloader


def train_val_test_split(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Split a PyTorch dataset into training, validation, and testing sets.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        train_ratio (float): The ratio of examples to include in the training set.
        val_ratio (float): The ratio of examples to include in the validation set.
        test_ratio (float): The ratio of examples to include in the test set.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets.
    """
    # Compute the sizes of each dataset
    num_examples = len(dataset)
    num_train_examples = int(train_ratio * num_examples)
    num_val_examples = int(val_ratio * num_examples)
    num_test_examples = int(test_ratio * num_examples)

    # Adjust the size of one dataset if necessary
    if num_train_examples + num_val_examples + num_test_examples != num_examples:
        num_train_examples += num_examples - (num_train_examples + num_val_examples + num_test_examples)

    # Use the random_split function to split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset,
                                                             [num_train_examples, num_val_examples, num_test_examples])

    return train_dataset, val_dataset, test_dataset

