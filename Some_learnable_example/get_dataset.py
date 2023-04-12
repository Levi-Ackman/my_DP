from collections import Counter,defaultdict
import numpy as np
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler
import torch

class CharacterDataset(Dataset):
    """Custom dataset.
    Parameters
    ----------
    text : str
        Input text that will be used to create the entire database.
    window_size : int
        Number of characters to use as input features.
    vocab_size : int
        Number of characters in the vocabulary. Note that the last character
        is always reserved for a special "~" out-of-vocabulary character.
    Attributes
    ----------
    ch2ix : defaultdict
        Mapping from the character to the position of that character in the
        vocabulary. Note that all characters that are not in the vocabulary
        will get mapped into the index `vocab_size - 1`.
    ix2ch : dict
        Mapping from the character position in the vocabulary to the actual
        character.
    vocabulary : list
        List of all characters. `len(vocabulary) == vocab_size`.
    """

    def __init__(self, text, window_size=1, vocab_size=50):
        self.text=text.replace('\n','')
        self.window_size=window_size
        self.ch2ix=defaultdict(lambda:vocab_size-1)

        '''only pick the most_common characters of the first (vocab_size-1) char '''
        most_common_ch2ix={
            x[0]:i
            for i ,x in enumerate(Counter(self.text).most_common()[:(vocab_size-1)])
        }
        self.ch2ix.update(most_common_ch2ix)
        self.ch2ix["~"] = vocab_size - 1

        self.ix2ch={
            v:k
            for  k,v in self.ch2ix.items()
        }
        self.vocabulary=[
            self.ix2ch[i]
            for i in range(vocab_size)
            ]

    def __len__(self):
        return len(self.text)-self.window_size ## here we use the entire text to generate the new text,but the real length have to be minus by window_size
    
    def __getitem__(self, ix):
        '''each slice of the text with the size of window_size will be transfered to tensor '''
        ## x is the input of the neural network
        ## we take a slice of the ori_text with window_size(let's set it to 3)
        # like 'kob', and then transfer it to '245'
        # where the 2 is the exact position/index of the char 'k' in the ch2ix dictionary

        x=torch.LongTensor(
            [
                self.ch2ix[c]
                for c in self.text[ix:ix+self.window_size]
            ]
            )
        ## the y/target_text is what we wanna the neural network to generate
        ## namely the new character
        ## for instance :
        ## we wanna generate: sample 
        # with window_size=3, and we input sam
        # so we wanna the neural-network generate p
        # i.e. : sam->p  amp->l mpl->e
        y=self.ch2ix[
            self.text[
                ix+self.window_size
                ]
            ]
        return x,y
