import torch

def train_one_ep(optimizer, criterion, data_loader, model,device):
    """
    Trains a PyTorch model for one epoch.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        criterion: The loss function to use for training.
        data_loader (torch.utils.data.DataLoader): The data loader to use for training.
        model: The PyTorch model to train.
        
    Returns:
        float: The average loss of the training loop.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)
    for i, (inputs, targets) in enumerate(data_loader):
        inputs, targets=inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        ''' followed line used for standard loss! '''
        # loss = criterion(outputs, targets)
        
        ''' followed line used for soft_dtw! '''
        loss = criterion(outputs.unsqueeze(1), targets.unsqueeze(1))
        loss=loss.mean()


        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print('output_pre: {}'.format(outputs))
    print('target: {}'.format(targets))
    
    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss

def evaluate_one_ep(criterion, data_loader, model,device):
    """
    Evaluates a PyTorch model.
    
    Args:
        criterion: The loss function to use for evaluation.
        data_loader (torch.utils.data.DataLoader): The data loader to use for evaluation.
        model: The PyTorch model to evaluate.
        
    Returns:
        float: The average loss of the evaluation loop.
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(data_loader)
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets=inputs.to(device), targets.to(device)
            outputs = model(inputs)

            ''' followed line used for standard loss! '''
            # loss = criterion(outputs, targets)
            
            ''' followed line used for soft_dtw! '''
            loss = criterion(outputs.unsqueeze(1), targets.unsqueeze(1))
            loss=loss.mean()

            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss

import torch
import torch.nn as nn
# Define the momentum encoder
class MomentumEncoder(nn.Module):
    def __init__(self,MainModel, alpha=0.999):
        super(MomentumEncoder, self).__init__()
        self.alpha = alpha
        self.encoder = MainModel

    def forward(self, x):
        # Extract features from the momentum encoder
        x = self.encoder(x)  # (B, seq_len, D)
        return x

    def update_momentum_encoder(self, main_encoder):
        # Update the momentum encoder by EMA (Exponential Moving Average)
        for param_q, param_k in zip(main_encoder.parameters(), self.parameters()):
            param_k.data = param_k.data * self.alpha + param_q.data * (1 - self.alpha)

import torch.nn.functional as F

def InfonceLoss(x1, x2, temperature=1.0):
 # 计算相似度得分矩阵
    S = torch.mm(x1, x2.T) / temperature
    S = F.softmax(S, dim=-1)

    # 计算 InfoNCE 损失
    N = x1.size(0)
    diag_idx = torch.arange(N).to(x1.device)
    loss = -torch.log(S[diag_idx, diag_idx] / S.sum(dim=-1)+1e-5).mean()

    return loss
    
# Define the MOCO model
class MOCO(nn.Module):
    def __init__(self,seq_len,feature_dim,pre_len,main_model,moco_model,pred_los,
                 contras_los=InfonceLoss, temperature=0.07,lambda_2=0.45, lamba_1=0.55,embed_dim=128):
        super(MOCO, self).__init__()
        # Define the main model and momentum encoder
        self.main_encoder = main_model
        self.momentum_encoder = moco_model
        # Copy the parameters of the main encoder to the momentum encoder
        # self.momentum_encoder.load_state_dict(self.main_encoder.state_dict())
        # Set the momentum encoder to be non-trainable
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False

        self.pred_los=pred_los
        self.contras_los=contras_los
        self.lamba_1=lamba_1
        self.lambda_2=lambda_2
        self.temperature=temperature

        self.pre_head=nn.Linear(seq_len*feature_dim,pre_len)

    def forward(self, x_i, x_j,pred):
        # Compute the features for the anchor
        batch_idx,_,_=x_i.shape
        z_i = self.main_encoder(x_i)  # (B, seq_len, D)
        # Compute the features for the positive examples
        with torch.no_grad():
            self.momentum_encoder.update_momentum_encoder(self.main_encoder)
            z_pos = self.momentum_encoder(x_j)  # (B, seq_len, D)
        # Compute the contrastive loss
        contastive_loss = self.contras_los(z_i,z_pos,self.temperature)
        target=self.pre_head(z_i)

        pred_loss=self.pred_los(target,pred)


        if batch_idx ==0:
            tar=target.detach()
            print('pred:.{}'.format(tar.squeeze(1)[:8]))
            print('target:{}'.format(pred[:8]))

        loss=self.lamba_1*pred_loss+self.lambda_2*contastive_loss
        
        return loss.mean()

import torch

def moco_train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    epoch_loss = 0.0
    for x1, x2, y in dataloader:
        # Move the inputs and targets to the device
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward + backward + optimize
        loss = model(x1, x2, y)

        loss.backward()
        optimizer.step()
        # Compute the epoch loss
        epoch_loss += loss.item() * len(x1)
    epoch_loss /= len(dataloader.dataset)
    # print("Epoch: {}, Training Loss: {:.4f}".format(epoch+1, epoch_loss))
    return epoch_loss


def moco_evaluate_one_epoch(model, dataloader, device, epoch):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for x1, x2, y in dataloader:
            # Move the inputs and targets to the device
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            # Forward pass
            loss = model(x1, x2, y)
            # Compute the epoch loss
            epoch_loss += loss.item() * len(x1)
    epoch_loss /= len(dataloader.dataset)
    # print("Epoch: {}, Validation Loss: {:.4f}".format(epoch+1, epoch_loss))
    return epoch_loss
