import torch
import torch.nn as nn

'''Quantile Loss: This loss function is similar to the Huber loss, but is based on the deviation of the predicted quantile from the true quantile.'''
class QuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)

        losses = []
        for i, q in enumerate(self.quantiles):
            error = target - preds[:, i]
            loss = torch.max((q-1) * error, q * error)
            losses.append(loss)

        return torch.mean(torch.sum(torch.stack(losses, dim=1), dim=1))

def quantile_loss(y_pred, y_true, quantiles=[0.1, 0.9]):
    assert not y_true.requires_grad
    assert y_pred.size(0) == y_true.size(0)
    assert y_pred.size(1) == len(quantiles)

    errors = y_true - y_pred
    loss = torch.max((quantiles-1) * errors, quantiles * errors)
    return torch.mean(loss)

    

'''Symmetric Mean Absolute Percentage Error (SMAPE): This loss function measures the percentage difference 
between the predicted and true values, but takes into account the scale of the variable being predicted.'''
def smape_loss(y_pred, y_true):
    numerator = torch.abs(y_pred - y_true)
    denominator = torch.abs(y_pred) + torch.abs(y_true)
    loss = 200.0 / len(y_pred) * torch.sum(torch.divide(numerator, denominator))
    return loss


'''Weighted Mean Absolute Percentage Error (WMAPE): This loss function is similar to SMAPE 
but allows for weighting of different time points based on their importance.'''
def wmape_loss(y_pred, y_true, weights=0.25):
    numerator = torch.abs(y_pred - y_true)
    denominator = y_true * weights
    loss = 100.0 / torch.sum(denominator) * torch.sum(torch.divide(numerator, denominator))
    return loss


'''Mean Absolute Scaled Error (MASE): This loss function measures the ratio of the MAE of the forecast to the MAE of a naive baseline forecast. 
It is useful for comparing the accuracy of different models.'''
def mase_loss(y_pred, y_true, n_seasons=1):
    diff = torch.abs(y_pred - y_true)
    mean_abs_diff_naive = torch.mean(torch.abs(y_true[n_seasons:] - y_true[:-n_seasons]))
    loss = torch.mean(diff) / mean_abs_diff_naive
    return loss

'''Log-Cosh Loss: This loss function is a smoothed version of the MSE loss function, 
and is useful for handling outliers or extreme values in the data'''
def log_cosh_loss(y_pred, y_true):
    x = y_pred - y_true
    loss = torch.log(torch.cosh(x)+1e-5)
    return torch.mean(loss)

'''Pinball Loss: This loss function is commonly used in quantile regression, 
which is useful for estimating conditional quantiles of a distribution. 
It measures the deviation between the predicted and true quantiles.'''
def pinball_loss(y_pred, y_true, tau=2.25):
    residual = y_true - y_pred
    loss = torch.mean(torch.max((tau * residual), ((tau - 1) * residual)))
    return loss

'''Huber Loss: This loss function is a combination of the MAE and MSE loss functions, 
and is useful when the data contains outliers or extreme values.'''
def huber_loss(y_pred, y_true, delta=1.0):
    residual = torch.abs(y_true - y_pred)
    condition = torch.logical_or(residual < delta, torch.isnan(residual))
    small_res = 0.5 * residual[condition]**2
    large_res = delta * (residual[~condition] - delta / 2)
    loss = torch.mean(torch.cat((small_res, large_res)))
    return loss

class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        mse = nn.MSELoss()(torch.log(y_pred + 1), torch.log(y_true + 1))
        return mse
