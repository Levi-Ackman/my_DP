import torch

def dtw_loss(ts1, ts2):
    # Compute the DTW distance between the two time series
    nrows = len(ts1)
    ncols = len(ts2)
    cost_matrix = torch.zeros((nrows, ncols), device=ts1.device)

    # Fill the first row and column with large values
    for i in range(nrows):
        cost_matrix[i, 0] = float('inf')
    for j in range(ncols):
        cost_matrix[0, j] = float('inf')
    cost_matrix[0, 0] = 0

    # Fill the rest of the matrix
    for i in range(1, nrows):
        for j in range(1, ncols):            
            cost_matrix[i, j] = abs(ts1[i] - ts2[j]) + torch.min(cost_matrix[i-1, j], 
                                    cost_matrix[i, j-1], cost_matrix[i-1, j-1])

    # Return the final DTW cost
    dtw_cost = cost_matrix[-1,-1]

    # Normalize the cost by the length of the time series
    norm_cost = dtw_cost / (nrows + ncols)

    # Return the normalized cost as the DTW loss
    return norm_cost
