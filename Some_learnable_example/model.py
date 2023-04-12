import torch
import torch.nn as nn

class Lstm_for_gen_text(nn.Module):
    """Custom network predicting the next character of a string.
    Parameters
    ----------
    vocab_size : int
        The number of characters in the vocabulary.
    embedding_dim : int
        Dimension of the character embedding vectors.
    dense_dim : int
        Number of neurons in the linear layer that follows the LSTM.
    hidden_dim : int
        Size of the LSTM hidden state.
    max_norm : int
        If any of the embedding vectors has a higher L2 norm than `max_norm`
        it is rescaled.
    n_layers : int
        Number of the layers of the LSTM.
    """
    def __init__(
        self,
        vocab_size,
        embedding_dim=2,
        dense_dim=32,
        hidden_dim=8,
        max_norm=2,
        n_layers=1,
       ):
       super().__init__()
       self.embedding=nn.Embedding(
        vocab_size,
        embedding_dim,
        padding_idx=vocab_size-1,
        norm_type=2,
        max_norm=max_norm,
       )
       self.lstm_1=nn.LSTM(
        embedding_dim,
        hidden_dim,
        batch_first=True,
        num_layers=n_layers,
       )
       self.lstm_2=nn.LSTM(
        hidden_dim,
        hidden_dim,
        batch_first=True,
        num_layers=n_layers,
       )
       self.lin_1=nn.Linear(hidden_dim,dense_dim)

    # project the last dim to vocab_size so later we  can pick the biggest prob to get the most probable character!
       self.lin_2=nn.Linear(dense_dim,vocab_size)
    
    def forward(self, x, h=None, c=None):
        """Run the forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(n_samples/batch_size, window_size)` of dtype
            `torch.int64`.
        h, c : torch.Tensor or None
            Hidden states of the LSTM.
        Returns
        -------
        logits : torch.Tensor
            Tensor of shape `(n_samples, vocab_size)`.
        h, c : torch.Tensor or None
            Hidden states of the LSTM.----- a detailed look can be find in the ori_paper!
        """
        emb=self.embedding(x)   # (n_samples, window_size) ->(n_samples, window_size, embedding_dim)
        
        out,(h1,c1)=self.lstm_1(emb)  #(n_samples, window_size, embedding_dim) ->(n_samples, window_size, hidden_dim)
        out,(h,c)=self.lstm_2(out)  #(n_samples, window_size, hidden_dim) ->(n_samples, window_size, embedding_dim)

        out=out.mean(dim=0)  # (n_samples, window_size, embedding_dim) ->(n_samples, window_size)
        out=self.lin_1(out)  # (n_samples, window_size) -> (n_samples, dense_size)
        logits=self.lin_2(out)  # (n_samples, dense_size) -> (n_samples, vocab_size)

        return logits,h,c


        
if __name__ =='__main__':
    batch,window_size,embed_dim=2,3,10
    x=torch.LongTensor([[1,2,3],[4,5,6]])
    model=Lstm_for_gen_text(
        vocab_size=50,
        embedding_dim=embed_dim,
        dense_dim=2,
        hidden_dim=2,
    )
    target,h,c=model(x)
    print(target.shape)

        

