import numpy as np
import argparse
import torch

# Hyperparameters model
vocab_size = 70
window_size = 10
embedding_dim = 2
hidden_dim = 16
dense_dim = 32
n_layers = 1
max_norm = 2

 # Training config
n_epochs = 25
train_val_split = 0.8
batch_size = 128
random_state = 13

loss_fn = torch.nn.CrossEntropyLoss()

def compute_loss(loss_fn,model,dataloader):
    """Computer average loss over the entire dataset."""
    model.eval()
    total_losses=[]
    for inp,target in dataloader:
        pres,_,_=model(inp)
        total_losses.append(loss_fn(pres, target).item())

    return np.mean(total_losses)

def generate_text(n_chars, model, dataset, initial_text="Hello", random_state=None):
    """Generate text with the character-level model.
    Parameters
    ----------
    n_chars : int
        Number of characters to generate.
    model : Module
        Character-level model.
    dataset : CharacterDataset
        Instance of the `CharacterDataset`.
    initial_text : str
        The starting text to be used as the initial condition for the model.
    random_state : None or int
        If not None, then the result is reproducible.
    Returns
    -------
    res : str
        Generated text.
    """

    if not initial_text:
        raise ValueError("You need to specify the initial text")

    res = initial_text
    model.eval()
    h, c = None, None

    if random_state is not None:
        np.random.seed(random_state)
    for _ in range(n_chars):
        previous_char=initial_text if res==initial_text else res[-1]
        features=torch.LongTensor(
            [
                [
                    dataset.ch2ix[c] for c in previous_char
                ]
            ])
        logits,_,_=model(features)
        probs=F.
