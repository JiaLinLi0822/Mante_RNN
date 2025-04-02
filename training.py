import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from env import *
from net import *

def train_model(num_trials=1000, batch_size=64, T=0.75, learning_rate=1e-3, device='cpu'):
    """
    Train a continuous-time RNN model using only MSE as the loss function, without using BPTT.
    
    Parameters:
      num_epochs   : Number of training epochs
      batch_size   : Number of trials per training batch
      T            : Number of time steps per trial (this parameter is currently unused)
      learning_rate: Learning rate
      device       : 'cpu' or 'cuda'
    
    Returns:
      model, loss_history
    """
    model = RNNModel(input_size=4, hidden_size=100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    env = RDM()
    
    loss_history = []
    for epoch in range(num_trials):
        model.train()
        input_seq, targets = env.generate_trial(batch_size)
        optimizer.zero_grad()
        
        # Directly obtain outputs from the model and compute MSE loss
        outputs = model(input_seq).squeeze()
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_trials} - Loss: {loss.item():.4f}")
            
    return model, loss_history

if __name__ == "__main__":

    # set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobid', type=str, default='1', help='job id')
    parser.add_argument('--path', type=str, default=os.path.join(os.getcwd(), 'results'), help='path to store results')
    args = parser.parse_args()
    exp_path = os.path.join(args.path, f'exp_{args.jobid}')
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loss_history = train_model(num_trials=1000, batch_size=64, T=0.75, learning_rate=1e-3, device=device)
    
    # plot the training curve
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.show()
    
    # save the model
    torch.save(model.state_dict(), os.path.join(exp_path, 'model.pth'))