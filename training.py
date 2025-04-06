import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from env import *
from net import *

def train(model, env, num_trials=1000, learning_rate=1e-3):
    """
    Train a continuous-time RNN model using only MSE as the loss function, without using BPTT.
    
    Parameters:
      num_trials   : Number of training trials
      batch_size   : Number of trials per training batch
      T            : Duration parameter (currently unused)
      learning_rate: Learning rate
      device       : 'cpu' or 'cuda'
    
    Returns:
      model, loss_history
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    loss_history = []

    # Set up dynamic plotting
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    line, = ax.plot(loss_history)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss(Dynamic)')

    for epoch in range(num_trials):
        model.train()
        input_seq, targets, _, _, _ = env.generate_trial()
        optimizer.zero_grad()
        
        outputs = model(input_seq).squeeze()
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())

        # Update dynamic plot every 100 epochs
        if (epoch + 1) % 100 == 0:
            line.set_xdata(range(len(loss_history)))
            line.set_ydata(loss_history)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            print(f"Epoch {epoch+1}/{num_trials} - Loss: {loss.item():.4f}")
            
    plt.ioff()  # Turn off interactive plotting
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

    model = RNNModel(input_size=4, hidden_size=100)
    env = RDM(T=750, batch_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, loss_history = train(model, env, num_trials=60000, learning_rate=1e-3)
    
    # Transform the x-axis as percentage of epochs and bin the epochs
    total_epochs = len(loss_history)
    bin_size = total_epochs // 100  # Bin size for 1% increments
    binned_loss = [np.mean(loss_history[i:i + bin_size]) * 100 for i in range(0, total_epochs, bin_size)]
    percentage_epochs = np.linspace(0, 100, len(binned_loss))

    plt.figure(figsize=(10, 6))
    plt.plot(percentage_epochs, binned_loss)
    plt.xlabel('Epochs (%)')
    plt.ylabel('Loss (%)')
    plt.title('Training Loss')
    plt.show()
    # save the model
    torch.save(model.state_dict(), os.path.join(exp_path, 'model.pth'))