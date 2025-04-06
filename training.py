import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from env import *
from net import *
from utils import *

def train(model, env, num_trials=1000, learning_rate=1e-3, device='cpu'):
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

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    loss_history = []

    # initialize the slow points
    slow_points = {0: None, 1: None}

    # Set up dynamic plotting
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    line, = ax.plot(loss_history)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss(Dynamic)')

    for epoch in range(num_trials):
        model.train()
        input_seq, targets, d_m, d_c, context_flags = env.generate_trial()
        input_seq = input_seq.to(device)
        targets = targets.to(device)
        context_flags = context_flags.to(device)

        batch_size = input_seq.size(1)
        h0_list = []
        for flag in context_flags:
            flag_val = int(flag.item())
            # halfway through training, use the slow points as initial states
            if (epoch >= num_trials//2) and (slow_points[flag_val] is not None):
                init_state = slow_points[flag_val].clone()  # shape [1, hidden_size]
            else:
                init_state = get_context_initial_state(model, flag_val, steps=200)
            h0_list.append(init_state)
        h0 = torch.cat(h0_list, dim=0)  # [batch_size, hidden_size]

        optimizer.zero_grad()
        outputs, _ = model(input_seq, h0=h0)
        
        loss = criterion(outputs, torch.stack([torch.zeros_like(targets), targets], dim=1).to(device))
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())

        # halfway through training, refine the slow points
        if (epoch + 1) == num_trials // 2:
            print("Refining slow points at mid-training...")
            # 分别对 motion 和 color context 进行 refine
            slow_points[0] = refine_slow_point(model, context_flag=0, steps=200, refine_steps=200, lr=1e-2)
            slow_points[1] = refine_slow_point(model, context_flag=1, steps=200, refine_steps=200, lr=1e-2)
            print("Updated slow points.")

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
    parser.add_argument('--jobid', type=str, default='0', help='job id')
    parser.add_argument('--path', type=str, default=os.path.join(os.getcwd(), 'results'), help='path to store results')
    args = parser.parse_args()
    exp_path = os.path.join(args.path, f'exp_{args.jobid}')
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = RNNModel(input_size=4, hidden_size=100, device=device)
    env = RDM(T=750, batch_size=64, device=device)

    model, loss_history = train(model, env, num_trials=1000, learning_rate=1e-3, device=device)
    
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