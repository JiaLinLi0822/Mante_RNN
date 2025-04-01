import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=100, tau=100.0, dt=1.0, noise_std=0.1):
        """
        Parameters:
        input_size: Input dimension, consisting of 4 channels: [motion, color, motion_context, color_context]
                    Corresponding to u_m, u_c, u_{cm}, u_{cc}.
        hidden_size: Number of neurons in the hidden layer (approximately 100 units in the paper)
        tau: Time constant, controls the speed of state changes
        dt: Discrete time step
        noise_std: Standard deviation of noise, used to simulate \rho_x
        """
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.tau = tau
        self.dt = dt
        self.noise_std = noise_std
        
        # Recurrent connection: weight matrix J (no bias), initialized orthogonally
        self.J = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.orthogonal_(self.J.weight)
        
        # Assign separate weight parameters for different inputs
        # Note: Each input is a scalar, so the parameter shape is (hidden_size, 1)
        self.b_m = nn.Parameter(torch.Tensor(hidden_size, 1))   # Corresponds to motion evidence u_m
        self.b_c = nn.Parameter(torch.Tensor(hidden_size, 1))   # Corresponds to motion context input u_c
        self.b_cm = nn.Parameter(torch.Tensor(hidden_size, 1))  # Corresponds to color context input u_{cm}
        self.b_cc = nn.Parameter(torch.Tensor(hidden_size, 1))  # Corresponds to color evidence u_{cc}
        
        # State bias c^x
        self.c_x = nn.Parameter(torch.Tensor(hidden_size))
        
        # Readout layer: maps the final state to decision output via a nonlinear transformation
        self.W_out = nn.Linear(hidden_size, 1)
        
        # Parameter initialization (using Xavier initialization here)
        nn.init.xavier_uniform_(self.b_m)
        nn.init.xavier_uniform_(self.b_c)
        nn.init.xavier_uniform_(self.b_cm)
        nn.init.xavier_uniform_(self.b_cc)
        nn.init.constant_(self.c_x, 0.0)
        nn.init.xavier_uniform_(self.W_out.weight)
    
    def forward(self, input_seq, h0=None):
        """
        Forward pass: Update state based on continuous-time equations and output decision signal at the final time step.
        
        Parameters:
          input_seq: Input sequence of shape [T, batch_size, 4], where each time step's input is
                     [u_m, u_c, u_{cm}, u_{cc}].
          h0: Initial hidden state (if None, initialized to zero)
          
        Returns:
          Readout result at the final time step, shape [batch_size]
        """
        T, batch_size, _ = input_seq.size()
        device = input_seq.device
        
        # Initial state x
        if h0 is None:
            x = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            x = h0
        
        # Discrete-time Euler update
        for t in range(T):
            # Input at the current time step, shape [batch_size, 4]
            u_t = input_seq[t]
            # Separate each channel, maintaining shape [batch_size, 1]
            u_m = u_t[:, 0:1]    # Motion evidence
            u_c = u_t[:, 1:2]   # Color evidence
            u_cm = u_t[:, 2:3]    # Motion context
            u_cc = u_t[:, 3:4]   # Color context
            
            # Recurrent nonlinear activation
            r = torch.tanh(x)
            
            # Compute contributions from each input:
            # Note: To match shapes, scalar inputs are multiplied by the transpose of the corresponding parameter matrix (resulting in [batch_size, hidden_size])
            input_term = (u_m @ self.b_m.t() +
                          u_c @ self.b_c.t() +
                          u_cm @ self.b_cm.t() +
                          u_cc @ self.b_cc.t())
            
            # Noise term, sampled at each step with the same shape as x
            noise = self.noise_std * torch.randn_like(x)
            
            # Euler update formula: x_{t+1} = x_t + (dt/tau) * ( -x_t + J*r + input_term + c_x + noise )
            dx = (self.dt / self.tau) * (-x + self.J(r) + input_term + self.c_x + noise)
            x = x + dx
        
        # Final state is transformed nonlinearly and passed through the linear readout layer to obtain the decision signal
        r_final = torch.tanh(x)
        output = self.W_out(r_final)
        return output.squeeze()
    
    def forward_with_states(self, input_seq, h0=None):
        """
        Similar to forward, but records the hidden state x at each time step for further analysis.
        Returns: State trajectory, shape [T, batch_size, hidden_size]
        """
        T, batch_size, _ = input_seq.size()
        device = input_seq.device
        if h0 is None:
            x = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            x = h0
        
        states = []
        for t in range(T):
            u_t = input_seq[t]
            u_m = u_t[:, 0:1]
            u_c = u_t[:, 1:2]
            u_cm = u_t[:, 2:3]
            u_cc = u_t[:, 3:4]
            
            r = torch.tanh(x)
            input_term = (u_m @ self.b_m.t() +
                          u_c @ self.b_c.t() +
                          u_cm @ self.b_cm.t() +
                          u_cc @ self.b_cc.t())
            noise = self.noise_std * torch.randn_like(x)
            dx = (self.dt / self.tau) * (-x + self.J(r) + input_term + self.c_x + noise)
            x = x + dx
            states.append(x)
        states = torch.stack(states, dim=0)
        return states
    
if __name__ == "__main__":
    # Test the model
    input_seq = torch.randn(10, 64, 4)  # 10 time steps, batch size 64, 4 input features
    model = RNNModel()
    output = model(input_seq)
    print(output.shape)  # Should be [64]
    print(output)  # Output decision signal
    
    states = model.forward_with_states(input_seq)
    print(states.shape)  # Should be [10, 64, hidden_size]