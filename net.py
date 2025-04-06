import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=100, tau=0.01, dt=0.001, noise_std=0.1, device='cpu'):
        """
        Parameters:
        input_size: Input dimension, consisting of 4 channels: [motion, color, motion_context, color_context]
                    Corresponding to u_m, u_c, u_{cm}, u_{cc}.
        hidden_size: Number of neurons in the hidden layer (approximately 100 units in the paper)
        tau: Time constant, controls the speed of state changes
        dt: Discrete time step
        noise_std: Standard deviation of noise, used to simulate rho_x
        """
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.tau = tau # time constant
        self.dt = dt
        self.noise_std = noise_std
        self.device = device
        self.input_size = input_size
        
        # Recurrent connection: weight matrix J (no bias)
        # Initialized from a normal distribution with zero mean and variance 1/hidden_size (std = hidden_size**(-0.5))
        self.J = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.normal_(self.J.weight, mean=0.0, std=hidden_size ** (-0.5))
        
        # Assign separate weight parameters for different inputs
        # Each input is a scalar, so initialize from a normal distribution with zero mean and standard deviation 0.5
        self.b_m = nn.Parameter(torch.Tensor(hidden_size, 1)).to(self.device)   # motion evidence u_m
        self.b_c = nn.Parameter(torch.Tensor(hidden_size, 1)).to(self.device)   # color evidence u_c
        self.b_cm = nn.Parameter(torch.Tensor(hidden_size, 1)).to(self.device)  # motion context input u_cm
        self.b_cc = nn.Parameter(torch.Tensor(hidden_size, 1)).to(self.device)  # color context input u_cc
        
        nn.init.normal_(self.b_m, mean=0.0, std=0.5)
        nn.init.normal_(self.b_c, mean=0.0, std=0.5)
        nn.init.normal_(self.b_cm, mean=0.0, std=0.5)
        nn.init.normal_(self.b_cc, mean=0.0, std=0.5)
        
        # State bias c_x initialized to zero
        self.c_x = nn.Parameter(torch.zeros(hidden_size, device=self.device))
        
        # Readout layer: maps the final state to decision output
        # Initialize output weights to zero
        self.W_out = nn.Linear(hidden_size, 1)
        nn.init.constant_(self.W_out.weight, 0.0)
    
    def forward(self, input_seq, h0=None, extra_time=0.2, train_mode=True):
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
        
        states = []
        # Discrete-time Euler update
        for t in range(T):
            # Input at the current time step, shape [batch_size, 4]
            u_t = input_seq[t]
            # Separate each channel, maintaining shape [batch_size, 1]
            u_m, u_c, u_cm, u_cc = u_t.split(1, dim=1) # motion, color, motion_context, color_context
            
            # Recurrent nonlinear activation
            r = torch.tanh(x)
            
            # Compute contributions from each input:
            # Note: To match shapes, scalar inputs are multiplied by the transpose of the corresponding parameter matrix (resulting in [batch_size, hidden_size])
            input_term = (u_m @ self.b_m.t() +
                          u_c @ self.b_c.t() +
                          u_cm @ self.b_cm.t() +
                          u_cc @ self.b_cc.t())
            
            # Noise term, sampled at each step with the same shape as x
            noise = self.noise_std * torch.randn_like(x) # shape [batch_size, hidden_size]
            
            # Euler update formula: x_{t+1} = x_t + (dt/tau) * ( -x_t + J*r + input_term + c_x + noise )
            dx = (self.dt / self.tau) * (-x + self.J(r) + input_term + self.c_x + noise)
            x = x + dx
            states.append(x)

            if t == 0:
                r_first = torch.tanh(x)
                output_first = self.W_out(r_first)
        
        if (not train_mode) and (extra_time > 0):
            steps = int(extra_time / self.dt)
            for _ in range(steps):
                r = torch.tanh(x)
                noise = self.noise_std * torch.randn_like(x)

                dx = (self.dt / self.tau) * (-x + self.J(r) + self.c_x + noise)
                x = x + dx
        
        # Final state is transformed nonlinearly and passed through the linear readout layer to obtain the decision signal
        r_final = torch.tanh(x)
        output_final = self.W_out(r_final)
        states = torch.stack(states, dim=0)
        outputs = torch.stack([output_first.squeeze(), output_final.squeeze()], dim=-1)  # shape [batch_size, 2]

        return outputs, states
    
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
            u_m, u_c, u_cm, u_cc = u_t.split(1, dim=1) # motion, color, motion_context, color_context
            
            r = torch.tanh(x)
            input_term = (u_m @ self.b_m.t() +
                          u_c @ self.b_c.t() +
                          u_cm @ self.b_cm.t() +
                          u_cc @ self.b_cc.t())
            noise = self.noise_std * torch.randn_like(x)
            dx = (self.dt / self.tau) * (-x + self.J(r) + input_term + self.c_x + noise)
            x = x + dx
            states.append(x)
        
        r_final = torch.tanh(x)
        output = self.W_out(r_final)
        states = torch.stack(states, dim=0)

        return states, output.squeeze()
    
if __name__ == "__main__":
    # Test the model
    input_seq = torch.randn(10, 64, 4)  # 10 time steps, batch size 64, 4 input features
    model = RNNModel()
    output = model(input_seq)
    print(output.shape)  # Should be [64]
    print(output)  # Output decision signal
    
    states = model.forward_with_states(input_seq)
    print(states.shape)  # Should be [10, 64, hidden_size]