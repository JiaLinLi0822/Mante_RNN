import torch
import numpy as np

class RDM:
    """
    Class for generating trial data based on the white noise model from Mante et al. (2013) supplementary material.
    Each trial contains T timesteps, and at each timestep, there are motion signal (u_m), color signal (u_c),
    and context signals.
    """
    def __init__(self,
                 T=750,
                 dt=0.001,
                 device='cpu',
                 motion_coherence_levels=[0.05, 0.15, 0.5],
                 color_coherence_levels=[0.06, 0.18, 0.5],
                 noise_std_per_sqrt_dt=3.1623,
                 batch_size=1):
        self.T = T
        self.dt = dt
        self.device = device
        self.motion_coherence_levels = motion_coherence_levels
        self.color_coherence_levels = color_coherence_levels
        self.noise_std_per_sqrt_dt = noise_std_per_sqrt_dt
        self.batch_size = batch_size

    def generate_trial(self):
        """
        Generate trial data using a white noise model (from Mante et al. (2013) supplementary material).
        Each trial consists of T timesteps, and each timestep includes:
           - motion signal (u_m)
           - color signal (u_c)
           - context signals
        
        Parameters:
          batch_size : Number of trials to generate in one batch.
        
        Returns:
          trial_input_seq : A [T, batch_size, 4] tensor representing the input sequence in the order [u_m, u_c, context_m, context_c].
          targets         : A [batch_size] tensor with values +1 or -1 indicating the decision target for each trial.
        """
        T = self.T
        dt = self.dt

        # Step 1: Select context: 0 for motion task, 1 for color task
        context_flags = np.random.randint(0, 2, size=(self.batch_size,))

        # Step 2: Randomly select absolute coherence values for motion/color and determine the sign
        motion_coherences = np.random.choice(self.motion_coherence_levels, size=self.batch_size)
        color_coherences  = np.random.choice(self.color_coherence_levels, size=self.batch_size)

        # Randomly choose sign: +1 or -1
        motion_signs = np.random.choice([1, -1], size=self.batch_size)
        color_signs  = np.random.choice([1, -1], size=self.batch_size)

        # Biases d_m and d_c
        d_m = motion_coherences * motion_signs
        d_c = color_coherences * color_signs

        # Step 3: Generate white noise: rho_m(t), rho_c(t)
        # Standard deviation = noise_std_per_sqrt_dt * sqrt(dt) (e.g., if dt=0.01 then std=3.1623)
        noise_std = self.noise_std_per_sqrt_dt * np.sqrt(dt)

        # Create noise for each trial and each timestep
        rho_m = np.random.randn(T, self.batch_size) * noise_std
        rho_c = np.random.randn(T, self.batch_size) * noise_std

        # Step 4: Generate final u_m(t) and u_c(t)
        um = np.zeros((T, self.batch_size))
        uc = np.zeros((T, self.batch_size))
        for i in range(self.batch_size):
            um[:, i] = d_m[i] + rho_m[:, i]
            uc[:, i] = d_c[i] + rho_c[:, i]

        # Step 5: Construct context input
        # For each trial, if context_flag == 0 then context = [1, 0]; otherwise [0, 1]
        context_input = np.zeros((self.batch_size, 2))
        context_input[np.arange(self.batch_size), context_flags] = 1.0

        # Expand context signals to T timesteps (remains constant during the stimulus period)
        context_m = np.tile(context_input[:, 0], (T, 1))
        context_c = np.tile(context_input[:, 1], (T, 1))

        # Step 6: Concatenate inputs into a sequence [u_m, u_c, context_m, context_c]
        # Final shape: [T, batch_size, 4]
        trial_input_seq = np.stack([um, uc, context_m, context_c], axis=-1)

        # Step 7: Compute decision targets
        # For a motion task, the target depends on the sign of d_m;
        # for a color task, it depends on the sign of d_c.
        # We use the sign directly without summing over timesteps.
        targets = np.zeros(self.batch_size, dtype=np.float32)
        for i in range(self.batch_size):
            if context_flags[i] == 0:
                # Motion task
                targets[i] = 1.0 if d_m[i] >= 0 else -1.0
            else:
                # Color task
                targets[i] = 1.0 if d_c[i] >= 0 else -1.0

        # Step 8: Convert arrays to torch tensors
        trial_input_seq = torch.tensor(trial_input_seq, dtype=torch.float32, device=self.device)
        targets = torch.tensor(targets, dtype=torch.float32, device=self.device)

        return trial_input_seq, targets, motion_coherences * motion_signs, color_coherences * color_signs, context_flags
    
if __name__ == "__main__":
    # Example usage
    env = RDM(batch_size=1)
    input_seq, targets, motion_coherences, color_coherences, context_flags = env.generate_trial()
    print("Input Sequence Shape:", input_seq.shape)
    print("Targets Shape:", targets.shape)
    print("Motion Coherences:", motion_coherences)
    print("Color Coherences:", color_coherences)
    print("Context Flags:", context_flags)
    print("Input Sequence:", input_seq)
    print("Targets:", targets)
