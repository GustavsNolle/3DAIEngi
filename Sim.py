import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class PhysicsEnv2DAdvanced:
    def __init__(self, width=100, height=200):
        self.width = width
        self.height = height
        self.T_amb = 300.0          # Ambient temperature (K)
        self.dt = 1.0               # Time step (s)
        self.max_steps = 500
        self.current_step = 0

        # --- Gas Fields ---
        # Gas temperature field (K)
        self.T_g = np.full((self.height, self.width), self.T_amb, dtype=np.float32)
        # Gas density field (moles per unit area, ambient = 1)
        self.n_g = np.ones((self.height, self.width), dtype=np.float32)
        # Pressure field computed by a simplified ideal gas law: P = n_g * T_g (units arbitrary)
        self.P = self.n_g * self.T_g

        # --- Metal Fields ---
        # Metal presence: a Boolean mask indicating where metal is present.
        self.metal_mask = np.zeros((self.height, self.width), dtype=bool)
        # Metal temperature field; only valid in cells where metal_mask is True.
        self.T_m = np.full((self.height, self.width), self.T_amb, dtype=np.float32)

        # --- Dynamics Parameters ---
        self.diff_coeff = 0.1       # Diffusion coefficient for gas temperature
        self.density_diff_coeff = 0.01  # Diffusion coefficient for gas density
        self.cooling_rate = 0.01    # Natural cooling rate (gas relaxes toward ambient)
        self.wind_x = 1             # Horizontal wind speed (units per step; positive = rightward)
        self.wind_y = 0             # Vertical wind speed (units per step; positive = upward)
        self.conduction_rate = 0.05 # Conduction coefficient between gas and metal

    def reset(self):
        self.T_g.fill(self.T_amb)
        self.n_g.fill(1.0)
        self.P = self.n_g * self.T_g
        self.metal_mask.fill(False)
        self.T_m.fill(self.T_amb)
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        """
        Returns a dictionary of the current fields. (The agent can process these via CNNs.)
        """
        return {
            'T_g': self.T_g.copy(),
            'n_g': self.n_g.copy(),
            'P': self.P.copy(),
            'metal_mask': self.metal_mask.copy(),
            'T_m': self.T_m.copy()
        }

    def step(self, action):
        """
        Accepts an action tuple: (action_type, x, y)
          - action_type: 0 = add metal; 1 = remove metal.
          - (x, y): Coordinates within the grid.
        Then, advances the simulation by one time step.
        """
        # --- Process Agent Action: Adding or Removing Metal ---
        action_type, x, y = action
        if 0 <= y < self.height and 0 <= x < self.width:
            if action_type == 0:  # Add metal
                if not self.metal_mask[y, x]:
                    self.metal_mask[y, x] = True
                    # Initialize metal temperature to the current gas temperature.
                    self.T_m[y, x] = self.T_g[y, x]
            elif action_type == 1:  # Remove metal
                if self.metal_mask[y, x]:
                    self.metal_mask[y, x] = False
                    # When metal is removed, transfer its temperature to the gas cell.
                    self.T_g[y, x] = self.T_m[y, x]

        # --- Gas Dynamics ---
        # 1. Advect gas temperature and density due to wind.
        self.T_g = self._advect_field(self.T_g, self.wind_x, self.wind_y, self.T_amb)
        self.n_g = self._advect_field(self.n_g, self.wind_x, self.wind_y, 1.0)
        # 2. Diffuse gas temperature and density.
        self.T_g = self._diffuse_field(self.T_g, self.diff_coeff)
        self.n_g = self._diffuse_field(self.n_g, self.density_diff_coeff)
        # 3. Natural cooling: relax gas temperature toward ambient.
        self.T_g += self.cooling_rate * (self.T_amb - self.T_g) * self.dt
        # 4. Update pressure (simplified ideal gas law).
        self.P = self.n_g * self.T_g

        # --- Metal and Gas Coupling ---
        self._conduct_heat_transfer()

        self.current_step += 1
        done = (self.current_step >= self.max_steps)
        reward = 0  # (Define a reward function as needed for your control objectives.)
        return self._get_obs(), reward, done, {}

    def _advect_field(self, field, wind_x, wind_y, ambient):
        """
        Advects a 2D field using a simple shift (np.roll) based on integer wind speeds.
        Open-boundary conditions are imposed by filling the "incoming" edge with the ambient value.
        """
        shift_y = int(np.sign(wind_y))
        shift_x = int(np.sign(wind_x))
        advected = np.roll(field, shift=(shift_y, shift_x), axis=(0, 1))
        # Impose open boundaries:
        if shift_y > 0:
            advected[0, :] = ambient
        elif shift_y < 0:
            advected[-1, :] = ambient
        if shift_x > 0:
            advected[:, 0] = ambient
        elif shift_x < 0:
            advected[:, -1] = ambient
        return advected

    def _diffuse_field(self, field, diffusion_coeff):
        """
        Diffuses a field using a finite-difference Laplacian.
        Open boundaries are handled by padding with the field’s ambient/mean value.
        """
        field_new = field.copy()
        # Use constant padding (for temperature, use ambient; for density, ambient is 1)
        pad_value = self.T_amb if field is self.T_g else np.mean(field)
        padded = np.pad(field, pad_width=1, mode='constant', constant_values=pad_value)
        laplacian = (padded[2:, 1:-1] + padded[0:-2, 1:-1] +
                     padded[1:-1, 2:] + padded[1:-1, 0:-2] -
                     4 * field)
        field_new += diffusion_coeff * laplacian * self.dt
        return field_new

    def _conduct_heat_transfer(self):
        """
        For each metal cell, compute heat exchange with adjacent gas cells.
        The metal cell's temperature is nudged toward the average temperature of its gas neighbors,
        while the gas cells receive an equal and opposite adjustment.
        """
        metal_indices = np.argwhere(self.metal_mask)
        for (i, j) in metal_indices:
            # Identify 4-connected neighbors that are gas cells.
            neighbors = []
            if i > 0 and not self.metal_mask[i - 1, j]:
                neighbors.append((i - 1, j))
            if i < self.height - 1 and not self.metal_mask[i + 1, j]:
                neighbors.append((i + 1, j))
            if j > 0 and not self.metal_mask[i, j - 1]:
                neighbors.append((i, j - 1))
            if j < self.width - 1 and not self.metal_mask[i, j + 1]:
                neighbors.append((i, j + 1))
            if neighbors:
                # Compute the average gas temperature of the neighbors.
                avg_gas_temp = np.mean([self.T_g[ni, nj] for (ni, nj) in neighbors])
                # Calculate the temperature difference.
                delta = self.conduction_rate * (avg_gas_temp - self.T_m[i, j]) * self.dt
                # Update metal temperature.
                self.T_m[i, j] += delta
                # Distribute the opposite change equally among the gas neighbors.
                delta_per_neighbor = delta / len(neighbors)
                for (ni, nj) in neighbors:
                    self.T_g[ni, nj] -= delta_per_neighbor

    def render(self, mode='human'):
        """
        Provides a simple UI using matplotlib. Four panels are displayed:
          - Gas Temperature Field
          - Gas Density Field
          - Gas Pressure Field
          - Metal Temperature (only in cells where metal is present; others masked out)
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        im0 = axs[0, 0].imshow(self.T_g, cmap=cm.inferno, origin='lower')
        axs[0, 0].set_title('Gas Temperature')
        fig.colorbar(im0, ax=axs[0, 0])
        
        im1 = axs[0, 1].imshow(self.n_g, cmap=cm.Blues, origin='lower')
        axs[0, 1].set_title('Gas Density')
        fig.colorbar(im1, ax=axs[0, 1])
        
        im2 = axs[1, 0].imshow(self.P, cmap=cm.plasma, origin='lower')
        axs[1, 0].set_title('Gas Pressure')
        fig.colorbar(im2, ax=axs[1, 0])
        
        # For metal temperature, only display values where metal is present.
        metal_display = np.where(self.metal_mask, self.T_m, np.nan)
        im3 = axs[1, 1].imshow(metal_display, cmap=cm.viridis, origin='lower')
        axs[1, 1].set_title('Metal Temperature (Metal Cells)')
        fig.colorbar(im3, ax=axs[1, 1])
        
        plt.tight_layout()
        plt.show()

# Example usage (for testing the environment):
if __name__ == "__main__":
    env = PhysicsEnv2DAdvanced()
    obs = env.reset()
    done = False
    step = 0
    while not done:
        # For demonstration, choose a random action:
        # Randomly decide to add (0) or remove (1) metal at a random coordinate.
        action_type = np.random.choice([0, 1])
        x = np.random.randint(0, env.width)
        y = np.random.randint(0, env.height)
        obs, reward, done, info = env.step((action_type, x, y))
        print(f"Step {step}: Action {(action_type, x, y)} | Reward: {reward}")
        step += 1
        if step % 50 == 0:
            env.render()
