#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

class AFMSimulator:
      def __init__(self, k=0.2, radius_nm=30.0):
                # Basic parameters
                self.k = k # Spring constant [N/m]
        self.R = radius_nm * 1e-9 # Tip radius [m]
        # Constants related to Lennard-Jones (typical values)
        self.A = 1e-19 # Hamaker constant [J]
        self.sigma = 0.34e-9 # Atomic size [m]
        self.h_min = 0.15e-9 # Minimum distance for calculation (to avoid collision)

    def force_ts(self, h):
              """Calculate the force between the tip and the sample (Lennard-Jones type)"""
              h = np.maximum(h, self.h_min)
              # Attraction term (Van der Waals)
              f_attr = - (self.A * self.R) / (6 * h**2)
              # Repulsion term (Pauli repulsion)
              f_rep = (self.A * self.R * self.sigma**6) / (180 * h**8)
              return f_attr + f_rep

    def solve_equilibrium(self, z_support, x_start):
              """
                      Find the deflection x for a given support height z.
                              Search for x such that k * x + F_ts(z - x) = 0.
                                      """
              x = x_start
              # Track the equilibrium point using a simple iterative method (similar to Newton's method)
              for _ in range(100):
                            h = z_support - x
                            f_ext = self.force_ts(h)
                            residual = self.k * x + f_ext
                            # Slope (k - dF/dh)
                            # Numerical differentiation for simplicity
                            dh = 1e-12
                            df_dh = (self.force_ts(h + dh) - self.force_ts(h)) / dh
                            slope = self.k - df_dh
                            dx = -residual / slope
                            x += dx
                            if abs(dx) < 1e-15:
                                              break
                                      return x

    def simulate(self, z_range):
              """Calculate deflection for a series of support heights"""
              x_values = []
              h_values = []
              x_current = 0.0 # Initial value (deflection is zero at far distance)
        for z in z_range:
                      x_current = self.solve_equilibrium(z, x_current)
                      x_values.append(x_current)
                      h_values.append(z - x_current)
                  return np.array(x_values), np.array(h_values)

def main():
      # Simulation settings
      sim = AFMSimulator(k=0.2, radius_nm=30.0)
    # Support movement range: round trip from 15nm to -2nm
    z_far = 15e-9
    z_near = -1e-9
    points = 1000
    z_approach = np.linspace(z_far, z_near, points)
    z_retract = np.linspace(z_near, z_far, points)

    print("Simulating Approach...")
    x_app, h_app = sim.simulate(z_approach)
    print("Simulating Retract...")
    x_ret, h_ret = sim.simulate(z_retract)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(z_approach * 1e9, x_app * 1e9, label='Approach', color='red', lw=2)
    plt.plot(z_retract * 1e9, x_ret * 1e9, label='Retract', color='blue', lw=2, linestyle='--')
    plt.title("AFM Force-Curve Simulation (R=30nm, k=0.2N/m)")
    plt.xlabel("Support Position z (nm)")
    plt.ylabel("Cantilever Deflection x (nm)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    # Invert Y-axis (deflection) as per AFM convention
    plt.gca().invert_yaxis()
    # Save
    plt.savefig("afm_plot.png")
    print("Done! Plot saved as 'afm_plot.png'")

if __name__ == "__main__":
      main()
