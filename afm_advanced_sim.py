import numpy as np
import matplotlib.pyplot as plt
import os

class AdvancedAFMSimulator:
      def __init__(self, hamaker_const=1e-19, sigma=0.34e-9):
                """
                        AFM Simulation based on Lennard-Jones potential for Sphere-Flat geometry.

                                        Args:
                                                    hamaker_const (float): Hamaker constant [J]. Default is 1e-19.
                                                                sigma (float): Atomic size / contact distance [m]. Default is 0.34nm.
                                                                        """
                self.A = hamaker_const
                self.sigma = sigma
                self.h_min = 0.15e-9 # Minimum separation for numerical stability

    def force_ts(self, h, R):
              """
                      Calculate Tip-Sample force using Lennard-Jones inspired Sphere-Plane model.
                              F(h) = - (A*R)/(6*h^2) + (A*R*sigma^6)/(180*h^8)
                                      """
              h = np.maximum(h, self.h_min)
              # Attractive Van der Waals term
              f_attr = - (self.A * R) / (6 * h**2)
              # Repulsive term (Pauli/Born repulsion)
              f_rep = (self.A * R * self.sigma**6) / (180 * h**8)
              return f_attr + f_rep

    def find_equilibrium(self, z, k, R, x_start):
              """
                      Find the deflection 'x' that satisfies: k*x + F_ts(z - x, R) = 0
                              """
              x = x_start
              for _ in range(200):
                            h = z - x
                            f_ext = self.force_ts(h, R)
                            residual = k * x + f_ext

            # Numeric derivative for Newton's method: slope = k - dF/dh
                  dh = 1e-13
            df_dh = (self.force_ts(h + dh, R) - self.force_ts(h, R)) / dh
            slope = k - df_dh

            dx = -residual / slope
            x += dx
            if abs(dx) < 1e-16:
                              break
                      return x

    def run_simulation(self, z_range, k, R_nm):
              """
                      Simulate an approach-retract cycle.
                              """
        R = R_nm * 1e-9
        z_approach = z_range
        z_retract = z_range[::-1]

        # Approach
        x_app = []
        x_curr = 0.0
        for z in z_approach:
                      x_curr = self.find_equilibrium(z, k, R, x_curr)
                      x_app.append(x_curr)

        # Retract
        x_ret = []
        x_curr = x_app[-1] # Start from the closest point
        for z in z_retract:
                      x_curr = self.find_equilibrium(z, k, R, x_curr)
                      x_ret.append(x_curr)

        return np.array(x_app), np.array(x_ret[::-1])

def plot_comparison(results, z_range_nm, title, filename):
      plt.figure(figsize=(10, 7))

    for label, data in results.items():
              plt.plot(z_range_nm, data['app'] * 1e9, label=f'{label} (App)', lw=2)
        plt.plot(z_range_nm, data['ret'] * 1e9, label=f'{label} (Ret)', lw=1.5, linestyle='--')

    plt.title(title, fontsize=14)
    plt.xlabel("Support Position z (nm)", fontsize=12)
    plt.ylabel("Cantilever Deflection x (nm)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.gca().invert_yaxis() # Convention: Positive deflection is downwards/away

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved: {filename}")

def main():
      sim = AdvancedAFMSimulator()

    # Simulation range: 15nm to -1nm
    z_range = np.linspace(15e-9, -1e-9, 1000)
    z_range_nm = z_range * 1e9

    # --- Case 1: Varying Spring Constant (k) ---
    print("Simulating Case 1: Varying k (R=20nm)...")
    results_k = {}
    R_fixed = 20.0
    for k in [0.05, 0.2, 0.5]:
              x_app, x_ret = sim.run_simulation(z_range, k, R_fixed)
        results_k[f'k={k}N/m'] = {'app': x_app, 'ret': x_ret}
    plot_comparison(results_k, z_range_nm, f"AFM Simulation: Effect of Spring Constant (R={R_fixed}nm)", "compare_k.png")

    # --- Case 2: Varying Tip Radius (R) ---
    print("Simulating Case 2: Varying R (k=0.1N/m)...")
    results_R = {}
    k_fixed = 0.1
    for R in [5, 20, 50]:
              x_app, x_ret = sim.run_simulation(z_range, k_fixed, R)
        results_R[f'R={R}nm'] = {'app': x_app, 'ret': x_ret}
    plot_comparison(results_R, z_range_nm, f"AFM Simulation: Effect of Tip Radius (k={k_fixed}N/m)", "compare_R.png")

if __name__ == "__main__":
      main()
