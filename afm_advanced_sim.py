import numpy as np
import matplotlib.pyplot as plt

class EnergyBasedAFMSim:
          def __init__(self, A=1e-19, sigma=0.34e-9):
                        self.A = A
                        self.sigma = sigma

          def potential_ts(self, h, R):
                        """Lennard-Jones Potential: Sphere-Flat"""
                        h = np.maximum(h, 0.1e-9)
                        # V = - (A*R)/(6*h) + (A*R*sigma^6)/(180*h^7)
                        return - (self.A * R) / (6 * h) + (self.A * R * self.sigma**6) / (1260 * h**7)

          def find_global_equilibrium(self, z, k, R, x_prev):
                        """
                                Total Energy E = 0.5 * k * x^2 + V_ts(z - x)
                                        Find the point x that minimizes this energy, considering continuity.
                                                """
                        h_range = np.linspace(0.1e-9, 20e-9, 3000)
                        x_candidates = z - h_range

        # Total Energy
              E_total = 0.5 * k * x_candidates**2 + self.potential_ts(h_range, R)

        # For stability, find the global minimum.
        idx_min = np.argmin(E_total)

        return x_candidates[idx_min]

    def simulate(self, z_range, k, R_nm):
                  R = R_nm * 1e-9

        # Approach
                  x_app = []
                  x_curr = 0.0
                  for z in z_range:
                                    x_curr = self.find_global_equilibrium(z, k, R, x_curr)
                                    x_app.append(x_curr)

        # Retract (trace back to show hysteresis)
                  x_ret = []
                  x_curr = x_app[-1]
                  for z in reversed(z_range):
                                    h_range = np.linspace(0.1e-9, 20e-9, 3000)
                                    x_cand = z - h_range
                                    E = 0.5 * k * x_cand**2 + self.potential_ts(h_range, R)

            x_curr = x_cand[np.argmin(E)]
            x_ret.append(x_curr)

        return np.array(x_app), np.array(x_ret[::-1])

def main():
          sim = EnergyBasedAFMSim()
    z_range = np.linspace(12e-9, -1e-9, 800)

    plt.figure(figsize=(9, 6))

    params = [
                  {'k': 0.2, 'R': 15, 'color': 'red', 'label': 'k=0.2, R=15nm'},
                  {'k': 0.8, 'R': 15, 'color': 'blue', 'label': 'k=0.8, R=15nm'}
    ]

    for p in params:
                  x_app, x_ret = sim.simulate(z_range, p['k'], p['R'])
                  z_nm = z_range * 1e9
                  plt.plot(z_nm, x_app * 1e9, label=f"{p['label']} (App)", color=p['color'], lw=2)
                  plt.plot(z_nm, x_ret * 1e9, label=f"{p['label']} (Ret)", color=p['color'], linestyle='--', alpha=0.5)

    plt.title("AFM Force Curve (Energy Minimization Model)")
    plt.xlabel("Z distance (nm)")
    plt.ylabel("Deflection (nm)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig("afm_final_curve.png")
    print("Graph generated successfully.")

if __name__ == "__main__":
          main()
