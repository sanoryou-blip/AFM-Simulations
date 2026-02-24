import numpy as np
import matplotlib.pyplot as plt

class PhysicalAFMSimulator:
    def __init__(self, Hamaker=2e-20, sigma=0.34e-9):
        # Hamaker constant (liquid approx: 2e-20 J)
        self.A = Hamaker
        self.sigma = sigma

    def get_force(self, h, R):
        """
        Tip-Sample Force (Lennard-Jones)
        Attraction is positive (pulling tip towards sample)
        Repulsion is negative
        """
        h = np.maximum(h, 0.1e-9)
        f_attr = (self.A * R) / (6 * h**2)
        f_rep = - (self.A * R * self.sigma**6) / (180 * h**8)
        return f_attr + f_rep

    def solve_step(self, z, k, R, x_prev):
        """
        Find equilibrium: k*x = Force(z - x)
        Track the current branch of equilibrium.
        """
        # Search range for deflection x around previous state
        # Tip-sample distance h = z - x must be > 0.1nm
        x_min = x_prev - 2e-9
        x_max = min(x_prev + 2e-9, z - 0.1e-9)
        
        x_search = np.linspace(x_min, x_max, 2000)
        # Residual: k*x - Force(z-x) = 0
        residual = k * x_search - self.get_force(z - x_search, R)
        
        # Look for zero crossings (equilibrium points)
        crossings = np.where(np.diff(np.sign(residual)))[0]
        
        if len(crossings) > 0:
            # Pick the equilibrium point closest to previous state (continuity)
            idx = crossings[np.argmin(np.abs(x_search[crossings] - x_prev))]
            return x_search[idx]
        else:
            # Branch disappeared -> Jump to the other stable state (global search)
            h_full = np.linspace(0.1e-9, 20e-9, 4000)
            x_full = z - h_full
            res_full = np.abs(k * x_full - self.get_force(h_full, R))
            return x_full[np.argmin(res_full)]

    def run(self, z_start, z_end, k, R_nm, steps=1000):
        R = R_nm * 1e-9
        z_approach = np.linspace(z_start, z_end, steps)
        z_retract = z_approach[::-1]
        
        # Approach
        x_app = []
        x_curr = 0.0
        for z in z_approach:
            x_curr = self.solve_step(z, k, R, x_curr)
            x_app.append(x_curr)
            
        # Retract
        x_ret = []
        x_curr = x_app[-1]
        for z in z_retract:
            x_curr = self.solve_step(z, k, R, x_curr)
            x_ret.append(x_curr)
            
        return z_approach, np.array(x_app), np.array(x_ret[::-1])

def main():
    sim = PhysicalAFMSimulator()
    # 5nm Scan Range (standard HS-AFM scale)
    z_start, z_end = 6e-9, -0.5e-9
    
    plt.figure(figsize=(8, 6))
    
    # Typical HS-AFM Cantilever (BL-AC10FS etc.)
    # k = 0.1 N/m, R = 5-10 nm
    configs = [
        {'k': 0.1, 'R': 8, 'color': 'red', 'label': 'Soft Tip (k=0.1, R=8nm)'},
        {'k': 0.2, 'R': 8, 'color': 'blue', 'label': 'Stiff Tip (k=0.2, R=8nm)'}
    ]
    
    for c in configs:
        z, x_app, x_ret = sim.run(z_start, z_end, c['k'], c['R'])
        z_nm = z * 1e9
        plt.plot(z_nm, x_app * 1e9, label=f"{c['label']} App", color=c['color'], lw=2)
        plt.plot(z_nm, x_ret * 1e9, label=f"{c['label']} Ret", color=c['color'], linestyle='--', alpha=0.7)

    plt.title("Realistic AFM Force Curve (Liquid/HS-AFM Scale)")
    plt.xlabel("Support Position z (nm)")
    plt.ylabel("Deflection x (nm)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    # Convention: Deflection towards sample is positive, but plots often invert Y
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig("afm_corrected_final.png")
    print("Graph generated: afm_corrected_final.png")

if __name__ == "__main__":
    main()
