import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

class AFMPanelApp:
      def __init__(self, root):
                self.root = root
                self.root.title("HS-AFM Physics Simulator - Control Panel")

        # --- Physical Constants ---
                self.sigma = 0.34e-9

        # --- UI Setup ---
                self.setup_ui()

        # --- Initial Plot ---
                self.update_plot()

      def setup_ui(self):
                # Main Layout
                self.main_frame = ttk.Frame(self.root, padding="10")
                self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Side: Controls
                self.ctrl_frame = ttk.LabelFrame(self.main_frame, text="Parameters", padding="10")
                self.ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # -- Spring Constant (k) --
                ttk.Label(self.ctrl_frame, text="Spring Constant k [N/m]:").pack(anchor=tk.W)
                self.k_val = tk.DoubleVar(value=0.1)
                self.k_slider = ttk.Scale(self.ctrl_frame, from_=0.01, to=2.0, variable=self.k_val, orient=tk.HORIZONTAL, command=self.update_plot)
                self.k_slider.pack(fill=tk.X, pady=5)
                self.k_label = ttk.Label(self.ctrl_frame, text="0.10")
                self.k_label.pack(anchor=tk.E)

          # -- Tip Radius (R) --
                ttk.Label(self.ctrl_frame, text="Tip Radius R [nm]:").pack(anchor=tk.W, pady=(10,0))
                self.r_val = tk.DoubleVar(value=10.0)
                self.r_slider = ttk.Scale(self.ctrl_frame, from_=1.0, to=50.0, variable=self.r_val, orient=tk.HORIZONTAL, command=self.update_plot)
                self.r_slider.pack(fill=tk.X, pady=5)
                self.r_label = ttk.Label(self.ctrl_frame, text="10.0")
                self.r_label.pack(anchor=tk.E)

          # -- Hamaker Constant (A) --
                ttk.Label(self.ctrl_frame, text="Hamaker Const [zJ] (10^-21):").pack(anchor=tk.W, pady=(10,0))
                self.a_val = tk.DoubleVar(value=20.0) # 20 zJ = 2e-20 J
        self.a_slider = ttk.Scale(self.ctrl_frame, from_=1.0, to=100.0, variable=self.a_val, orient=tk.HORIZONTAL, command=self.update_plot)
        self.a_slider.pack(fill=tk.X, pady=5)
        self.a_label = ttk.Label(self.ctrl_frame, text="20.0")
        self.a_label.pack(anchor=tk.E)

        # Reset Button
        ttk.Button(self.ctrl_frame, text="Reset to Default", command=self.reset_params).pack(fill=tk.X, pady=20)

        # Right Side: Plot
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def reset_params(self):
              self.k_val.set(0.1)
              self.r_val.set(10.0)
              self.a_val.set(20.0)
              self.update_plot()

    def get_force(self, h, R, A):
              h = np.maximum(h, 0.15e-9)
              f_attr = (A * R) / (6 * h**2)
              f_rep = - (A * R * self.sigma**6) / (180 * h**8)
              return f_attr + f_rep

    def solve_branch(self, z_range, k, R, A, forward=True):
              x_vals = []
              x_prev = 0.0

        loop_z = z_range if forward else z_range[::-1]

        for z in loop_z:
                      # Local search for equilibrium
                      h_search = np.linspace(0.15e-9, 25e-9, 1500)
                      x_cand = z - h_search
                      res = k * x_cand - self.get_force(h_search, R, A)

            crossings = np.where(np.diff(np.sign(res)))[0]
            if len(crossings) > 0:
                              idx = crossings[np.argmin(np.abs(x_cand[crossings] - x_prev))]
                              x_curr = x_cand[idx]
else:
                # Discontinuous jump (Snap-in / Jump-off)
                  x_curr = x_cand[np.argmin(np.abs(res))]

            x_vals.append(x_curr)
            x_prev = x_curr

        return np.array(x_vals) if forward else np.array(x_vals[::-1])

    def update_plot(self, *args):
              # Update Labels
              k = self.k_val.get()
              R_nm = self.r_val.get()
              A_zj = self.a_val.get()

        self.k_label.config(text=f"{k:.2f}")
        self.r_label.config(text=f"{R_nm:.1f}")
        self.a_label.config(text=f"{A_zj:.1f}")

        # Run Simulation
        A = A_zj * 1e-21
        R = R_nm * 1e-9
        z_range = np.linspace(8e-9, -0.5e-9, 400)

        x_app = self.solve_branch(z_range, k, R, A, forward=True)
        x_ret = self.solve_branch(z_range, k, R, A, forward=False)

        # Plot
        self.ax.clear()
        z_nm = z_range * 1e9
        self.ax.plot(z_nm, x_app * 1e9, 'red', lw=2, label="Approach")
        self.ax.plot(z_nm, x_ret * 1e9, 'blue', linestyle='--', alpha=0.7, label="Retract")

        self.ax.set_title(f"Force Curve (k={k:.2f}, R={R_nm:.1f}nm, A={A_zj:.1f}zJ)")
        self.ax.set_xlabel("Z Position [nm]")
        self.ax.set_ylabel("Deflection [nm]")
        self.ax.invert_yaxis()
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
      root = tk.Tk()
    app = AFMPanelApp(root)
    root.mainloop()
