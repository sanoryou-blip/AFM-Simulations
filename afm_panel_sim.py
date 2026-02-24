import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import threading
import csv
import time
import os

class AFMPanelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HS-AFM Physics Simulator - Liquid Mode")
        
        # --- Physical Constants ---
        self.sigma = 0.34e-9 
        self.nu_s = 0.3      
        
        # --- Liquid Material Presets (E [GPa], A [zJ]) ---
        # Hamaker constants (A) are reduced for liquid environment (~1/10 of air)
        self.materials = {
            "Mica (in Water)": (60.0, 10.0),    
            "Silicon (in Water)": (160.0, 15.0),
            "Cell (Soft)": (0.0001, 2.0),   
            "Lipid Bilayer": (0.02, 5.0)    
        }
        
        # --- Data Storage ---
        self.last_results = None 
        self.view_mode = tk.StringVar(value="ForceCurve")
        self.z_mode = tk.StringVar(value="Sine")
        
        # --- UI Setup ---
        self.setup_ui()
        self.on_material_change() # Init entries
        self.status_var.set("Ready (Liquid Mode)")

    def setup_ui(self):
        # Main Layout
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- Left Side: Scrollable Controls ---
        self.left_container = ttk.Frame(self.main_frame)
        self.left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        self.ctrl_canvas = tk.Canvas(self.left_container, width=280, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.left_container, orient="vertical", command=self.ctrl_canvas.yview)
        
        # Scrollable inner frame
        self.scrollable_frame = ttk.Frame(self.ctrl_canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.ctrl_canvas.configure(scrollregion=self.ctrl_canvas.bbox("all"))
        )
        
        self.ctrl_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.ctrl_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Re-map ctrl_frame to scrollable_frame
        self.ctrl_frame = ttk.LabelFrame(self.scrollable_frame, text="Liquid AFM Parameters", padding="15")
        self.ctrl_frame.pack(fill=tk.BOTH, expand=True)
        
        def add_entry(parent, label_text, default_val):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=4)
            ttk.Label(frame, text=label_text).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default_val))
            entry = ttk.Entry(frame, textvariable=var, width=10, justify=tk.RIGHT)
            entry.pack(side=tk.RIGHT)
            return var

        # --- Section: Cantilever ---
        self.k_var = add_entry(self.ctrl_frame, "Spring Const k [N/m]:", 1.0)
        self.r_var = add_entry(self.ctrl_frame, "Tip Radius R [nm]:", 10.0)
        self.visc_var = add_entry(self.ctrl_frame, "Viscosity Cnt [nN·s/m]:", 5.0)
        
        ttk.Separator(self.ctrl_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        # --- Section: Material ---
        ttk.Label(self.ctrl_frame, text="Sample Preset (Liquid):", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        self.mat_choice = tk.StringVar(value="Mica (in Water)")
        self.mat_combo = ttk.Combobox(self.ctrl_frame, textvariable=self.mat_choice, values=list(self.materials.keys()), state="readonly")
        self.mat_combo.pack(fill=tk.X, pady=5)
        self.mat_combo.bind("<<ComboboxSelected>>", self.on_material_change)
        
        self.e_var = add_entry(self.ctrl_frame, "Young's Modulus [GPa]:", 60.0)
        self.a_var = add_entry(self.ctrl_frame, "Hamaker Const [zJ]:", 10.0)
        
        ttk.Separator(self.ctrl_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # --- Section: Scan ---
        self.z_start_var = add_entry(self.ctrl_frame, "Scan Start [nm]:", 6.0)
        self.z_end_var = add_entry(self.ctrl_frame, "Scan End [nm]:", -3.0)
        self.freq_var = add_entry(self.ctrl_frame, "Frequency [Hz]:", 3.0)
        self.steps_var = add_entry(self.ctrl_frame, "Pts per Cycle:", 500)
        self.noise_var = add_entry(self.ctrl_frame, "Noise Level [nm]:", 0.1)

        # --- Z Ramp Mode ---
        ttk.Label(self.ctrl_frame, text="Z Ramp Mode:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(10,0))
        ttk.Radiobutton(self.ctrl_frame, text="Linear (Triangular)", variable=self.z_mode, value="Linear").pack(anchor=tk.W)
        ttk.Radiobutton(self.ctrl_frame, text="Sine (Oscillation)", variable=self.z_mode, value="Sine").pack(anchor=tk.W)

        # --- View Control ---
        ttk.Label(self.ctrl_frame, text="Display Mode:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(10,0))
        ttk.Radiobutton(self.ctrl_frame, text="Force Curve (d vs Z)", variable=self.view_mode, value="ForceCurve", command=self.refresh_plot).pack(anchor=tk.W)
        ttk.Radiobutton(self.ctrl_frame, text="Time Trajectory", variable=self.view_mode, value="Trajectory", command=self.refresh_plot).pack(anchor=tk.W)

        # Status
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(self.ctrl_frame, textvariable=self.status_var, foreground="blue", font=("Helvetica", 10, "bold"))
        self.status_label.pack(pady=10)

        # Buttons
        self.calc_button = ttk.Button(self.ctrl_frame, text="RUN CALCULATION", command=self.start_calculation)
        self.calc_button.pack(fill=tk.X, pady=5)
        
        self.save_button = ttk.Button(self.ctrl_frame, text="SAVE DATA TO CSV", command=self.save_to_csv, state=tk.DISABLED)
        self.save_button.pack(fill=tk.X, pady=5)
        
        ttk.Button(self.ctrl_frame, text="Reset to Default", command=self.reset_params).pack(fill=tk.X, pady=5)

        # --- Right Side: Plot (Responsive) ---
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig, self.ax = plt.subplots(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_material_change(self, event=None):
        mat_name = self.mat_choice.get()
        if mat_name in self.materials:
            E_gpa, A_zj = self.materials[mat_name]
            self.e_var.set(str(E_gpa))
            self.a_var.set(str(A_zj))

    def reset_params(self):
        self.k_var.set("1.0")
        self.r_var.set("10.0")
        self.visc_var.set("5.0")
        self.mat_choice.set("Mica (in Water)")
        self.on_material_change()
        self.z_start_var.set("6.0")
        self.z_end_var.set("-3.0")
        self.freq_var.set("3.0")
        self.noise_var.set("0.1")
        self.z_mode.set("Sine")
        self.status_var.set("Ready")

    def get_force(self, h, R, A, E_s):
        h_eff = np.maximum(h, 0.1e-9) 
        f_vdw = - (A * R) / (6 * h_eff**2)
        f_rep = 0.0
        if h < self.sigma:
            indent = self.sigma - h
            E_star = E_s / (1 - self.nu_s**2)
            f_rep = (4/3) * E_star * np.sqrt(R) * (indent**1.5)
        return f_vdw + f_rep

    def solve_trajectory(self, z_total, v_total, k, R, A, E_s, gamma):
        """
        Includes Viscous Drag Force: 
        Equilibrium: k*d = F_ts(Z + d) + F_drag(v_piezo)
        """
        d_vals = []
        d_prev = 0.0
        for Z_val, V_val in zip(z_total, v_total):
            # F_drag = - gamma * V_val (approx v_piezo for simplicity)
            f_drag = - gamma * V_val
            
            d_cand = np.linspace(-15e-9, 45e-9, 1000)
            h_cand = Z_val + d_cand 
            
            f_ts = np.array([self.get_force(h, R, A, E_s) for h in h_cand])
            # Residual including drag
            res = k * d_cand - (f_ts + f_drag)
            
            crossings = np.where(np.diff(np.sign(res)))[0]
            if len(crossings) > 0:
                idx = crossings[np.argmin(np.abs(d_cand[crossings] - d_prev))]
                d_curr = d_cand[idx]
            else:
                d_curr = d_cand[np.argmin(np.abs(res))]
            d_vals.append(d_curr)
            d_prev = d_curr
        return np.array(d_vals)

    def save_to_csv(self):
        if self.last_results is None:
            return
            
        t, z_t, d_t, z_cycle, d_cycle, half, k, freq, mat_name = self.last_results
        
        # Prepare filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        clean_mat = mat_name.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"afm_data_{clean_mat}_{freq}Hz_{timestamp}.csv"
        
        try:
            with open(filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time (s)", "Support_Z (nm)", "Tip_Height_H (nm)", "Deflection (nm)"])
                
                # Convert to nm for easier reading
                tip_h_nm = (z_t + d_t + self.sigma) * 1e9
                support_z_nm = (z_t + self.sigma) * 1e9
                deflection_nm = d_t * 1e9
                
                for i in range(len(t)):
                    writer.writerow([
                        f"{t[i]:.6f}", 
                        f"{support_z_nm[i]:.4f}", 
                        f"{tip_h_nm[i]:.4f}", 
                        f"{deflection_nm[i]:.4f}"
                    ])
            
            self.status_var.set(f"Saved: {filename}")
            self.status_label.config(foreground="darkgreen")
        except Exception as e:
            self.status_var.set(f"Save Error: {e}")

    def start_calculation(self):
        self.status_var.set("Calculating...")
        self.status_label.config(foreground="red")
        self.calc_button.config(state=tk.DISABLED)
        self.root.update_idletasks()
        thread = threading.Thread(target=self.perform_calc)
        thread.start()

    def perform_calc(self):
        try:
            k = float(self.k_var.get())
            R = float(self.r_var.get()) * 1e-9
            gamma = float(self.visc_var.get()) * 1e-9 # nN.s/m to N.s/m
            z_s_nm = float(self.z_start_var.get())
            z_e_nm = float(self.z_end_var.get())
            freq = float(self.freq_var.get())
            pts_per_cycle = int(self.steps_var.get())
            noise_nm = float(self.noise_var.get())
            z_mode = self.z_mode.get()
            
            E_s = float(self.e_var.get()) * 1e9
            A = float(self.a_var.get()) * 1e-21
            
            z_s = (z_s_nm - self.sigma*1e9) * 1e-9
            z_e = (z_e_nm - self.sigma*1e9) * 1e-9
            
            total_duration = 1.0
            total_points = int(pts_per_cycle * freq * total_duration)
            t = np.linspace(0, total_duration, total_points)
            dt = t[1] - t[0]
            
            center = (z_s + z_e) / 2
            amp = abs(z_s - z_e) / 2
            if z_mode == "Sine":
                z_t = center + amp * np.cos(2 * np.pi * freq * t)
            else:
                z_t = center + amp * (2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1)
            
            # Velocity for drag (v = dz/dt)
            v_z = np.gradient(z_t, dt)
            
            d_t = self.solve_trajectory(z_t + self.sigma, v_z, k, R, A, E_s, gamma)
            
            if noise_nm > 0:
                d_t += np.random.normal(0, noise_nm * 1e-9, d_t.size)

            n_cycle = int(pts_per_cycle)
            half = n_cycle // 2
            z_cycle = z_t[:n_cycle] + self.sigma
            d_cycle = d_t[:n_cycle]
            
            self.last_results = (t, z_t, d_t, z_cycle, d_cycle, half, k, freq, self.mat_choice.get())
            
            self.root.after(0, self.refresh_plot)
            self.root.after(0, lambda: self.save_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_var.set("Calculation Done"))
            self.root.after(0, lambda: self.status_label.config(foreground="green"))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error: {e}"))
        finally:
            self.root.after(0, lambda: self.calc_button.config(state=tk.NORMAL))

    def refresh_plot(self):
        if self.last_results is None: return
        self.update_ui(*self.last_results)

    def update_ui(self, t, z_t, d_t, z_cycle, d_cycle, half, k, freq, mat_name):
        self.ax.clear()
        mode = self.view_mode.get()
        
        if mode == "ForceCurve":
            self.ax.plot(z_cycle[:half] * 1e9, d_cycle[:half] * 1e9, 'red', lw=1, label="Approach")
            self.ax.plot(z_cycle[half:] * 1e9, d_cycle[half:] * 1e9, 'blue', linestyle='--', alpha=0.6, label="Retract")
            self.ax.set_title(f"Liquid Force Curve: {mat_name}\n(Drag included @ {freq}Hz)")
            self.ax.set_xlabel("Support Z Pos [nm]")
            self.ax.set_ylabel("Deflection [nm]")
        
        else:
            tip_h = (z_t + d_t + self.sigma) * 1e9
            support_h = (z_t + self.sigma) * 1e9
            deflection_nm = d_t * 1e9
            
            self.ax.plot(t, support_h, color='gray', linestyle=':', alpha=0.5, label="Support Z")
            self.ax.plot(t, tip_h, 'red', lw=1.2, label="Tip Height")
            self.ax.plot(t, deflection_nm, 'green', lw=1, alpha=0.8, label="Deflection (Signal)")
            
            self.ax.set_title(f"Liquid Trajectory: {mat_name} ({freq}Hz)")
            self.ax.set_xlabel("Time [s]")
            self.ax.set_ylabel("Height / Deflection [nm]")
            
            all_vals = np.concatenate([tip_h, support_h, deflection_nm])
            self.ax.set_ylim(np.min(all_vals)-1, np.max(all_vals)+1)

        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='best', fontsize='small')
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AFMPanelApp(root)
    root.geometry("1000x700")
    root.mainloop()
