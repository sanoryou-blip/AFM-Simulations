import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
import threading
import csv
import time
import os

class AFMPanelApp:
    def __init__(self, root):
        self.root = root
        # --- Liquid Material Presets (E [GPa], Eps [zJ], Sigma [nm], Rho_t, Rho_s) ---
        # Adapted from A ~ pi^2 * rho^2 * 4 * eps * sigma^6
        self.materials = {
            "Mica (in Water)": (60.0, 3.95, 0.34, 1e28, 1e28),    
            "Silicon (in Water)": (160.0, 5.92, 0.34, 1e28, 1e28),
            "Cell (Soft)": (0.0001, 0.79, 0.34, 1e28, 1e28),   
            "Lipid Bilayer": (0.02, 1.97, 0.34, 1e28, 1e28)    
        }
        
        # --- Data Storage ---
        self.last_results = None 
        
        # --- UI Variables ---
        self.view_mode = tk.StringVar(value="ForceCurve")
        self.z_mode = tk.StringVar(value="Sine")
        self.mat_choice = tk.StringVar(value="Mica (in Water)")
        
        # Cantilever
        self.k_var = tk.StringVar(value="1.0")
        self.r_var = tk.StringVar(value="10.0")
        self.visc_var = tk.StringVar(value="5.0")
        
        # Microscopic / Material
        self.e_var = tk.StringVar(value="60.0")
        self.eps_var = tk.StringVar(value="3.95")
        self.sig_var = tk.StringVar(value="0.34")
        self.rho_t_var = tk.StringVar(value="1e28")
        self.rho_s_var = tk.StringVar(value="1e28")
        
        # Scan
        self.z_start_var = tk.StringVar(value="6.0")
        self.z_end_var = tk.StringVar(value="-3.0")
        self.freq_var = tk.StringVar(value="3.0")
        self.steps_var = tk.StringVar(value="500")
        self.duration_var = tk.StringVar(value="0.1")
        self.noise_var = tk.StringVar(value="0.1")
        
        self.status_var = tk.StringVar(value="Ready (Liquid Mode)")
        
        # --- UI Setup ---
        self.setup_ui()
        self.on_material_change() # Sync presets

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
        
        def add_entry(parent, label_text, var):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=4)
            ttk.Label(frame, text=label_text).pack(side=tk.LEFT)
            entry = ttk.Entry(frame, textvariable=var, width=10, justify=tk.RIGHT)
            entry.pack(side=tk.RIGHT)
            return entry

        # --- Section: Cantilever ---
        add_entry(self.ctrl_frame, "Spring Const k [N/m]:", self.k_var)
        add_entry(self.ctrl_frame, "Tip Radius R [nm]:", self.r_var)
        add_entry(self.ctrl_frame, "Viscosity Cnt [nN·s/m]:", self.visc_var)
        
        ttk.Separator(self.ctrl_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        # --- Section: Material ---
        ttk.Label(self.ctrl_frame, text="Sample Preset (Liquid):", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        self.mat_combo = ttk.Combobox(self.ctrl_frame, textvariable=self.mat_choice, values=list(self.materials.keys()), state="readonly")
        self.mat_combo.pack(fill=tk.X, pady=5)
        self.mat_combo.bind("<<ComboboxSelected>>", self.on_material_change)
        
        add_entry(self.ctrl_frame, "Young's Modulus [GPa]:", self.e_var)
        
        # Microscopic LJ Parameters
        ttk.Label(self.ctrl_frame, text="Microscopic Interactions:", font=("Helvetica", 9, "italic")).pack(anchor=tk.W, pady=(5,0))
        add_entry(self.ctrl_frame, "L-J Epsilon [zJ]:", self.eps_var)
        add_entry(self.ctrl_frame, "L-J Sigma [nm]:", self.sig_var)
        add_entry(self.ctrl_frame, "Rho Tip [1/m^3]:", self.rho_t_var)
        add_entry(self.ctrl_frame, "Rho Sample [1/m^3]:", self.rho_s_var)
        
        ttk.Separator(self.ctrl_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # --- Section: Scan ---
        add_entry(self.ctrl_frame, "Scan Start [nm]:", self.z_start_var)
        add_entry(self.ctrl_frame, "Scan End [nm]:", self.z_end_var)
        add_entry(self.ctrl_frame, "Frequency [Hz]:", self.freq_var)
        add_entry(self.ctrl_frame, "Pts per Cycle:", self.steps_var)
        add_entry(self.ctrl_frame, "Duration [s]:", self.duration_var)
        add_entry(self.ctrl_frame, "Noise Level [nm]:", self.noise_var)

        # --- Z Ramp Mode ---
        ttk.Label(self.ctrl_frame, text="Z Ramp Mode:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(10,0))
        ttk.Radiobutton(self.ctrl_frame, text="Linear (Triangular)", variable=self.z_mode, value="Linear").pack(anchor=tk.W)
        ttk.Radiobutton(self.ctrl_frame, text="Sine (Oscillation)", variable=self.z_mode, value="Sine").pack(anchor=tk.W)

        # --- View Control ---
        ttk.Label(self.ctrl_frame, text="Display Mode:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(10,0))
        ttk.Radiobutton(self.ctrl_frame, text="Force Curve (d vs Z)", variable=self.view_mode, value="ForceCurve", command=self.refresh_plot).pack(anchor=tk.W)
        ttk.Radiobutton(self.ctrl_frame, text="Time Trajectory", variable=self.view_mode, value="Trajectory", command=self.refresh_plot).pack(anchor=tk.W)

        # Status
        self.status_label = ttk.Label(self.ctrl_frame, textvariable=self.status_var, foreground="blue", font=("Helvetica", 10, "bold"))
        self.status_label.pack(pady=10)

        # Buttons
        self.calc_button = ttk.Button(self.ctrl_frame, text="RUN CALCULATION", command=self.start_calculation)
        self.calc_button.pack(fill=tk.X, pady=5)
        
        self.save_button = ttk.Button(self.ctrl_frame, text="SAVE DATA TO CSV", command=self.save_to_csv, state=tk.DISABLED)
        self.save_button.pack(fill=tk.X, pady=5)
        
        self.save_png_button = ttk.Button(self.ctrl_frame, text="SAVE PLOT AS PNG", command=self.save_plot_png)
        self.save_png_button.pack(fill=tk.X, pady=5)
        
        ttk.Button(self.ctrl_frame, text="Reset to Default", command=self.reset_params).pack(fill=tk.X, pady=5)

        # --- Right Side: Plot (Responsive) ---
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig, self.ax = plt.subplots(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add Navigation Toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def on_material_change(self, event=None):
        mat_name = self.mat_choice.get()
        if mat_name in self.materials:
            E, eps, sig, rt, rs = self.materials[mat_name]
            self.e_var.set(str(E))
            self.eps_var.set(str(eps))
            self.sig_var.set(str(sig))
            self.rho_t_var.set(str(rt))
            self.rho_s_var.set(str(rs))

    def reset_params(self):
        self.k_var.set("1.0")
        self.r_var.set("10.0")
        self.visc_var.set("5.0")
        self.mat_choice.set("Mica (in Water)")
        self.on_material_change()
        self.z_start_var.set("6.0")
        self.z_end_var.set("-3.0")
        self.freq_var.set("3.0")
        self.duration_var.set("0.1")
        self.noise_var.set("0.1")
        self.z_mode.set("Sine")
        self.status_var.set("Ready")

    def get_force(self, h, R, Ah, Bh, E_s, sigma):
        """
        Calculates Tip-Sample Force (L-J + Hertz):
        Ah: Attraction coefficient
        Bh: Repulsion coefficient
        sigma: distance where repulsion starts
        """
        h_eff = np.maximum(h, 0.15e-9) # Minimum cutoff
        
        # 1. LJ Force: -Ah*R/(6*h^2) + Bh*R/(180*h^8)
        f_lj = - (Ah * R) / (6 * h_eff**2) + (Bh * R) / (180 * h_eff**8)
        
        # 2. Hertzian Repulsion (Positive)
        f_hertz = 0.0
        if h < sigma:
            indent = sigma - h
            E_star = E_s / (1 - 0.3**2) # using nu_s = 0.3
            f_hertz = (4/3) * E_star * np.sqrt(R) * (indent**1.5)
        
        return f_lj + f_hertz

    def solve_trajectory(self, z_total, v_total, k, R, Ah, Bh, E_s, gamma, sigma):
        d_vals = []
        d_prev = 0.0
        
        for Z_val, V_val in zip(z_total, v_total):
            d_cand = np.linspace(-10e-9, 40e-9, 1200)
            h_cand = Z_val + d_cand
            
            # Drag force acts on the cantilever: F_drag = -gamma * (V_support + V_deflection)
            # Since V_deflection is small relative to piezo in HS-AFM, we approximate with V_support
            f_drag = -gamma * V_val
            
            # Balance: k*d = F_ts(Z+d) + F_drag
            forces = np.array([self.get_force(h, R, Ah, Bh, E_s, sigma) for h in h_cand])
            res = k * d_cand - (forces + f_drag)
            
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
        if self.last_results is None: return
        t, z_t, d_t, z_cycle, d_cycle, half, k, freq, mat_name = self.last_results
        sig_current = float(self.sig_var.get()) * 1e-9
        
        # Prepare filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        clean_mat = mat_name.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"afm_data_{clean_mat}_{freq}Hz_{timestamp}.csv"
        
        try:
            with open(filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time (s)", "Support_Z (nm)", "Tip_Height_H (nm)", "Deflection (nm)"])
                
                # Convert to nm for easier reading
                tip_h_nm = (z_t + d_t + sig_current) * 1e9
                support_z_nm = (z_t + sig_current) * 1e9
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
            self.status_label.config(foreground="red")

    def save_plot_png(self):
        mat_name = self.mat_choice.get().replace(" ", "_").replace("(", "").replace(")", "")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"afm_plot_{mat_name}_{timestamp}.png"
        try:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            self.status_var.set(f"Plot saved: {filename}")
            self.status_label.config(foreground="darkgreen")
        except Exception as e:
            self.status_var.set(f"PNG Save Error: {e}")
            self.status_label.config(foreground="red")

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
            eps = float(self.eps_var.get()) * 1e-21
            sig = float(self.sig_var.get()) * 1e-9
            rt = float(self.rho_t_var.get())
            rs = float(self.rho_s_var.get())
            
            # Calculate A_h and B_h like Prof. Uchihashi
            c6 = 4.0 * eps * sig**6
            c12 = 4.0 * eps * sig**12
            pref = np.pi**2 * rt * rs
            Ah = pref * c6
            Bh = pref * c12
            
            z_s = (z_s_nm - sig*1e9) * 1e-9
            z_e = (z_e_nm - sig*1e9) * 1e-9
            
            total_duration = float(self.duration_var.get())
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
            
            d_t = self.solve_trajectory(z_t + sig, v_z, k, R, Ah, Bh, E_s, gamma, sig)
            
            if noise_nm > 0:
                d_t += np.random.normal(0, noise_nm * 1e-9, d_t.size)

            n_cycle = int(pts_per_cycle)
            half = n_cycle // 2
            z_cycle = z_t[:n_cycle] + sig
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
        sig_current = float(self.sig_var.get()) * 1e-9
        
        if mode == "ForceCurve":
            self.ax.plot(z_cycle[:half] * 1e9, d_cycle[:half] * 1e9, 'red', lw=1, label="Approach")
            self.ax.plot(z_cycle[half:] * 1e9, d_cycle[half:] * 1e9, 'blue', linestyle='--', alpha=0.6, label="Retract")
            self.ax.set_title(f"Liquid Force Curve: {mat_name}\n(Drag included @ {freq}Hz)")
            self.ax.set_xlabel("Support Z Pos [nm]")
            self.ax.set_ylabel("Deflection [nm]")
        
        else:
            tip_h = (z_t + d_t + sig_current) * 1e9
            support_h = (z_t + sig_current) * 1e9
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
