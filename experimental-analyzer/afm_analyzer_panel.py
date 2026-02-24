import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog
import csv
import os

class AFMAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AFM Data Analyzer PRO")
        
        # --- Taskbar Separation (Windows) ---
        try:
            import ctypes
            myappid = 'sanoryou.afmsim.analyzer.v1'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            
            icon_path = os.path.join(os.path.dirname(__file__), "analyzer_icon.png")
            if os.path.exists(icon_path):
                self.icon_img = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(False, self.icon_img)
        except Exception:
            pass

        # --- Data Storage ---
        self.raw_data = None  # Original loaded columns
        self.processed_data = None
        self.file_path = None
        
        # --- UI Variables ---
        self.def_sens_var = tk.StringVar(value="50.0")  # nm/V
        self.piezo_sens_var = tk.StringVar(value="100.0") # nm/V
        self.k_var = tk.StringVar(value="1.0")        # N/m
        
        self.col_time = tk.IntVar(value=0)
        self.col_piezo = tk.IntVar(value=1)
        self.col_def = tk.IntVar(value=2)
        
        self.status_var = tk.StringVar(value="Please load an experimental CSV/TXT file.")
        
        # New: Oscilloscope/Manual controls
        self.skip_lines_var = tk.StringVar(value="0")
        self.inv_piezo_var = tk.BooleanVar(value=False)
        self.inv_def_var = tk.BooleanVar(value=False)
        
        self.setup_ui()

    def setup_ui(self):
        self.main_frame = ttk.Frame(self.root, padding="15")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Sidebar (Controls) ---
        self.sidebar = ttk.Frame(self.main_frame, width=300)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))

        # 1. File Loading
        load_frame = ttk.LabelFrame(self.sidebar, text="1. Data Input", padding="10")
        load_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(load_frame, text="OPEN EXPERIMENTAL FILE", command=self.load_file).pack(fill=tk.X, pady=5)
        self.file_label = ttk.Label(load_frame, text="No file selected", font=("Helvetica", 8, "italic"), wraplength=250)
        self.file_label.pack(fill=tk.X)

        # 2. Column Mapping
        col_frame = ttk.LabelFrame(self.sidebar, text="2. Column Indices (0-based)", padding="10")
        col_frame.pack(fill=tk.X, pady=5)
        
        def add_col_entry(parent, label, var):
            f = ttk.Frame(parent)
            f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=label).pack(side=tk.LEFT)
            ttk.Entry(f, textvariable=var, width=5, justify=tk.RIGHT).pack(side=tk.RIGHT)

        add_col_entry(col_frame, "Time Col:", self.col_time)
        add_col_entry(col_frame, "Piezo (V) Col:", self.col_piezo)
        add_col_entry(col_frame, "Deflection (V) Col:", self.col_def)
        
        ttk.Label(col_frame, text="Skip Meta Lines:").pack(anchor=tk.W, pady=(5,0))
        ttk.Entry(col_frame, textvariable=self.skip_lines_var, width=5).pack(anchor=tk.W)
        
        ttk.Checkbutton(col_frame, text="Invert Piezo Signal", variable=self.inv_piezo_var).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(col_frame, text="Invert Deflection Signal", variable=self.inv_def_var).pack(anchor=tk.W, pady=2)

        # 3. Calibration Parameters
        cal_frame = ttk.LabelFrame(self.sidebar, text="3. Calibration Specs", padding="10")
        cal_frame.pack(fill=tk.X, pady=5)
        
        def add_val_entry(parent, label, var):
            f = ttk.Frame(parent)
            f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=label).pack(side=tk.LEFT)
            ttk.Entry(f, textvariable=var, width=8, justify=tk.RIGHT).pack(side=tk.RIGHT)

        add_val_entry(cal_frame, "Def Sens [nm/V]:", self.def_sens_var)
        add_val_entry(cal_frame, "Piezo Sens [nm/V]:", self.piezo_sens_var)
        add_val_entry(cal_frame, "Spring Const [N/m]:", self.k_var)

        # 4. Actions
        ttk.Button(self.sidebar, text="RE-PROCESS & PLOT", command=self.process_data, style="Accent.TButton").pack(fill=tk.X, pady=20)
        
        self.status_label = ttk.Label(self.sidebar, textvariable=self.status_var, wraplength=250, foreground="blue")
        self.status_label.pack(fill=tk.X)

        # --- Right Side (Visualization) ---
        self.viz_frame = ttk.Frame(self.main_frame)
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.viz_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV/Text files", "*.csv *.txt *.dat"), ("All files", "*.*")])
        if not path: return
        
        self.file_path = path
        self.file_label.config(text=os.path.basename(path))
        
        try:
            skip = int(self.skip_lines_var.get())
            
            # Robust reading for Oscilloscope CSVs
            # First, check delimiter
            with open(path, 'r') as f:
                for _ in range(skip): f.readline()
                first_data_line = f.readline()
                if ',' in first_data_line: delimiter = ','
                elif '\t' in first_data_line: delimiter = '\t'
                else: delimiter = None # Space or fixed width

            # Use genfromtxt with manual skip
            data = np.genfromtxt(path, delimiter=delimiter, skip_header=skip)
            
            # If multiple headers exist (like in some Osc formats), genfromtxt might return nans
            # We try to filter rows that are strictly numeric
            mask = ~np.isnan(data).any(axis=1)
            self.raw_data = data[mask]
            
            if self.raw_data.size == 0:
                raise ValueError("No numeric data found. Check 'Skip Meta Lines'.")

            self.status_var.set(f"Loaded {self.raw_data.shape[0]} rows.")
            self.process_data()
            
        except Exception as e:
            self.status_var.set(f"Load Error: {e}")

    def process_data(self):
        if self.raw_data is None: return
        
        try:
            c_t = self.col_time.get()
            c_p = self.col_piezo.get()
            c_d = self.col_def.get()
            
            t = self.raw_data[:, c_t]
            p_v = self.raw_data[:, c_p]
            d_v = self.raw_data[:, c_d]
            
            # Apply Inversions
            if self.inv_piezo_var.get(): p_v = -p_v
            if self.inv_def_var.get(): d_v = -d_v
            
            d_sens = float(self.def_sens_var.get())
            p_sens = float(self.piezo_sens_var.get())
            k = float(self.k_var.get())
            
            # --- Conversions ---
            # 1. Deflection in nm
            deflection_nm = d_v * d_sens
            
            # 2. Piezo Position in nm
            piezo_nm = p_v * p_sens
            
            # 3. Base Correction (Simple: set starting deflection to zero)
            deflection_nm -= np.mean(deflection_nm[:50])
            
            # Plot 1: Raw Signal over Time
            self.ax1.clear()
            self.ax1.plot(t, d_v, 'g-', label="Deflection Signal (V)")
            self.ax1.set_ylabel("Deflection [V]")
            self.ax1.set_title("Experimental Raw Data")
            self.ax1.legend(loc="upper right")
            self.ax1.grid(True, alpha=0.3)
            
            ax1_piezo = self.ax1.twinx()
            ax1_piezo.plot(t, p_v, 'b--', alpha=0.5, label="Piezo (V)")
            ax1_piezo.set_ylabel("Piezo [V]", color='b')
            
            # Plot 2: Force Curve (Processed)
            self.ax2.clear()
            # Split into Approach/Retract based on Piezo slope
            velocity = np.gradient(piezo_nm)
            idx_app = velocity < 0
            idx_ret = velocity >= 0
            
            self.ax2.plot(piezo_nm[idx_app], deflection_nm[idx_app], 'r-', label="Approach")
            self.ax2.plot(piezo_nm[idx_ret], deflection_nm[idx_ret], 'b--', alpha=0.6, label="Retract")
            
            self.ax2.set_title("Processed Force Curve")
            self.ax2.set_xlabel("Piezo Position [nm]")
            self.ax2.set_ylabel("Deflection [nm]")
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            self.status_var.set("Processing complete.")
            
        except Exception as e:
            self.status_var.set(f"Processing Error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    # Apply a slightly better style if available
    style = ttk.Style()
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    
    app = AFMAnalyzerApp(root)
    root.geometry("1100x800")
    root.mainloop()
