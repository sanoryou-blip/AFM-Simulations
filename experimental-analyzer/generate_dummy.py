import numpy as np
import csv

def generate_dummy_afm_data(filename="dummy_experimental_data.csv"):
    # Parameters
    pts = 2000
    t = np.linspace(0, 1, pts)
    
    # Simulate a Piezo Sine Wave (V)
    # Let's say 0 to 10V
    piezo_v = 5 + 5 * np.cos(2 * np.pi * 1 * t)
    
    # Simulate Deflection (V)
    # Sensitivity nm/V = 50, Piezo nm/V = 100
    # Contact at piezo < 3V (roughly)
    deflection_v = np.zeros(pts)
    
    # Simple hard contact model
    for i in range(pts):
        v = piezo_v[i]
        if v < 3.0:
            # Force contact (Z + d = const)
            # In V: (v * 100 + d * 50) = constant?
            # Rough approximation for dummy data:
            deflection_v[i] = (3.0 - v) * 2.0 # 2.0 V/V slope
            
    # Add noise
    deflection_v += np.random.normal(0, 0.05, pts)
    
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time(s)", "Piezo(V)", "Deflection(V)"])
        for i in range(pts):
            writer.writerow([f"{t[i]:.6f}", f"{piezo_v[i]:.4f}", f"{deflection_v[i]:.4f}"])
    
    print(f"Generated {filename}")

if __name__ == "__main__":
    generate_dummy_afm_data()
