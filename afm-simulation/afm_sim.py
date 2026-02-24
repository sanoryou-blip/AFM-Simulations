#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

class AFMSimulator:
    def __init__(self, k=0.2, radius_nm=30.0):
        # 基本パラメータ
        self.k = k  # バネ定数 [N/m]
        self.R = radius_nm * 1e-9  # 探針半径 [m]
        
        # レナードジョーンズに関連する定数 (標準的な値)
        self.A = 1e-19  # ハマカー定数 [J] (引力項の強さ)
        self.sigma = 0.34e-9  # 原子サイズ [m]
        self.h_min = 0.15e-9  # 計算上の最小距離 (衝突回避用)

    def force_ts(self, h):
        """探針-試料間の力を計算 (Lennard-Jones型)"""
        h = np.maximum(h, self.h_min)
        # 引力項 (Van der Waals)
        f_attr = - (self.A * self.R) / (6 * h**2)
        # 斥力項 (Pauli repulsion)
        f_rep = (self.A * self.R * self.sigma**6) / (180 * h**8)
        return f_attr + f_rep

    def solve_equilibrium(self, z_support, x_start):
        """
        支持部の高さ z に対して、カ ンチレバーのたわみ x を求める
        k * x + F_ts(z - x) = 0  となる x を探す
        """
        x = x_start
        # 簡易的な反復法 (ニュートン法に近い形) で平衡点を追跡
        for _ in range(100):
            h = z_support - x
            f_ext = self.force_ts(h)
            residual = self.k * x + f_ext
            
            # 傾き (k - dF/dh)
            # 簡易化のため数値微分
            dh = 1e-12
            df_dh = (self.force_ts(h + dh) - self.force_ts(h)) / dh
            slope = self.k - df_dh
            
            dx = -residual / slope
            x += dx
            if abs(dx) < 1e-15:
                break
        return x

    def simulate(self, z_range):
        """一連の支持部高さに対して、たわみを計算する"""
        x_values = []
        h_values = []
        x_current = 0.0 # 初期値（遠方ではたわみゼロ）
        
        for z in z_range:
            x_current = self.solve_equilibrium(z, x_current)
            x_values.append(x_current)
            h_values.append(z - x_current)
            
        return np.array(x_values), np.array(h_values)

def main():
    # シミュレーションの設定
    sim = AFMSimulator(k=0.2, radius_nm=30.0)
    
    # 支持部の移動範囲: 15nm から -2nm まで往復
    z_far = 15e-9
    z_near = -1e-9
    points = 1000
    
    z_approach = np.linspace(z_far, z_near, points)
    z_retract = np.linspace(z_near, z_far, points)
    
    print("Simulating Approach...")
    x_app, h_app = sim.simulate(z_approach)
    
    print("Simulating Retract...")
    x_ret, h_ret = sim.simulate(z_retract)
    
    # グラフ描画
    plt.figure(figsize=(8, 6))
    plt.plot(z_approach * 1e9, x_app * 1e9, label='Approach (接近)', color='red', lw=2)
    plt.plot(z_retract * 1e9, x_ret * 1e9, label='Retract (離脱)', color='blue', lw=2, linestyle='--')
    
    plt.title("AFM Force-Curve Simulation (R=30nm, k=0.2N/m)")
    plt.xlabel("Support Position z (nm)")
    plt.ylabel("Cantilever Deflection x (nm)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    # AFMの慣習に合わせてY軸（たわみ）を反転させることが多い
    plt.gca().invert_yaxis()
    
    # 保存
    plt.savefig("afm_plot.png")
    print("Done! Plot saved as 'afm_plot.png'")
    # plt.show() # 必要に応じて

if __name__ == "__main__":
    main()
