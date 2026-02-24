import matplotlib.pyplot as plt
import numpy as np

def create_analyzer_icon(filename="analyzer_icon.png"):
    fig = plt.figure(figsize=(2, 2), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Background
    ax.set_facecolor('#1e1e1e')
    
    # Draw a stylized force curve
    z = np.linspace(0, 10, 100)
    # Approach
    ax.plot(z[:50], np.zeros(50), color='#00f2ff', lw=8, alpha=0.8)
    ax.plot(z[50:], (z[50:]-5)**2, color='#00f2ff', lw=8, alpha=0.8)
    
    # Retract
    ax.plot(z[40:], (z[40:]-4)**2 + 0.5, color='#ff007c', lw=8, alpha=0.6, linestyle='--')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 15)
    ax.axis('off')
    
    plt.savefig(filename, transparent=False)
    plt.close()
    print(f"Icon created: {filename}")

if __name__ == "__main__":
    create_analyzer_icon()
