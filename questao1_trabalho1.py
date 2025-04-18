import numpy as np
import matplotlib.pyplot as plt

# Funções definidas
def rect_pulse(t, t0, Tb):
    return np.where(np.abs((t - t0) / Tb) <= 0.5, 1, 0)

def tri_pulse(t, t0, Tb):
    x = (t - t0) / Tb
    return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)

# Eixo do tempo
t = np.linspace(-5, 5, 1000)

# Parâmetros dos casos (a) e (b)
params = [(1, 0.5), (2, 3)]
labels = ['(a) Tb=1, t0=0.5', '(b) Tb=2, t0=3']

# Plotagem
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
for i, (Tb, t0) in enumerate(params):
    rect = rect_pulse(t, t0, Tb)
    tri = tri_pulse(t, t0, Tb)

    axes[i][0].plot(t, rect)
    axes[i][0].set_title(f'Pulso Retangular {labels[i]}')
    axes[i][0].set_xlabel('t')
    axes[i][0].set_ylabel('Amplitude')
    axes[i][0].grid(True)

    axes[i][1].plot(t, tri)
    axes[i][1].set_title(f'Pulso Triangular {labels[i]}')
    axes[i][1].set_xlabel('t')
    axes[i][1].set_ylabel('Amplitude')
    axes[i][1].grid(True)

plt.tight_layout()
plt.show()
