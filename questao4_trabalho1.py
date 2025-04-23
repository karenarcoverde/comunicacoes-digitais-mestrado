import numpy as np
import matplotlib.pyplot as plt

# Parâmetros gerais
d = 2                   # distância entre níveis de símbolo (±d/2)
N = 10**6               # número de símbolos transmitidos
symbols = np.array([-d/2, d/2])
Eb = d**2 / 8           # energia por bit

# --- (a) Para σ = 1: gerar x[n], v[n], y[n] e plotar ---
sigma = 1
# 1. Sequência de transmissão x[n]
x = np.random.choice(symbols, size=N)
# 2. Ruído AWGN v[n] ~ N(0, σ^2)
v = np.random.normal(0, sigma, size=N)
# 3. Sinal recebido y[n] = x[n] + v[n]
y = x + v
# 4. Plot das primeiras 100 amostras
plt.figure(); plt.plot(x[:100]); plt.title('x[n] (σ=1)'); plt.xlabel('n'); plt.ylabel('x[n]')
plt.figure(); plt.plot(v[:100]); plt.title('v[n] (σ=1)'); plt.xlabel('n'); plt.ylabel('v[n]')
plt.figure(); plt.plot(y[:100]); plt.title('y[n] (σ=1)'); plt.xlabel('n'); plt.ylabel('y[n]')
# 5. Histogramas de x, v, y
plt.figure(); plt.hist(x, bins=2, density=True); plt.title('Histograma x[n]')
plt.figure(); plt.hist(v, bins=100, density=True); plt.title('Histograma v[n]')
plt.figure(); plt.hist(y, bins=100, density=True); plt.title('Histograma y[n]')
plt.show()
# 6. Decisão x̂[n] com limiar em zero
x_hat = np.where(y >= 0,  d/2, -d/2)
# 7. Taxa de erro de símbolo (SER)
ser_list = [np.mean(x_hat != x)]
print(ser_list)