import numpy as np
import matplotlib.pyplot as plt
import math

# Define o número de amostras para cada variável aleatória
N = 10**6  # 1.000.000 de pontos

# --- Geração das amostras ---
samples1 = np.random.uniform(0, np.pi, N)
samples2 = np.random.normal(0, 1, N)
X = np.random.normal(0, 1, N)
Y = np.random.normal(0, 1, N)
samples3 = X + Y

# Estatísticas
print("Uniform[0,π]: mean = {:.4f}, var = {:.4f}".format(samples1.mean(), samples1.var()))
print("Normal(0,1):  mean = {:.4f}, var = {:.4f}".format(samples2.mean(), samples2.var()))
print("Z = X + Y:    mean = {:.4f}, var = {:.4f}".format(samples3.mean(), samples3.var()))

# --- Preparação das curvas teóricas ---

# 1) Uniforme estendido
x1 = np.linspace(-1, np.pi+1, 300)                             # grid de -1 a π+1
pdf1 = np.where((x1 >= 0) & (x1 <= np.pi), 1/np.pi, 0)         # 1/π em [0,π], 0 fora
cdf1 = np.clip(x1 / np.pi, 0, 1)                               # linear em [0,π], platô em 0 e 1

# 2) Normal(0,1)
x2 = np.linspace(-5, 5, 200)
pdf2 = 1/np.sqrt(2*np.pi) * np.exp(-x2**2 / 2)
erf_vec = np.vectorize(math.erf)
cdf2 = 0.5 * (1 + erf_vec(x2 / np.sqrt(2)))

# 3) Z ~ N(0,2)
x3 = np.linspace(-5, 5, 200)
sigma3 = np.sqrt(2)
pdf3 = 1/(sigma3 * np.sqrt(2*np.pi)) * np.exp(-x3**2 / (2 * sigma3**2))
cdf3 = 0.5 * (1 + erf_vec(x3 / (sigma3 * np.sqrt(2))))

# --- Plotagem ---
fig, ax = plt.subplots(3, 2, figsize=(12, 12))

# PDF Uniform
ax[0, 0].hist(samples1, bins=100, density=True, alpha=0.6, label="Empírica")
ax[0, 0].plot(x1, pdf1, 'r-', label="Teórica")
ax[0, 0].set_title("PDF Uniform[0,π]")
ax[0, 0].legend()

# CDF Uniform
s1 = np.sort(samples1)
ax[0, 1].plot(s1, np.arange(1, N+1)/N, label="Empírica")
ax[0, 1].plot(x1, cdf1, 'r-', label="Teórica")
ax[0, 1].set_title("CDF Uniform[0,π]")
ax[0, 1].legend()

# PDF Normal
ax[1, 0].hist(samples2, bins=100, density=True, alpha=0.6, label="Empírica")
ax[1, 0].plot(x2, pdf2, 'r-', label="Teórica")
ax[1, 0].set_title("PDF N(0,1)")
ax[1, 0].legend()

# CDF Normal
s2 = np.sort(samples2)
ax[1, 1].plot(s2, np.arange(1, N+1)/N, label="Empírica")
ax[1, 1].plot(x2, cdf2, 'r-', label="Teórica")
ax[1, 1].set_title("CDF N(0,1)")
ax[1, 1].legend()

# PDF Z = X+Y
ax[2, 0].hist(samples3, bins=100, density=True, alpha=0.6, label="Empírica")
ax[2, 0].plot(x3, pdf3, 'r-', label="Teórica (N(0,2))")
ax[2, 0].set_title("PDF Z = X + Y")
ax[2, 0].legend()

# CDF Z = X+Y
s3 = np.sort(samples3)
ax[2, 1].plot(s3, np.arange(1, N+1)/N, label="Empírica")
ax[2, 1].plot(x3, cdf3, 'r-', label="Teórica (N(0,2))")
ax[2, 1].set_title("CDF Z = X + Y")
ax[2, 1].legend()

plt.tight_layout()
plt.show()
