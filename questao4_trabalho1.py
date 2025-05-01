import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as ticker

### a-d
# 1) Geração dos dados
d      = 2
N      = 10**6
symbols = np.array([-d/2, d/2])
sigma_list = [1, 0.5, 0.25, 0.125]

for sigma in sigma_list:
    x = np.random.choice(symbols, size=N)      # PMF discreta
    v = np.random.normal(0, sigma, size=N)     # PDF contínua
    y = x + v

    # ------------------------------------------------------------
    # 2) Primeiras 100 amostras
    plt.figure(figsize=(6,4))
    plt.subplot(3,1,1)
    plt.plot(x[:100], '.', markersize=4)
    plt.title(f'x[n] (σ={sigma})'); plt.ylabel('x[n]')
    plt.subplot(3,1,2)
    plt.plot(v[:100], '.', markersize=4)
    plt.title(f'v[n] (σ={sigma})'); plt.ylabel('v[n]')
    plt.subplot(3,1,3)
    plt.plot(y[:100], '.', markersize=4)
    plt.title(f'y[n] (σ={sigma})'); plt.xlabel('n'); plt.ylabel('y[n]')
    plt.tight_layout()

    # ------------------------------------------------------------
    # 3) PMF de x[n]
    # Cálculo da PMF
    vals, cnts = np.unique(x, return_counts=True)
    probs = cnts / cnts.sum()

    # Plot com barras estreitas em -1 e +1
    plt.figure(figsize=(4, 2.5))
    plt.bar(vals, probs, width=0.4, align='center', alpha=0.7)
    plt.xticks(vals)
    plt.ylim(0, 1.1)
    plt.title(f'PMF de x[n] (σ={sigma})')
    plt.xlabel('x[n]')
    plt.ylabel('P(x)')
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 4) PDF empírica de v[n] e y[n]
    for data, name in [(v,'v[n]'), (y,'y[n]')]:
        cnts, bins = np.histogram(data, bins=100, density=True)
        plt.figure(figsize=(4,2.5))
        plt.stairs(cnts, bins, fill=True, alpha=0.6)
        plt.title(f'PDF empírica de {name} (σ={sigma})')
        plt.xlabel(name); plt.ylabel('Densidade')

    # ------------------------------------------------------------
    # 5) Decisão e SER
    x_hat = np.where(y>=0, d/2, -d/2)
    ser = np.mean(x_hat != x)
    print(f'σ={sigma}: SER = {ser:.6f}')

plt.show()



### e
# Parâmetro
d = 2

# Valores de sigma correspondentes aos itens (a)-(d)
sigma_list = [1, 0.5, 0.25, 0.125]

# Supondo que ser_list já foi calculada anteriormente no script:
# ser_list = [SER para sigma=1, sigma=0.5, sigma=0.25, sigma=0.125]
ser_list = [0.158667 , 0.023012 , 0.000029 , 0]

# Cálculo de Pe teórica para cada sigma
pe_list = [0.5 * math.erfc((d/2) / (sigma * math.sqrt(2))) for sigma in sigma_list]

# Exibe comparação
print("σ     |   SER simulada   |   Pe teórica")
print("----------------------------------------")
for sigma, ser, pe in zip(sigma_list, ser_list, pe_list):
    print(f"{sigma:<5} | {ser:>12.6e} | {pe:>12.6e}")


 ### f 
# Dados
sigma_list = [1, 0.5, 0.25, 0.125]
ser_sim     = np.array([0.16,  0.023,   0.0,    0.0])
pe_teo      = np.array([0.16,  0.023,   0.0,    0.0])
d, Eb       = 2, (2**2)/8
EbN0_dB     = [10*np.log10(Eb/(2*s**2)) for s in sigma_list]

fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)

# Plota
ax.plot(EbN0_dB, ser_sim, marker='v', color='r', linestyle='-',  label='SER simulada')
ax.plot(EbN0_dB, pe_teo,  marker='^', color='b', linestyle='--', label='Pe teórica (Q)')

# ESCALA LINEAR NO Y
ax.set_yscale('linear')

# ticks X e Y fixos e uniformes
ax.set_xticks(EbN0_dB)
ax.set_yticks(np.arange(0, 0.18, 0.02))   # 0.00,0.02,0.04,...,0.16

# formatação ponto flutuante, 2 casas
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# rótulos, título, grade e legenda
ax.set_xlabel(r'$E_b/N_0$ (dB)', fontsize=10, labelpad=6)
ax.set_ylabel('Taxa de erro',    fontsize=10, labelpad=6)
ax.set_title(r'SER e $P_e$ vs $E_b/N_0$', fontsize=12, pad=10)
ax.grid(True, linestyle=':', linewidth=0.5)
ax.legend(fontsize=9, loc='upper right')

plt.show()


