import numpy as np
import matplotlib.pyplot as plt

# Parâmetros gerais
d = 2                   # distância entre níveis de símbolo (±d/2)
N = 10**6               # número de símbolos transmitidos
symbols = np.array([-d/2, d/2])

# Valores de σ para os itens (b), (c) e (d)
sigma_list = [1, 0.5, 0.25, 0.125]

for sigma in sigma_list:
    # 1. Sequência de transmissão x[n]
    x = np.random.choice(symbols, size=N)
    # 2. Ruído AWGN v[n] ~ N(0, σ^2)
    v = np.random.normal(0, sigma, size=N)
    # 3. Sinal recebido y[n] = x[n] + v[n]
    y = x + v

    # 4. Plot das primeiras 100 amostras
    plt.figure()
    plt.plot(x[:100])
    plt.title(f'Primeiras 100 amostras de x[n] (σ={sigma})')
    plt.xlabel('n')
    plt.ylabel('x[n]')

    plt.figure()
    plt.plot(v[:100])
    plt.title(f'Primeiras 100 amostras de v[n] (σ={sigma})')
    plt.xlabel('n')
    plt.ylabel('v[n]')

    plt.figure()
    plt.plot(y[:100])
    plt.title(f'Primeiras 100 amostras de y[n] (σ={sigma})')
    plt.xlabel('n')
    plt.ylabel('y[n]')

    # 5. Histogramas de x, v, y
    plt.figure()
    plt.hist(x, bins=2, density=True, alpha=0.7)
    plt.title(f'Histograma de x[n] (σ={sigma})')
    plt.xlabel('Valor')
    plt.ylabel('Prob. empírica')

    plt.figure()
    plt.hist(v, bins=100, density=True, alpha=0.7)
    plt.title(f'Histograma de v[n] (σ={sigma})')
    plt.xlabel('Valor')
    plt.ylabel('Prob. empírica')

    plt.figure()
    plt.hist(y, bins=100, density=True, alpha=0.7)
    plt.title(f'Histograma de y[n] (σ={sigma})')
    plt.xlabel('Valor')
    plt.ylabel('Prob. empírica')

    # 6. Decisão x̂[n] com limiar em zero
    x_hat = np.where(y >= 0,  d/2, -d/2)
    # 7. Taxa de erro de símbolo (SER)
    ser_list = [np.mean(x_hat != x)]
    print(ser_list)

# Exibe todos os gráficos
plt.show()