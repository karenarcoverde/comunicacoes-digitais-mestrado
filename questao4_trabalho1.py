import numpy as np
import matplotlib.pyplot as plt
import math

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



# Parâmetro
d = 2

# Valores de sigma correspondentes aos itens (a)-(d)
sigma_list = [1, 0.5, 0.25, 0.125]

# Supondo que ser_list já foi calculada anteriormente no script:
# ser_list = [SER para sigma=1, sigma=0.5, sigma=0.25, sigma=0.125]
# Caso não tenha, pode-se recalcular simulando novamente ou usando valores já obtidos.

# Para efeito de demonstração, vamos usar ser_list de um exemplo hipotético:
# (Substitua pelos seus valores simulados)
ser_list = [0.158809, 0.022617, 3.5e-05, 0]

# Cálculo de Pe teórica para cada sigma
pe_list = [0.5 * math.erfc((d/2) / (sigma * math.sqrt(2))) for sigma in sigma_list]

# Exibe comparação
print("σ     |   SER simulada   |   Pe teórica")
print("----------------------------------------")
for sigma, ser, pe in zip(sigma_list, ser_list, pe_list):
    print(f"{sigma:<5} | {ser:>12.6e} | {pe:>12.6e}")