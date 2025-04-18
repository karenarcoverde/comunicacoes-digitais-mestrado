import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Funções definidas
# ===============================

def rect_pulse(t, t0, Tb):
    """
    Gera um pulso retangular Π((t - t0)/Tb)
    Ativa (valor 1) quando |(t - t0)/Tb| <= 0.5, senão é 0
    """
    return np.where(np.abs((t - t0) / Tb) <= 0.5, 1, 0)

def tri_pulse(t, t0, Tb):
    """
    Gera um pulso triangular Λ((t - t0)/Tb)
    Valor decresce linearmente de 1 até 0 quando |(t - t0)/Tb| <= 1
    Fora desse intervalo é 0
    """
    x = (t - t0) / Tb
    return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)

# ===============================
# Eixo do tempo
# ===============================

# Cria um vetor de tempo de -5 a 5 com 1000 amostras
t = np.linspace(-5, 5, 1000)

# ===============================
# Parâmetros dos casos (a) e (b)
# ===============================

params = [(1, 0.5), (2, 3)]  # Lista com os pares (Tb, t0)
labels = ['(a) Tb=1, t0=0.5', '(b) Tb=2, t0=3']  # Rótulos para os gráficos

# ===============================
# Plotagem dos gráficos
# ===============================

# Cria uma grade de 2 linhas e 2 colunas de subgráficos
fig, axes = plt.subplots(2, 2, figsize=(12, 6))

# Para cada conjunto de parâmetros (caso a e b)
for i, (Tb, t0) in enumerate(params):
    rect = rect_pulse(t, t0, Tb)  # Calcula pulso retangular
    tri = tri_pulse(t, t0, Tb)    # Calcula pulso triangular

    # Gráfico do pulso retangular
    axes[i][0].plot(t, rect)
    axes[i][0].set_title(f'Pulso Retangular {labels[i]}')
    axes[i][0].set_xlabel('t')
    axes[i][0].set_ylabel('Amplitude')
    axes[i][0].grid(True)

    # Gráfico do pulso triangular
    axes[i][1].plot(t, tri)
    axes[i][1].set_title(f'Pulso Triangular {labels[i]}')
    axes[i][1].set_xlabel('t')
    axes[i][1].set_ylabel('Amplitude')
    axes[i][1].grid(True)

# Ajusta o layout para evitar sobreposição
plt.tight_layout()

# Exibe os gráficos na tela
plt.show()
