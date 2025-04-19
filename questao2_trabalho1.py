import numpy as np
import matplotlib.pyplot as plt

# Número de amostras: mais pontos aumentam a resolução em tempo e frequência
N = 8192

# vetor de tempo de -40 a 40 segundos
t = np.linspace(-40, 40, N)
# passo de tempo (incremento entre amostras)
dt = t[1] - t[0]

# (a) Geração do pulso triangular p(t) = Λ(t)
# - Amplitude máxima 1 em t=0
# - Decai linearmente até 0 em |t| = 1
p = np.where(np.abs(t) <= 1, 1 - np.abs(t), 0)

# (b) Transformada de Fourier P(f) via FFT
# - fft(p) computa a DFT discreta
# - fftshift centraliza o zero da frequência
# - multiplicar por dt aproxima a integral contínua
P = np.fft.fftshift(np.fft.fft(p)) * dt

# eixo de frequência correspondente (Hz)
# - fftfreq gera os bins de frequência
# - fftshift realinha para que f=0 fique no centro
f = np.fft.fftshift(np.fft.fftfreq(N, d=dt))

# (c) Transformada inversa para reconstruir o sinal no tempo
# - ifftshift e ifft revertendo o fftshift e fft
# - divisão por dt aproxima a transformada inversa contínua
p_inv = np.fft.ifft(np.fft.ifftshift(P)) / dt

# (d1) Plot do espectro |P(f)| em torno da frequência zero
plt.figure(figsize=(10, 4))
plt.plot(f, np.abs(P))
plt.title('Magnitude of Fourier Transform |P(f)| vs Frequency f')
plt.xlabel('Frequency f (Hz)')
plt.ylabel('|P(f)|')
plt.xlim(-10, 10)   # foco na região do lóbulo principal
plt.grid(True)

# (d2) Plot do sinal reconstruído pinv(t) no intervalo de tempo completo
plt.figure(figsize=(10, 4))
plt.plot(t, np.real(p_inv))
plt.title('Inverse Fourier Transform pinv(t) vs Time t')
plt.xlabel('Time t')
plt.ylabel('pinv(t)')
plt.xlim(-40, 40)   # interval completo de -40 a +40
plt.grid(True)

plt.show()
