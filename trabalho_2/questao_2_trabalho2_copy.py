import numpy as np
import matplotlib.pyplot as plt
from commpy.modulation import QAMModem
from scipy.special import erfc

# ── Parâmetros gerais ─────────────────────────────────────────────────
alpha = 0.15
T = 1e-3
span = 40
fc = 100e6
Fs = 4 * fc
upsample_rate = 16
num_symbols = 1000_00
EbN0_dB_range = np.arange(0, 25, 2)

# ── Gera vetor de tempo e pulso RRC ──────────────────────────────────
# 2) Gera os coeficientes do filtro RRC (p) e do matched filter (q = p invertido)
def design_rrc(alpha, span, sps):
    N = span * sps + 1                # Numero de taps: suporte total x amostras + 1
    t = np.linspace(-span/2, span/2, N)  # Eixo do tempo de -span/2 a +span/2
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0):
            # formula do RRC no tempo zero
            h[i] = 1 + alpha*(4/np.pi - 1)
        elif np.isclose(abs(ti), 1/(4*alpha)):
            # formula em ti = +-1/(4alpha)
            h[i] = (alpha/np.sqrt(2))*(
                (1 + 2/np.pi)*np.sin(np.pi/(4*alpha)) +
                (1 - 2/np.pi)*np.cos(np.pi/(4*alpha))
            )
        else:
            # expressao geral do pulso RRC
            num = np.sin(np.pi*ti*(1-alpha)) + 4*alpha*ti*np.cos(np.pi*ti*(1+alpha))
            den = np.pi*ti*(1 - (4*alpha*ti)**2)
            h[i] = num/den
    h /= np.linalg.norm(h)             # Normaliza a energia do filtro para 1
    return h, h[::-1]                  # Retorna p e q (espelhado)

p, q   = design_rrc(alpha, span, upsample_rate)
delay  = len(p) - 1                   # Atraso total do par de filtros
N = len(p)
t = np.linspace(-span/2, span/2, N)

# ── Plota o pulso RRC ────────────────────────────────────────────────
plt.figure(figsize=(8,3))
plt.plot(t, p, linewidth=1.5)
plt.title(f'Pulso RRC (α={alpha}, T={T*1e3:.1f} ms, span={span})')
plt.xlabel('Tempo (s)'); plt.ylabel('Amplitude')
plt.grid(True); plt.tight_layout()
plt.show()

# ── Funções de up/down‐sampling ────────────────────────────────────────
def upsample(x, L):
    y = np.zeros(len(x) * L, dtype=x.dtype)
    y[::L] = x
    return y

# ── Simulação para cada b = 2,4,6 ───────────────────────────────────
b_values = [2, 4, 6]
received        = {}
tx_constellation = {}

for b in b_values:
    M = 2**b
    modem = QAMModem(M)

    ber_sim = []
    ber_th = []
    # guarda constelação pura
    tx_constellation[b] = modem.constellation.copy()

    for EbN0_dB in EbN0_dB_range:
        bits = np.random.randint(0, 2, b * num_symbols)
        print("teste2")
        print(bits)
        symbols_tx = modem.modulate(bits)
        I_seq, Q_seq = symbols_tx.real, symbols_tx.imag

        I_up = upsample(I_seq, upsample_rate)
        Q_up = upsample(Q_seq, upsample_rate)

        I_t = np.convolve(I_up, p, mode='full')
        Q_t = np.convolve(Q_up, p, mode='full')

        t_sig = np.arange(len(I_t)) / Fs
        carrier_cos = np.sqrt(2)*np.cos(2*np.pi*fc*t_sig)
        carrier_sin = np.sqrt(2)*np.sin(2*np.pi*fc*t_sig)
        x_pass = I_t*carrier_cos - Q_t*carrier_sin

        Ex = np.mean(np.abs(symbols_tx)**2)
        Eb = Ex / b
        EbN0 = 10**(EbN0_dB / 10)
        sigma = np.sqrt((Eb / EbN0) / 2)
        noise = sigma * np.random.randn(len(x_pass))
        x_noisy = x_pass + noise

        I_rx = x_noisy * carrier_cos
        Q_rx = x_noisy * (-carrier_sin)
        I_mf = np.convolve(I_rx, p[::-1], mode='full')
        Q_mf = np.convolve(Q_rx, p[::-1], mode='full')

        overall_delay = delay
        YI_k = I_mf[overall_delay::upsample_rate][:num_symbols]
        YQ_k = Q_mf[overall_delay::upsample_rate][:num_symbols]

        received[b] = (YI_k, YQ_k)
        symbols_rx = YI_k + 1j * YQ_k
 
        bits_rx = modem.demodulate(symbols_rx, 'hard')
        print("teste1")
        print(bits_rx)
        ber_sim.append(np.mean(bits_rx != bits))
        print(ber_sim)

        k = np.log2(M)
        ber_th.append((4/k)*(1 - 1/np.sqrt(M))*0.5*erfc(np.sqrt((3*k*EbN0)/(M-1))/np.sqrt(2)))

    plt.figure(figsize=(8, 5))
    plt.semilogy(EbN0_dB_range, ber_sim, 'o-', label='Simulado')
    plt.semilogy(EbN0_dB_range, ber_th, '--', label='Teórico')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title(f'BER simulada vs teórica para {M}-QAM')
    plt.grid(which='both', linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.show()

# fora do for‐principal, DEPOIS que todas as chaves 2,4,6 existem
fig, axes = plt.subplots(1, 3, figsize=(15,4))
for ax, b_plot in zip(axes, b_values):
    tx = tx_constellation[b_plot]
    YI_k, YQ_k = received[b_plot]
    ax.scatter(tx.real,  tx.imag,  s=80, ec='blue', label='Tx', zorder=2)
    ax.scatter(YI_k, YQ_k, s=8,  color='orange', label='Rx', zorder=1)
    ax.set_title(f'{2**b_plot}-QAM')
    ax.axis('equal'); ax.grid(True); ax.legend()
plt.tight_layout(); plt.show()

