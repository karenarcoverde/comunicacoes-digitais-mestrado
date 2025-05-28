import numpy as np
import matplotlib.pyplot as plt
from commpy.modulation import QAMModem
from scipy.special import erfc

# ── Parâmetros gerais ─────────────────────────────────────────────────
alpha        = 0.15        # roll-off do RRC
T            = 1e-3        # símbolo de 1 ms
span         = 40          # RRC abrange ±span símbolos
fc           = 4e3         # portadora “digital” de 4 kHz
Fs           = 4 * fc      # frequência de amostragem = 16 kHz
upsample_rate     = 16          # amostras por símbolo
num_symbols  = 1000        # quantos símbolos gerar
EbN0_dB_plot = 24          # Eb/N0 (dB) para a constelação que vamos visualizar

# ── Gera vetor de tempo e pulso RRC ──────────────────────────────────
num_taps = int(span * T * Fs)
t        = np.linspace(-span*T/2, span*T/2, num_taps, endpoint=False)

def raised_cosine(t, T, alpha):
    x   = t/T
    num = np.sinc(x) * np.cos(np.pi * alpha * x)
    den = 1 - (2 * alpha * x)**2
    rc  = num/den
    sing = np.isclose(den, 0)
    rc[sing] = np.pi/4 * np.sinc(1/(2*alpha))
    return rc

# pulso RRC normalizado em energia
p_rc  = raised_cosine(t, T, alpha)
p_rc /= np.sqrt(np.sum(p_rc**2))
delay = (len(p_rc) - 1)//2

# ── Plota o pulso RRC ────────────────────────────────────────────────
plt.figure(figsize=(8,3))
plt.plot(t, p_rc, linewidth=1.5)
plt.title(f'Pulso RRC (α={alpha}, T={T*1e3:.1f} ms, span={span})')
plt.xlabel('Tempo (s)'); plt.ylabel('Amplitude')
plt.grid(True); plt.tight_layout()
plt.show()


# ── Funções de up/down‐sampling ────────────────────────────────────────
def upsample(x, L):
    """
    Upsampling puro: insere L–1 zeros entre cada amostra de x.
    """
    y = np.zeros(len(x) * L, dtype=x.dtype)
    y[::L] = x
    return y

def downsample(x, M):
    """
    Downsampling puro (decimação): pega uma amostra a cada M, sem filtragem.
    """
    return x[::M]

# ── Simulação para cada b = 2,4,6 ───────────────────────────────────
from commpy.modulation import QAMModem

b_values        = [2, 4, 6]
received        = {}
tx_constellation = {}

for b in b_values:
    M     = 2**b
    modem = QAMModem(M)

    # 1) Gera bits
    bits   = np.random.randint(0, 2, b * num_symbols)
    # 2) Modula VIA CommPy
    s      = modem.modulate(bits)
    I_seq  = s.real
    Q_seq  = s.imag

    # guarda constelação pura
    tx_constellation[b] = modem.constellation.copy()
    print(tx_constellation)

    # 2) Upsampling puro
    I_up = upsample(I_seq, upsample_rate)
    Q_up = upsample(Q_seq, upsample_rate)

    # 4) Pulse shaping (time-domain)
    I_t = np.convolve(I_up, p_rc, mode='full')
    Q_t = np.convolve(Q_up, p_rc, mode='full')

    # 5) Monta portadora “digital”
    t_sig       = np.arange(len(I_t)) / Fs
    carrier_cos = np.sqrt(2)*np.cos(2*np.pi*fc*t_sig)
    carrier_sin = np.sqrt(2)*np.sin(2*np.pi*fc*t_sig)
    x_pass      = I_t*carrier_cos - Q_t*carrier_sin

    # 6) Energia por bit
    Ex  = np.mean(np.abs(s)**2)
    Eb  = Ex / b

    # 7) Só para o Eb/N0 de interesse, adiciona ruído e demodula
    EbN0  = 10**(EbN0_dB_plot/10)
    sigma = np.sqrt((Eb/EbN0)/2)
    noise = sigma * np.random.randn(len(x_pass))
    x_noisy = x_pass + noise

    I_rx = x_noisy * carrier_cos
    Q_rx = x_noisy * (-carrier_sin)
    I_mf = np.convolve(I_rx, p_rc[::-1], mode='full')
    Q_mf = np.convolve(Q_rx, p_rc[::-1], mode='full')

    # 8) Downsampling puro em t = kT  → gera YI_k, YQ_k
    overall_delay = 2 * delay
    YI_k = I_mf[overall_delay::upsample_rate][:num_symbols]
    YQ_k = Q_mf[overall_delay::upsample_rate][:num_symbols]

    received[b] = (YI_k, YQ_k)
    

# ── Plota as constelações Tx vs Rx ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15,5))
for ax, b in zip(axes, b_values):
    tx = tx_constellation[b]
    tx_I = tx.real
    tx_Q = tx.imag
    YI_k, YQ_k = received[b]

    ax.scatter(tx_I, tx_Q, s=80,
               edgecolors='blue',
               label='Tx', zorder=1)
    ax.scatter(YI_k, YQ_k, s=10, color='orange',
               label='Rx', zorder=2)

    ax.set_title(f'{2**b}-QAM — Eb/N0 = {EbN0_dB_plot} dB')
    ax.set_xlabel('I'); ax.set_ylabel('Q')
    ax.grid(True); ax.axis('equal')
    ax.legend()

plt.tight_layout()
plt.show()
