import numpy as np
import matplotlib
matplotlib.use('Agg')       # usa Agg em vez de TkAgg
import matplotlib.pyplot as plt
from scipy.special import erfc

# 1) Parâmetros de simulação
b_vals    = [1, 2, 4]                  # bits por símbolo
alpha     = 0.15                       # roll‑off do filtro RRC
sps       = 8                          # amostras por símbolo
span      = 40                          # extensão do filtro em símbolos
EbN0_dB   = np.arange(0, 25, 4)        # Eb/N0: 0, 4, 8, …, 24 dB
num_bits  = 1000_000                      # bits para estimação de BER

# 2) Design do filtro Raiz‑Cosseno Levantado (RRC)
def design_rrc(alpha, span, sps):
    N = span * sps + 1
    t = np.linspace(-span/2, span/2, N)
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0):
            h[i] = 1 + alpha*(4/np.pi - 1)
        elif np.isclose(abs(ti), 1/(4*alpha)):
            h[i] = (alpha/np.sqrt(2))*(
                (1 + 2/np.pi)*np.sin(np.pi/(4*alpha)) +
                (1 - 2/np.pi)*np.cos(np.pi/(4*alpha))
            )
        else:
            num = np.sin(np.pi*ti*(1-alpha)) + 4*alpha*ti*np.cos(np.pi*ti*(1+alpha))
            den = np.pi*ti*(1 - (4*alpha*ti)**2)
            h[i] = num/den
    h /= np.linalg.norm(h)
    return h, h[::-1]

p, q   = design_rrc(alpha, span, sps)
delay  = len(p) - 1

# 3) Mapeamento Gray para M‑PAM
def bits_to_symbols(bits, b):
    syms = bits.reshape(-1, b)
    ints = syms.dot(1 << np.arange(b)[::-1])
    gray = ints ^ (ints >> 1)
    M    = 2**b
    return 2*gray - (M - 1)

# 4) Transmissão: upsampling + filtro p(t)
def transmit(a, p, sps):
    ups = np.zeros(len(a)*sps)
    ups[::sps] = a
    return np.convolve(ups, p, mode='full')

# 5) Canal AWGN
def awgn(x, Eb, EbN0_dB):
    N0    = Eb / (10**(EbN0_dB/10))
    sigma = np.sqrt(N0/2)
    return x + sigma*np.random.randn(len(x))

# 6) Recepção: filtro casado q(t) + amostragem
def receive(y, q, sps, delay):
    z = np.convolve(y, q, mode='full')
    return z[delay::sps]

# 7) Detecção e demapeamento inverso Gray→binário
def detect(y_samp, M, b):
    consts   = np.arange(-M+1, M, 2)
    idxs     = np.argmin(np.abs(y_samp[:,None] - consts[None,:]), axis=1)
    decisions= consts[idxs]
    gray_hat = (decisions + (M-1))//2
    binary   = gray_hat.copy()
    shift    = gray_hat >> 1
    while np.any(shift):
        binary ^= shift
        shift >>= 1
    return ((binary[:,None] >> np.arange(b)[::-1]) & 1).flatten()

# 8) BER teórico
def ber_theoretical(EbN0_dB, M, b):
    arg = np.sqrt(3*b/(M**2-1) * 10**(EbN0_dB/10))
    return (2*(M-1)/(M*b)) * 0.5 * erfc(arg)

# 9a) Constellations (uma figura por Eb/N0)
for EbN0 in EbN0_dB:
    fig, axes = plt.subplots(1, len(b_vals), figsize=(12, 3))
    for ax, b in zip(axes, b_vals):
        M    = 2**b
        bits = np.random.randint(0,2,num_bits)
        a    = bits_to_symbols(bits, b)
        x    = transmit(a, p, sps)
        Eb   = np.mean(a**2)/b
        y    = awgn(x, Eb,EbN0)
        y_s  = receive(y, q, sps, delay)[:1000]
        ax.scatter(a[:1000], y_s, s=2)
        ax.set_title(f'{M}-PAM @ Eb/N0={EbN0} dB')
        ax.set_xlabel('Tx'); ax.set_ylabel('Rx'); ax.grid(True)
    fig.tight_layout()
    fig.savefig(f'constellation_{EbN0}dB.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

# 9b) Curvas de BER vs Eb/N0 separadas por M‑PAM
for b in b_vals:
    M = 2**b
    ber_sim  = []
    ber_theo = []
    for EbN0 in EbN0_dB:
        bits = np.random.randint(0,2,num_bits)
        a    = bits_to_symbols(bits, b)
        x    = transmit(a, p, sps)
        Eb   = np.mean(a**2)/b
        y    = awgn(x, Eb, EbN0)
        y_s  = receive(y, q, sps, delay)[:len(a)]
        ber_sim.append(np.mean(bits != detect(y_s, M, b)))
        ber_theo.append(ber_theoretical(EbN0, M, b))
    fig, ax = plt.subplots(figsize=(6,4))
    ax.semilogy(EbN0_dB, ber_sim,  'o-', label='Simulado')
    ax.semilogy(EbN0_dB, ber_theo, '--', label='Teórico')
    ax.set_title(f'BER vs Eb/N0 ({M}-PAM)')
    ax.set_xlabel('Eb/N0 (dB)')
    ax.set_ylabel('BER')
    ax.grid(True, which='both')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'ber_{M}PAM.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
