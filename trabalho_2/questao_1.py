import numpy as np
import matplotlib
matplotlib.use('Agg')  # não abre janelas
import matplotlib.pyplot as plt
from scipy.special import erfc

# ── 1) Parâmetros ──────────────────────────────────────────────────────────────
b_vals    = [1, 2, 4]                  # bits por símbolo
M_vals    = [2**b for b in b_vals]     # ordem da PAM
EbN0_dB   = np.arange(0, 25, 4)        # Eb/N0: 0,4,8,…,24 dB
num_sym   = 1000000                    # símbolos por ponto de SER
sps       = 8                          # oversampling
alpha     = 0.15                       # roll-off RRC
span      = 6                          # extensão do filtro em símbolos

# ── 2) Design do filtro Raiz-Cosseno Levantado (RRC) ───────────────────────────
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

# ── 3) Gray mapping → símbolos M-PAM ──────────────────────────────────────────
def bits_to_symbols(bits, b):
    syms = bits.reshape(-1, b)
    ints = syms.dot(1 << np.arange(b)[::-1])
    gray = ints ^ (ints >> 1)
    M    = 2**b
    return 2*gray - (M - 1)

# ── 4) Transmissão: upsampling + filtro p(t) ─────────────────────────────────
def transmit(a, p, sps):
    ups = np.zeros(len(a)*sps)
    ups[::sps] = a
    return np.convolve(ups, p, mode='full')

# ── 5) Canal AWGN ─────────────────────────────────────────────────────────────
#    (após matched-filter e amostragem, noise var = N0/2)
def awgn(x, Eb, EbN0_dB):
    N0    = Eb / (10**(EbN0_dB/10))
    sigma = np.sqrt(N0/2)
    return x + sigma * np.random.randn(len(x))

# ── 6) Recepção: filtro casado + amostragem ───────────────────────────────────
def receive(y, q, sps, delay):
    z = np.convolve(y, q, mode='full')
    return z[delay::sps]

# ── 7) Decisão por constelação (símbolos PAM) ─────────────────────────────────
def detect_symbols(y_samp, M):
    consts = np.arange(-M+1, M, 2)
    idxs   = np.argmin(np.abs(y_samp[:,None] - consts[None,:]), axis=1)
    return consts[idxs]

# ── 8) SER teórica Gray-mapped M-PAM ──────────────────────────────────────────
def ser_theoretical(EbN0_dB, M):
    # converte dB → razão linear
    gamma = 10**(EbN0_dB/10)
    # monta o argumento do Q:
    arg = np.sqrt(6*np.log2(M)/(M**2 - 1) * gamma)
    # calcula Q(arg) via erfc
    Q = 0.5 * erfc(arg/np.sqrt(2))
    # aplica o fator 2*(M-1)/M
    return 2*(M-1)/M * Q

# ── 9a) Constellations para cada Eb/N0 ───────────────────────────────────────
for EbN0 in EbN0_dB:
    fig, axes = plt.subplots(1, len(b_vals), figsize=(12,3))
    for ax, b in zip(axes, b_vals):
        M    = 2**b
        bits = np.random.randint(0,2, num_sym*b)
        a    = bits_to_symbols(bits, b)
        x    = transmit(a, p, sps)
        Eb   = np.mean(a**2)/b
        y    = awgn(x, Eb, EbN0)
        y_s  = receive(y, q, sps, delay)[:num_sym]
        a_hat= detect_symbols(y_s, M)
        ax.scatter(a[:1000], y_s[:1000], s=3, alpha=0.6)
        ax.set_title(f'{M}-PAM @ Eb/N₀={EbN0} dB')
        ax.set_xlabel('Tx'); ax.set_ylabel('Rx'); ax.grid(True)
    fig.tight_layout()
    fig.savefig(f'constellation_{EbN0}dB.png', dpi=150)
    plt.close(fig)

# ── 9b) Curvas de SER vs Eb/N0 ────────────────────────────────────────────────
for b in b_vals:
    M       = 2**b
    SER_sim = []
    SER_th  = []

    # calcula SER
    for EbN0 in EbN0_dB:
        bits = np.random.randint(0,2, num_sym*b)
        a    = bits_to_symbols(bits, b)
        x    = transmit(a, p, sps)
        Eb   = np.mean(a**2)/b
        y    = awgn(x, Eb, EbN0)
        y_s  = receive(y, q, sps, delay)[:num_sym]
        a_hat= detect_symbols(y_s, M)
        SER_sim.append( np.mean(a_hat != a) )
        SER_th .append( ser_theoretical(EbN0, M) )

    # **Imprime os valores teóricos e simulados lado a lado**
    print(f'\n{M}-PAM SER @ Eb/N0:')
    print(' Eb/N0 (dB) |  Simulado  |  Teórico')
    for EbN0, sim, theo in zip(EbN0_dB, SER_sim, SER_th):
        print(f'   {EbN0:>2d}      |  {sim:.3e}  |  {theo:.3e}')

    # plota SER
    fig, ax = plt.subplots(figsize=(6,4))
    ax.semilogy(EbN0_dB, SER_sim, 'o-',  label='SER Simulado')
    ax.semilogy(EbN0_dB, SER_th,  '--^', label='SER Teórico')
    ax.set_title(f'SER vs Eb/N₀ ({M}-PAM)')
    ax.set_xlabel('Eb/N₀ (dB)')
    ax.set_ylabel('SER')
    ax.grid(True, which='both')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'ser_{M}PAM.png', dpi=150)
    plt.close(fig)