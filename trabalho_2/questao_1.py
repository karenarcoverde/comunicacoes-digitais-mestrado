import numpy as np
import matplotlib
matplotlib.use('Agg')  # não abre janelas
import matplotlib.pyplot as plt
from scipy.special import erfc

# ── Parâmetros gerais ─────────────────────────────────────────────────────────
b_vals    = [1, 2, 4]                # bits por símbolo
EbN0_dB   = np.arange(0, 25, 4)      # Eb/N0: 0,4,8,…,24 dB
num_sym   = 200_000                  # símbolos por ponto (ajuste p/ tempo)

# ── 1) Gray mapping → símbolos M-PAM ───────────────────────────────────────────
def bits_to_symbols(bits, b):
    syms = bits.reshape(-1, b)
    ints = syms.dot(1 << np.arange(b)[::-1])
    gray = ints ^ (ints >> 1)
    M    = 2**b
    return 2*gray - (M - 1)

# ── 2) Constellation plots (AWGN ideal, sem up/downsampling) ─────────────────
for EbN0 in EbN0_dB:
    fig, axes = plt.subplots(1, len(b_vals), figsize=(12,3))
    for ax, b in zip(axes, b_vals):
        M    = 2**b
        # gera um número pequeno de símbolos para visualizar a nuvem
        bits = np.random.randint(0,2, 1000 * b)
        a    = bits_to_symbols(bits, b)
        # adiciona AWGN ideal
        Eb   = np.mean(a**2)/b
        N0   = Eb/10**(EbN0/10)
        sigma= np.sqrt(N0/2)
        y    = a + sigma * np.random.randn(len(a))
        # scatter Tx vs Rx
        ax.scatter(a, y, s=4, alpha=0.6)
        ax.set_title(f'{M}-PAM @ Eb/N₀={EbN0} dB')
        ax.set_xlabel('Tx')
        ax.set_ylabel('Rx')
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(f'constellation_ideal_{EbN0}dB.png', dpi=150)
    plt.close(fig)

# ── 3) AWGN ideal + decisão (sem up/downsampling) ─────────────────────────────
def awgn_ser_ideal(a, EbN0_dB, b):
    Eb = np.mean(a**2) / b
    N0 = Eb / (10**(EbN0_dB/10))
    sigma = np.sqrt(N0/2)
    y     = a + sigma * np.random.randn(len(a))
    M      = 2**b
    consts = np.arange(-M+1, M, 2)
    idxs   = np.argmin(np.abs(y[:,None] - consts[None,:]), axis=1)
    a_hat  = consts[idxs]
    return np.mean(a_hat != a)

# ── 4) SER teórica Gray-mapped M-PAM ──────────────────────────────────────────
def ser_theoretical(EbN0_dB, M, b):
    gamma = 10**(EbN0_dB/10)
    arg   = np.sqrt(6*b/(M**2 - 1) * gamma)
    Q     = 0.5 * erfc(arg/np.sqrt(2))
    return 2*(M-1)/M * Q

# ── 5) Loop SER e plot (AWGN ideal) ────────────────────────────────────────────
for b in b_vals:
    M       = 2**b
    SER_sim = []
    SER_th  = []
    print(f"\nSimulação {M}-PAM (AWGN ideal)")
    for EbN0 in EbN0_dB:
        bits = np.random.randint(0,2, num_sym * b)
        a    = bits_to_symbols(bits, b)
        ser  = awgn_ser_ideal(a, EbN0, b)
        SER_sim.append(ser)
        SER_th .append(ser_theoretical(EbN0, M, b))
        print(f"{EbN0:2d} dB → Sim: {ser:.2e}, Teo: {SER_th[-1]:.2e}")
    # plota
    plt.figure(figsize=(6,4))
    plt.semilogy(EbN0_dB, SER_sim, 'o-', label='SER Simulado')
    plt.semilogy(EbN0_dB, SER_th,  '--^', label='SER Teórico')
    plt.title(f'SER vs Eb/N₀ ({M}-PAM) - AWGN ideal')
    plt.xlabel('Eb/N₀ (dB)')
    plt.ylabel('SER')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'ser_{M}PAM_ideal.png', dpi=150)
    plt.close(fig)
