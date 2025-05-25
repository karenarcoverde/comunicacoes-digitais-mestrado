import numpy as np
import matplotlib
matplotlib.use('Agg')          # não abre janelas
import matplotlib.pyplot as plt
from scipy.special import erfc

# ── Parâmetros gerais ─────────────────────────────────────────────────────────
b_vals    = [1, 2, 4]                # bits por símbolo
EbN0_dB   = np.arange(0, 25, 4)      # Eb/N0: 0,4,8,…,24 dB
num_bits  = 200_000                  # bits por ponto (ajuste p/ tempo)

# ── 1) Mapeamento Gray → símbolos M-PAM ────────────────────────────────────────
def bits_to_symbols(bits, b):
    syms = bits.reshape(-1, b)
    ints = syms.dot(1 << np.arange(b)[::-1])
    gray = ints ^ (ints >> 1)
    M    = 2**b
    return 2*gray - (M - 1)

# ── 2) Inversão Gray → bits ───────────────────────────────────────────────────
def symbols_to_bits(a_hat, b):
    # pega os índices Gray de volta
    M       = 2**b
    gray    = (a_hat + (M - 1)) // 2
    # converte Gray → binário natural
    binary  = gray.copy()
    shift   = gray >> 1
    while np.any(shift):
        binary ^= shift
        shift >>= 1
    # agora extrai bits
    bits_hat = ((binary[:,None] >> np.arange(b)[::-1]) & 1).flatten()
    return bits_hat

# ── 3) Constellation plots (AWGN ideal, sem up/downsampling) ────────────────
for EbN0 in EbN0_dB:
    fig, axes = plt.subplots(1, len(b_vals), figsize=(12,3))
    for ax, b in zip(axes, b_vals):
        M    = 2**b
        # gera alguns bits para visualizar
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
        ax.set_xlabel('Tx'); ax.set_ylabel('Rx'); ax.grid(True)
    fig.tight_layout()
    fig.savefig(f'constellation_ideal_{EbN0}dB.png', dpi=150)
    plt.close(fig)

# ── 4) AWGN ideal + decisão + BER ─────────────────────────────────────────────
def awgn_ber_ideal(bits, EbN0_dB, b):
    # mapeia bits→símbolos
    a  = bits_to_symbols(bits, b)
    Eb = np.mean(a**2) / b
    N0 = Eb / (10**(EbN0_dB/10))
    sigma = np.sqrt(N0/2)
    # adiciona ruído
    y     = a + sigma * np.random.randn(len(a))
    # decisão por constelação
    M      = 2**b
    consts = np.arange(-M+1, M, 2)
    idxs   = np.argmin(np.abs(y[:,None] - consts[None,:]), axis=1)
    a_hat  = consts[idxs]
    # converte símbolos decididos→bits
    bits_hat = symbols_to_bits(a_hat, b)
    return np.mean(bits_hat != bits)

# ── 5) BER teórica aprox. (Gray-mapped M-PAM) ─────────────────────────────────
def ber_theoretical(EbN0_dB, M, b):
    gamma = 10**(EbN0_dB/10)
    arg   = np.sqrt(6*b/(M**2 - 1) * gamma)
    Q     = 0.5 * erfc(arg/np.sqrt(2))
    # aproximadamente BER ≈ SER/b para Gray mapping
    return (2*(M-1)/M * Q) / b

# ── 6) Loop BER e plot (AWGN ideal) ────────────────────────────────────────────
for b in b_vals:
    M      = 2**b
    BER_sim = []
    BER_th  = []
    print(f"\nSimulação {M}-PAM - BER")
    for EbN0 in EbN0_dB:
        bits = np.random.randint(0,2, num_bits)
        ber  = awgn_ber_ideal(bits, EbN0, b)
        BER_sim.append(ber)
        BER_th .append(ber_theoretical(EbN0, M, b))
        print(f"{EbN0:2d} dB → BER Simulado: {ber:.2e}, Teórico: {BER_th[-1]:.2e}")

    # plota
    plt.figure(figsize=(6,4))
    plt.semilogy(EbN0_dB, BER_sim, 'o-', label='BER Simulado')
    plt.semilogy(EbN0_dB, BER_th,  '--^', label='BER Teórico')
    plt.title(f'BER vs Eb/N₀ ({M}-PAM)')
    plt.xlabel('Eb/N₀ (dB)')
    plt.ylabel('BER')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'ber_{M}PAM_ideal.png', dpi=150)
    plt.close()
