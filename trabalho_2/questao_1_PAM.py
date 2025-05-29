import numpy as np
import matplotlib
matplotlib.use('Agg')       # backend sem janelas
import matplotlib.pyplot as plt
from scipy.special import erfc

# ── Parâmetros gerais ─────────────────────────────────────────────────
b_values       = [1, 2, 4]                 # Bits por símbolo → 2-PAM, 4-PAM, 16-PAM
num_symbols    = 1000_00                  # símbolos por sequência
EbN0_dB_range  = np.arange(0, 25, 4)       # Eb/N0 em dB: [0,4,8,…,24]
alpha, sps, span = 0.15, 16, 40             # RRC: roll‐off, samples/símbolo, span em símbolos

# ── 1) Gera coeficientes RRC (p) e matched filter (q) ────────────────
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

p, q   = design_rrc(alpha, span, sps)
delay  = len(p) - 1    

# ── 2) Map bits → símbolos Gray‐PAM ────────────────────────────────────
def bits_to_symbols(bits, b):
    syms = bits.reshape(-1, b)
    ints = syms.dot(1 << np.arange(b)[::-1])
    gray = ints ^ (ints >> 1)
    M    = 2**b
    return 2*gray - (M - 1)

# ── 3) Pulse‐shaping (banda‐base) ─────────────────────────────────────
def transmit(a, p, sps):
    ups = np.zeros(len(a)*sps)
    ups[::sps] = a
    return np.convolve(ups, p, mode='full')

# ── 4) Canal AWGN ─────────────────────────────────────────────────────
def awgn(x, Eb, EbN0_dB):
    N0    = Eb / (10**(EbN0_dB/10))
    sigma = np.sqrt(N0/2)
    return x + sigma*np.random.randn(len(x))

# ── 5) Recepção (matched filter + downsample) ─────────────────────────
def receive(y, q, sps, delay):
    z = np.convolve(y, q, mode='full')
    return z[delay::sps]

# ── 6) Decisão symbol‐by‐symbol ────────────────────────────────────────
def detect(y_s, M):
    consts = np.arange(-M+1, M, 2)
    idxs   = np.argmin(np.abs(y_s[:,None] - consts[None,:]), axis=1)
    return consts[idxs]

# ── 7) BER teórico para Gray‐PAM ───────────────────────────────────────
def ber_theoretical(EbN0_dB, M, b):
    k   = np.log2(M)
    EbN0 = 10**(EbN0_dB/10)
    # argumento corretamente normalizado para erfc
    arg = np.sqrt(((6*k)/(M**2 - 1)) * EbN0)
    # SER = 2*(M-1)/M * Q(arg)
    Pe = (2*(M-1)/M) * 0.5 * erfc(arg/np.sqrt(2))
    # BER = SER / k
    return Pe / k

# ── prepara constelações ideais ───────────────────────────────────────
tx_constellation = {
    b: np.arange(-2**b+1, 2**b, 2)
    for b in b_values
}
rx_constellation = {
    Eb: {} for Eb in EbN0_dB_range
}

# ── 8) Simulação e plot de BER vs Eb/N0 ───────────────────────────────
for b in b_values:
    M = 2**b
    ber_sim, ber_th = [], []
    for EbN0_dB in EbN0_dB_range:
        bits = np.random.randint(0, 2, b * num_symbols)
        a    = bits_to_symbols(bits, b)
        x    = transmit(a, p, sps)
        Eb   = np.mean(a**2)/b
        y    = awgn(x, Eb, EbN0_dB)
        y_s  = receive(y, q, sps, delay)[:len(a)]
        decisions = detect(y_s, M)
        # 1. Converte símbolo → código Gray inteiro
        gray_rx = (decisions + (M-1)) // 2  

        # 2. Reverte Gray → binário inteiro
        bin_rx = gray_rx.copy()
        shift = gray_rx >> 1
        while np.any(shift):
            bin_rx ^= shift
            shift >>= 1

        # 3. Expande inteiro → bits Rx
        #   bin_rx.shape = (num_symbols,)
        # queremos bits_rx.shape = (num_symbols * b,)
        bits_rx = ((bin_rx[:,None] >> np.arange(b)[::-1]) & 1).flatten()

        # 4. Calcula BER
        ber_sim.append(np.mean(bits_rx != bits))
        ber_th.append(ber_theoretical(EbN0_dB, M, b))
    fig, ax = plt.subplots(figsize=(6,4))
    ax.semilogy(EbN0_dB_range, ber_sim, 'o-', label='Simulado')
    ax.semilogy(EbN0_dB_range, ber_th,  '--', label='Teórico')
    ax.set_xlabel('Eb/N0 (dB)'); ax.set_ylabel('BER')
    ax.set_title(f'BER vs Eb/N0 — {M}-PAM')
    ax.grid(True, which='both', linestyle=':')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'ber_{M}PAM.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

# ── 9) Constelações base‐band lado a lado ─────────────────────────────
for EbN0_dB in EbN0_dB_range:
    fig, axes = plt.subplots(1, len(b_values), figsize=(15,4))
    for ax, b in zip(axes, b_values):
        M = 2**b
        # gera nova sequência para visualizar dispersão
        bits = np.random.randint(0, 2, b * num_symbols)
        a    = bits_to_symbols(bits, b)
        x    = transmit(a, p, sps)
        Eb   = np.mean(a**2)/b
        y    = awgn(x, Eb, EbN0_dB)
        y_s  = receive(y, q, sps, delay)[:len(a)]
        rx_constellation[EbN0_dB][b] = y_s  # opcional se quiser guardar

        # plot ideal Tx (vazado azul) e Rx (laranja cheio)
        ax.scatter( tx_constellation[b],
                    np.zeros_like(tx_constellation[b]),
                    s=5, edgecolors='blue',
                    label='Tx', zorder=2)
        ax.scatter( y_s,
                    np.zeros_like(y_s),
                    s=5, color='orange', alpha=0.7,
                    label='Rx', zorder=1)

        # coloca legenda nesse subplot
        ax.legend(loc='upper center', fontsize='small', ncol=2, frameon=False)
        ax.set_yticks([])                   # oculta eixo Y
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('data', 0))
        ax.set_xlabel('Em fase')
        ax.grid(False)
    plt.suptitle(f'Constelações M-PAM @ Eb/N0={EbN0_dB} dB', y=1.05)
    fig.tight_layout(rect=[0,0,1,0.9])
    fig.savefig(f'constellation_PAM_{EbN0_dB}dB.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
