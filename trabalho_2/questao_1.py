import numpy as np
import matplotlib
matplotlib.use('Agg')       # Usa backend 'Agg' para plotagem sem janelas
import matplotlib.pyplot as plt
from scipy.special import erfc

# 1) Parametros de simulacao
b_vals    = [1, 2, 4]                  # Bits por simbolo (1->2-PAM, 2->4-PAM, 4->16-PAM)
alpha     = 0.15                       # Roll-off do filtro RRC
sps       = 8                          # Amostras por simbolo
span      = 40                         # Suporte do filtro em simbolos (+-40 simbolos)
EbN0_dB   = np.arange(0, 25, 4)        # Eb/N0 em dB: [0, 4, 8, ..., 24]
num_bits  = 1_000_000                  # Total de bits para estimar o BER

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

p, q   = design_rrc(alpha, span, sps)
delay  = len(p) - 1                   # Atraso total do par de filtros

# 3) Mapeia bits -> simbolos Gray-codificados para M-PAM
def bits_to_symbols(bits, b):
    syms = bits.reshape(-1, b)                         # Agrupa bits de b em b
    ints = syms.dot(1 << np.arange(b)[::-1])           # Converte grupo em inteiro
    gray = ints ^ (ints >> 1)                          # Converte para codigo Gray
    M    = 2**b
    return 2*gray - (M - 1)                            # Mapeia para {-M+1, ..., +M-1}

# 4) Transmissao: upsampling + convolucao com p(t)
def transmit(a, p, sps):
    ups = np.zeros(len(a)*sps)     # Vetor de zeros numa taxa maior
    ups[::sps] = a                 # Insere cada simbolo a em um slot
    return np.convolve(ups, p, mode='full')  # Pulse-shaping

# 5) Canal AWGN: adiciona ruido gaussiano
def awgn(x, Eb, EbN0_dB):
    N0    = Eb / (10**(EbN0_dB/10))         # Densidade espectral de ruido
    sigma = np.sqrt(N0/2)                  # Desvio-padrao por amostra
    return x + sigma*np.random.randn(len(x))

# 6) Recepcao: matched filter q(t) + downsampling
def receive(y, q, sps, delay):
    z = np.convolve(y, q, mode='full')     # Filtra com matched filter
    return z[delay::sps]                   # Compensa atraso e amostra por simbolo

# 7) Decisao e demapeamento Gray->bits
def detect(y_samp, M, b):
    consts   = np.arange(-M+1, M, 2)       # Constelacao teorica
    idxs     = np.argmin(np.abs(y_samp[:,None] - consts[None,:]), axis=1)
    decisions= consts[idxs]                # Simbolos detectados
    gray_hat = (decisions + (M-1))//2      # Converte de volta pra Gray code
    binary   = gray_hat.copy()
    shift    = gray_hat >> 1
    # Reverte Gray->binario
    while np.any(shift):
        binary ^= shift
        shift >>= 1
    return ((binary[:,None] >> np.arange(b)[::-1]) & 1).flatten()

# 8) BER teorico para Gray-M-PAM
def ber_theoretical(EbN0_dB, M, b):
    arg = np.sqrt(3*b/(M**2-1) * 10**(EbN0_dB/10))
    return (2*(M-1)/(M*b)) * 0.5 * erfc(arg)

# 9a) Gera e salva diagramas de constelacao para cada Eb/N0
for EbN0 in EbN0_dB:
    fig, axes = plt.subplots(1, len(b_vals), figsize=(12, 3))
    for ax, b in zip(axes, b_vals):
        M    = 2**b
        bits = np.random.randint(0, 2, num_bits)
        a    = bits_to_symbols(bits, b)
        x    = transmit(a, p, sps)
        Eb   = np.mean(a**2)/b
        y    = awgn(x, Eb, EbN0)
        y_s  = receive(y, q, sps, delay)[:1000]  # so plotar 1000 pontos
        ax.scatter(a[:1000], y_s, s=2)
        ax.set_title(f'{M}-PAM @ Eb/N0={EbN0} dB')
        ax.set_xlabel('Tx'); ax.set_ylabel('Rx'); ax.grid(True)
    fig.tight_layout()
    fig.savefig(f'constellation_{EbN0}dB.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

# 9b) Gera e salva curvas de BER vs Eb/N0 para cada M-PAM
for b in b_vals:
    M = 2**b
    ber_sim  = []
    ber_theo = []
    for EbN0 in EbN0_dB:
        bits = np.random.randint(0, 2, num_bits)
        a    = bits_to_symbols(bits, b)
        x    = transmit(a, p, sps)
        Eb   = np.mean(a**2)/b
        y    = awgn(x, Eb, EbN0)
        y_s  = receive(y, q, sps, delay)[:len(a)]
        ber_sim.append(np.mean(bits != detect(y_s, M, b)))
        ber_theo.append(ber_theoretical(EbN0, M, b))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(EbN0_dB, ber_sim,  'o-', label='Simulado')
    ax.semilogy(EbN0_dB, ber_theo, '--', label='Teorico')
    ax.set_title(f'BER vs Eb/N0 ({M}-PAM)')
    ax.set_xlabel('Eb/N0 (dB)')
    ax.set_ylabel('BER')
    ax.grid(True, which='both')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'ber_{M}PAM.png', dpi=150, bbox_inches='tight')
    plt.close(fig)