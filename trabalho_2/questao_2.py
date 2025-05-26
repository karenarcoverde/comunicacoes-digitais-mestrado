import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import pandas as pd

# ---  Funções auxiliares de teoria de probabilidade de erro ------------------

def Q(x):
    """Função Q, necessária para o cálculo teórico de BER."""
    return 0.5 * erfc(x / np.sqrt(2))

def ber_qam_theoretical(EbN0, M, b):
    """
    BER teórico aproximado para constelação quadrada M-QAM em AWGN:
      BER ≈ [4(√M−1)/(√M·b)] · Q(√(3b·Eb/N0/(M−1)))
    """
    return (4 * (np.sqrt(M) - 1) / (np.sqrt(M) * b)) * \
           Q(np.sqrt(3 * b * EbN0 / (M - 1)))


# ---  Parâmetros gerais do sistema -------------------------------------------

b_list       = [2, 4, 6]               # b = bits/símbolo ➔ 4-, 16-, 64-QAM
EbN0_dB_list = [0, 4, 8, 12, 16, 20, 24]  # Eb/N0 em dB
Nsym_const   = 2000                    # símbolos para as constelações
Nbits_ber    = 20000                   # bits totais para simulação de BER


# -----------------------------------------------------------------------------
# 1) Transmissor de banda-base + RF (M-QAM):
#    - Geração de símbolos
#    - Mapemento I/Q
#    - Upsampling e pulse-shaping
#    - Up-conversion (banda-passante)
# -----------------------------------------------------------------------------

# Índices aleatórios para plotar apenas parte da constelação
idx = np.random.choice(Nsym_const, 500, replace=False)

for EbN0_dB in EbN0_dB_list:
    EbN0 = 10**(EbN0_dB / 10)         # converter dB ➔ razão linear
    fig = plt.figure(figsize=(12, 4))

    for i, b in enumerate(b_list):
        M = 2**b
        m = int(np.sqrt(M))

        # --- 1.1) Geração de bits e mapeamento para M-QAM --------------------
        bits = np.random.randint(0, 2, size=(Nsym_const, b))
        # parte In-Phase (I)
        di = bits[:, :b//2].dot(2**np.arange(b//2)[::-1])
        # parte Quadrature (Q)
        dq = bits[:, b//2:].dot(2**np.arange(b//2)[::-1])
        # amplitudes simétricas {±1, ±3, ...}
        aI = 2*di - (m - 1)
        aQ = 2*dq - (m - 1)
        # sinal complexo base-band
        sym = aI + 1j*aQ

        # --- 1.2) Cálculo da energia média por bit ----------------------------
        Ex    = np.mean(np.abs(sym)**2)  # energia média do símbolo
        Eb_sym= Ex / b                   # energia por bit

        # --- 1.3) Canal AWGN aplicado em I/Q (simples para BER) -------------
        # variância σ² = N0/2 com N0 = Eb/N0_linear
        N0    = Eb_sym / EbN0
        noise = np.sqrt(N0/2) * \
                (np.random.randn(Nsym_const) + 1j*np.random.randn(Nsym_const))
        # sinal recebido base-band
        r = sym + noise

        # --- 1.4) Plot da constelação Tx vs Rx (banda-base) -----------------
        ax = fig.add_subplot(1, len(b_list), i+1)
        # Rx primeiro (por baixo)  
        ax.scatter(r.real[idx], r.imag[idx],
                   s=8, c='orange', marker='x', label='Rx', zorder=1)
        # Tx depois (por cima)
        ax.scatter(sym.real[idx], sym.imag[idx],
                   s=8, c='blue', marker='o', label='Tx', zorder=2)
        ax.set_title(f'{M}-QAM\nEb/N0={EbN0_dB} dB')
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.grid(True)
        ax.legend(fontsize='small')

    fig.tight_layout()
    # salva cada figura sem exibir
    fig.savefig(f'constellation_EbN0_{EbN0_dB}dB.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)


# -----------------------------------------------------------------------------
# 2) Demodulação e cálculo de BER
#    - Decisão por vizinho mais próximo em I e Q
#    - Comparação simulada vs. teórica
# -----------------------------------------------------------------------------

# Prepara dicionários para armazenar BER
ber_sim  = {b: [] for b in b_list}
ber_theo = {b: [] for b in b_list}

for b in b_list:
    M    = 2**b
    m    = int(np.sqrt(M))
    Nsym = Nbits_ber // b

    # 2.1) Geração de quadro de símbolos (base-band)
    bits_tx = np.random.randint(0, 2, size=(Nsym, b))
    di       = bits_tx[:, :b//2].dot(2**np.arange(b//2)[::-1])
    dq       = bits_tx[:, b//2:].dot(2**np.arange(b//2)[::-1])
    sym_bb   = (2*di - (m - 1)) + 1j*(2*dq - (m - 1))
    Ex       = np.mean(np.abs(sym_bb)**2)
    Eb_sym   = Ex / b

    for EbN0_dB in EbN0_dB_list:
        EbN0 = 10**(EbN0_dB / 10)
        N0    = Eb_sym / EbN0

        # 2.2) Canal AWGN
        noise = np.sqrt(N0/2) * \
                (np.random.randn(Nsym) + 1j*np.random.randn(Nsym))
        r     = sym_bb + noise

        # 2.3) Decisão (detector de vizinho mais próximo)
        levels = 2*np.arange(m) - (m - 1)
        di_hat = levels[np.argmin(np.abs(r.real[:, None] - levels), axis=1)]
        dq_hat = levels[np.argmin(np.abs(r.imag[:, None] - levels), axis=1)]

        # 2.4) Reconstrução de bits e cálculo de BER
        bits_rx = np.zeros_like(bits_tx)
        for k in range(Nsym):
            di_i = int((di_hat[k] + (m - 1)) / 2)
            dq_i = int((dq_hat[k] + (m - 1)) / 2)
            # converte índices de volta para bits
            bits_rx[k, :b//2] = ((di_i >> np.arange(b//2)[::-1]) & 1)
            bits_rx[k, b//2:] = ((dq_i >> np.arange(b//2)[::-1]) & 1)

        ber_sim[b].append(np.mean(bits_tx != bits_rx))
        ber_theo[b].append(ber_qam_theoretical(EbN0, M, b))


# --- 3) Plot das curvas BER vs Eb/N0 e salvar --------------------------------

for b in b_list:
    M = 2**b
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(EbN0_dB_list, ber_sim[b], 'o-', label=f'{M}-QAM Simulado')
    ax.semilogy(EbN0_dB_list, ber_theo[b], '--', label=f'{M}-QAM Teórico')
    ax.set_xlabel('Eb/N0 (dB)')
    ax.set_ylabel('BER')
    ax.set_title(f'BER vs Eb/N0 para {M}-QAM')
    ax.grid(True, which='both')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'ber_{M}QAM.png', dpi=150, bbox_inches='tight')
    plt.close(fig) 
