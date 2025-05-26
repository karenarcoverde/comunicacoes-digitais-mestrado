import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ---  Funções auxiliares de teoria de probabilidade de erro ------------------

def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))

def ber_qam_theoretical(EbN0, M, b):
    return (4 * (np.sqrt(M) - 1) / (np.sqrt(M) * b)) * \
           Q(np.sqrt(3 * b * EbN0 / (M - 1)))

# ---  Parâmetros gerais do sistema -------------------------------------------

b_list       = [2, 4, 6]               # b = bits/símbolo ➔ 4-,16-,64-QAM
EbN0_dB_list = [0, 4, 8, 12, 16, 20, 24]
Nsym_const   = 2000                    # símbolos para constelações
Nbits_ber    = 20000                   # bits totais para BER

# Parâmetros de pulse-shaping
T      = 1e-3      # símbolo de duração 1 ms
dt     = T/100     # resolução
L      = 8         # upsampling
p_len  = int(T/dt) # amostras de p(t)
p      = np.ones(p_len)    # pulso retangular
h      = p[::-1]           # matched filter p(T-t)

# --- 1) Constelações Tx vs Rx c/ upsampling e filtragem ---------------------

idx = np.random.choice(Nsym_const, 500, replace=False)
sample_pts = np.arange(Nsym_const)*L + (p_len - 1)  # pontos de amostragem após p

for EbN0_dB in EbN0_dB_list:
    EbN0 = 10**(EbN0_dB/10)
    fig = plt.figure(figsize=(12,4))

    for i, b in enumerate(b_list):
        M = 2**b
        m = int(np.sqrt(M))

        # Mapeamento QAM
        bits = np.random.randint(0,2,(Nsym_const,b))
        di   = bits[:,:b//2].dot(2**np.arange(b//2)[::-1])
        dq   = bits[:,b//2:].dot(2**np.arange(b//2)[::-1])
        aI   = 2*di - (m-1)
        aQ   = 2*dq - (m-1)

        # Upsampling
        sI = np.zeros(Nsym_const*L)
        sQ = np.zeros(Nsym_const*L)
        sI[::L] = aI
        sQ[::L] = aQ

        # Pulse-shaping (convolução com p(t))
        xI = np.convolve(sI, p, mode='full')
        xQ = np.convolve(sQ, p, mode='full')

        # AWGN no canal baseband
        Ex    = np.mean(aI**2 + aQ**2)
        Eb_sym= Ex / b
        N0    = Eb_sym / EbN0
        noiseI = np.sqrt(N0/2)*np.random.randn(len(xI))
        noiseQ = np.sqrt(N0/2)*np.random.randn(len(xQ))
        rI = xI + noiseI
        rQ = xQ + noiseQ

        # Matched filter (convolução com h(t))
        yI = np.convolve(rI, h, mode='full')
        yQ = np.convolve(rQ, h, mode='full')

        # Amostragem em t = iL + (p_len-1)
        r_sym = yI[sample_pts] + 1j*yQ[sample_pts]
        tx_sym= aI + 1j*aQ

        # Plot
        ax = fig.add_subplot(1, len(b_list), i+1)
        ax.scatter(r_sym.real[idx], r_sym.imag[idx],
                   s=8, c='orange', marker='x', label='Rx', zorder=1)
        ax.scatter(tx_sym.real[idx], tx_sym.imag[idx],
                   s=8, c='blue', marker='o', label='Tx', zorder=2)
        ax.set_title(f'{M}-QAM\nEb/N0={EbN0_dB} dB')
        ax.set_xlabel('I'); ax.set_ylabel('Q')
        ax.grid(True); ax.legend(fontsize='small')

    fig.tight_layout()
    fig.savefig(f'constellation_EbN0_{EbN0_dB}dB.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)

# --- 2) Simulação BER vs teórico (mantida) ----------------------------------

ber_sim  = {b: [] for b in b_list}
ber_theo = {b: [] for b in b_list}

for b in b_list:
    M    = 2**b; m = int(np.sqrt(M))
    Nsym = Nbits_ber // b

    # Geração baseband
    bits_tx = np.random.randint(0,2,(Nsym,b))
    di       = bits_tx[:,:b//2].dot(2**np.arange(b//2)[::-1])
    dq       = bits_tx[:,b//2:].dot(2**np.arange(b//2)[::-1])
    aI       = 2*di - (m-1)
    aQ       = 2*dq - (m-1)

    # Upsampling e pulse-shaping
    sI = np.zeros(Nsym*L); sQ = np.zeros_like(sI)
    sI[::L] = aI; sQ[::L] = aQ
    xI = np.convolve(sI, p, mode='full'); xQ = np.convolve(sQ, p, mode='full')

    Ex    = np.mean(aI**2 + aQ**2)
    Eb_sym= Ex / b

    for EbN0_dB in EbN0_dB_list:
        EbN0 = 10**(EbN0_dB/10)
        N0    = Eb_sym / EbN0
        # AWGN
        noiseI = np.sqrt(N0/2)*np.random.randn(len(xI))
        noiseQ = np.sqrt(N0/2)*np.random.randn(len(xQ))
        rI = xI + noiseI; rQ = xQ + noiseQ

        # Matched filter
        yI = np.convolve(rI, h, mode='full')
        yQ = np.convolve(rQ, h, mode='full')
        sample_pts = np.arange(Nsym)*L + (p_len-1)

        # Amostragem e decisão
        r_sym = yI[sample_pts] + 1j*yQ[sample_pts]
        levels = 2*np.arange(m) - (m-1)
        di_hat = levels[np.argmin(np.abs(r_sym.real[:,None]-levels), axis=1)]
        dq_hat = levels[np.argmin(np.abs(r_sym.imag[:,None]-levels), axis=1)]

        # Reconstrução de bits
        bits_rx = np.zeros_like(bits_tx)
        for k in range(Nsym):
            di_i = int((di_hat[k] + (m-1)) / 2)
            dq_i = int((dq_hat[k] + (m-1)) / 2)
            bits_rx[k,:b//2] = ((di_i>>np.arange(b//2)[::-1]) & 1)
            bits_rx[k,b//2:] = ((dq_i>>np.arange(b//2)[::-1]) & 1)

        ber_sim[b].append(np.mean(bits_tx != bits_rx))
        ber_theo[b].append(ber_qam_theoretical(EbN0, M, b))

# Plot e salvar BER
for b in b_list:
    M = 2**b
    fig, ax = plt.subplots(figsize=(6,4))
    ax.semilogy(EbN0_dB_list, ber_sim[b], 'o-', label=f'{M}-QAM Simulado')
    ax.semilogy(EbN0_dB_list, ber_theo[b], '--', label=f'{M}-QAM Teórico')
    ax.set_xlabel('Eb/N0 (dB)'); ax.set_ylabel('BER')
    ax.set_title(f'BER vs Eb/N0 para {M}-QAM')
    ax.grid(True, which='both'); ax.legend()
    fig.tight_layout()
    fig.savefig(f'ber_{M}QAM.png', dpi=150, bbox_inches='tight')
    plt.close(fig)