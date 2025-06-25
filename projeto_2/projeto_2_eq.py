import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from numpy.fft import fft, ifft

# === Parâmetros do sistema OFDM ===
N = 64
CP = 16
S = int(1e4)
M = 64
b = int(np.log2(M))
SNR_dB = np.arange(-10, 25, 5)
r = 3
L = 1

# === Escolha do tipo de canal ===
canal_rayleigh = False  # True para Rayleigh + AWGN, False para apenas AWGN

# Canal Rayleigh (se ativado)
if canal_rayleigh:
    h = (np.random.randn(L) + 1j*np.random.randn(L)) / np.sqrt(2 * L)
else:
    h = np.array([1.0])  # canal ideal (AWGN)

# === Geração de bits ===
d = np.random.randint(0, 2, b * N * S)
num_bits = len(d)
d_rep = np.repeat(d, r)

BER_sem_cod = []
BER_com_cod = []

# === Modulação QAM ===
def qam_mod(bits, M):
    b = int(np.log2(M))
    symbols = np.array([int("".join(str(bi) for bi in bits[i:i + b]), 2)
                        for i in range(0, len(bits), b)])
    sqrt_M = int(np.sqrt(M))
    I = 2 * (symbols % sqrt_M) - sqrt_M + 1
    Q = 2 * (symbols // sqrt_M) - sqrt_M + 1
    qam = I + 1j * Q
    norm_factor = np.sqrt(np.mean(np.abs(qam) ** 2))
    qam = qam / norm_factor
    return qam, norm_factor

# === Demodulação QAM ===
def qam_demod(symbols, M, norm_factor):
    sqrt_M = int(np.sqrt(M))
    symbols = symbols * norm_factor
    I = np.round((np.real(symbols) + sqrt_M - 1) / 2).astype(int)
    Q = np.round((np.imag(symbols) + sqrt_M - 1) / 2).astype(int)
    I = np.clip(I, 0, sqrt_M - 1)
    Q = np.clip(Q, 0, sqrt_M - 1)
    symbols_idx = Q * sqrt_M + I
    b = int(np.log2(M))
    bits = np.array([list(np.binary_repr(int(i), b)) for i in symbols_idx], dtype=int).flatten()
    return bits

# === Loop sobre SNR ===
for snr_db in SNR_dB:
    snr_linear = 10 ** (snr_db / 10)

    # === SEM Correção ===
    symbols, norm_factor = qam_mod(d, M)
    symbols_matrix = symbols.reshape(N, -1)
    ofdm_time = ifft(symbols_matrix, axis=0)
    ofdm_cp = np.vstack([ofdm_time[-CP:], ofdm_time])
    tx_signal = ofdm_cp.flatten()

    noise_var = np.var(tx_signal) / snr_linear
    noise = np.sqrt(noise_var / 2) * (np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal)))

    if canal_rayleigh:
        rx_signal = convolve(tx_signal, h, mode='full')[:len(tx_signal)] + noise
    else:
        rx_signal = tx_signal + noise

    rx_matrix = rx_signal.reshape(N + CP, -1)
    rx_no_cp = rx_matrix[CP:, :]
    rx_freq = fft(rx_no_cp, axis=0)

    if canal_rayleigh:
        h_pad = np.zeros(N, dtype=complex)
        h_pad[:L] = h
        H = fft(h_pad, N)
        H[np.abs(H) < 1e-3] = 1e-3
        rx_eq = rx_freq / H[:, None]
    else:
        rx_eq = rx_freq

    rx_symbols = rx_eq.flatten()
    d_hat = qam_demod(rx_symbols, M, norm_factor)
    BER_sem_cod.append(np.sum(d[:len(d_hat)] != d_hat[:len(d)]) / num_bits)

    # === COM Correção ===
    symbols_rep, norm_factor_rep = qam_mod(d_rep, M)
    symbols_matrix = symbols_rep.reshape(N, -1)
    ofdm_time = ifft(symbols_matrix, axis=0)
    ofdm_cp = np.vstack([ofdm_time[-CP:], ofdm_time])
    tx_signal = ofdm_cp.flatten()

    noise_var = np.var(tx_signal) / snr_linear
    noise = np.sqrt(noise_var / 2) * (np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal)))

    if canal_rayleigh:
        rx_signal = convolve(tx_signal, h, mode='full')[:len(tx_signal)] + noise
    else:
        rx_signal = tx_signal + noise

    rx_matrix = rx_signal.reshape(N + CP, -1)
    rx_no_cp = rx_matrix[CP:, :]
    rx_freq = fft(rx_no_cp, axis=0)

    if canal_rayleigh:
        h_pad = np.zeros(N, dtype=complex)
        h_pad[:L] = h
        H = fft(h_pad, N)
        H[np.abs(H) < 1e-3] = 1e-3
        rx_eq = rx_freq / H[:, None]
    else:
        rx_eq = rx_freq

    rx_symbols = rx_eq.flatten()
    d_hat_rep = qam_demod(rx_symbols, M, norm_factor_rep)

    valid_length = (len(d_hat_rep) // r) * r
    d_hat_rep = d_hat_rep[:valid_length]
    d_hat_reshape = d_hat_rep.reshape(-1, r)
    d_decod = (np.sum(d_hat_reshape, axis=1) > (r / 2)).astype(int)

    BER_com_cod.append(np.sum(d != d_decod[:len(d)]) / num_bits)

# === Plotagem ===
canal_nome = "Rayleigh + AWGN" if canal_rayleigh else "AWGN"
plt.figure()
plt.semilogy(SNR_dB, BER_sem_cod, 'bo-', label='Sem Correção', linewidth=2)
plt.semilogy(SNR_dB, BER_com_cod, 'rs-', label=f'Com Código de Repetição (r = {r})', linewidth=2)
plt.grid(True, which='both')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title(f'BER vs SNR - {M}-QAM e Canal {canal_nome}')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()