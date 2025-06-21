import numpy as np
import matplotlib.pyplot as plt
from commpy.modulation import QAMModem

# Parâmetros OFDM
N, CP, S = 64, 16, 10            # Número de subportadoras, comprimento do prefixo cíclico, símbolos OFDM
SNRs_dB  = np.arange(-10, 21, 5) # Faixa de SNR em dB
total    = N * S                 # Total de símbolos transmitidos

# 1) Simulação OFDM com QPSK
def simulate_ofdm_qpsk():
    k = 2  # bits por símbolo QPSK
    # Passo 2: gera bits aleatórios (2 linhas × total colunas)
    bits = np.random.randint(0, 2, size=(k, total))
    
    # Passo 3: mapeamento QPSK: s = (1 - 2*d1) + j*(1 - 2*d2)
    sym = (1 - 2*bits[0]) + 1j*(1 - 2*bits[1])
    # Aqui faz o S/P: converte esse vetor serial em uma matriz N×S,
    # onde cada coluna é um símbolo OFDM paralelo de N sub-portadoras.
    sym = sym.reshape((N, S), order='F')  # reorganiza em matriz N×S
    
    # Passo 4: IFFT para domínio do tempo
    tx = np.fft.ifft(sym, axis=0)
    
    # Passo 5: adiciona prefixo cíclico (últimas CP linhas) e serializa
    tx_cp = np.vstack([tx[-CP:], tx]).reshape(-1, order='F')
    μ = np.mean(tx_cp)                                 # média complexa de tx_cp
    signal_var = np.mean(np.abs(tx_cp - μ)**2)         # var(tx_cp) = E[|x–μ|²]
    
    ber = []
    for snr_db in SNRs_dB:
        # Passo 6: canal AWGN - adiciona ruído com variância adequada
        snr = 10**(snr_db/10)
        noise_var = signal_var / snr
        noise = np.sqrt(noise_var/2)*(np.random.randn(*tx_cp.shape) + 1j*np.random.randn(*tx_cp.shape))
        rx = tx_cp + noise
        
        # Passo 7: paraleliza e remove prefixo cíclico
        rx_mat = rx.reshape((N+CP, S), order='F')[CP:,:]
        
        # Passo 8: FFT de volta para o domínio da frequência e serializa
        y = np.fft.fft(rx_mat, axis=0).reshape(-1, order='F')
        
        # Passo 9: demodulação QPSK (decisão hard no sinal real/imaginário)
        b1 = (y.real < 0).astype(int)
        b2 = (y.imag < 0).astype(int)
        
        # Passo 10: calcula BER comparando com bits transmitidos
        errs = np.sum(bits != np.vstack([b1, b2]))
        ber.append(errs / bits.size)

        # — bloco de espectro símbolo 0 —
        freq_axis = np.linspace(-0.5, 0.5, N)
        
        t0  = tx[:, 0]
        r0  = rx_mat[:, 0]
        T0  = np.fft.fft(t0, N)
        R0  = np.fft.fft(r0, N)

        plt.figure(figsize=(6,6))
        plt.subplot(2,1,1)
        plt.plot(freq_axis, np.abs(np.fft.fftshift(T0)))
        plt.title('Espectro Antes (símbolo 0) – QPSK')
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)

        plt.subplot(2,1,2)
        plt.plot(freq_axis, np.abs(np.fft.fftshift(R0)))
        plt.title(f'Espectro Após AWGN ({snr_db} dB) – QPSK')
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'espectro_qpsk_{snr_db}dB.png')
        plt.close()
        # — fim bloco —
    return ber

# 2) Simulação genérica M-QAM usando CommPy QAMModem
def simulate_ofdm_qam(M):
    modem = QAMModem(M)                 # inicializa modem QAM
    k     = modem.num_bits_symbol       # bits por símbolo QAM
    
    # Passo 2: gera vetor de bits aleatório de comprimento k*total
    bits = np.random.randint(0, 2, size=(k, total)).reshape(-1, order='F')
     
    # Passo 3: modula bits em símbolos QAM e paraleliza
    sym = modem.modulate(bits).reshape((N, S), order='F')
    
    # Passo 4: IFFT
    tx = np.fft.ifft(sym, axis=0)
    
    # Passo 5: adiciona prefixo cíclico e serializa
    tx_cp = np.vstack([tx[-CP:], tx]).reshape(-1, order='F')
    μ = np.mean(tx_cp)                                 # média complexa de tx_cp
    signal_var = np.mean(np.abs(tx_cp - μ)**2)         # var(tx_cp) = E[|x–μ|²]
    
    ber = []
    for snr_db in SNRs_dB:
        # Passo 6: canal AWGN
        snr = 10**(snr_db/10)
        noise_var = signal_var / snr
        noise = np.sqrt(noise_var/2)*(np.random.randn(*tx_cp.shape) + 1j*np.random.randn(*tx_cp.shape))
        rx = tx_cp + noise
        
        # Passo 7: paraleliza e remove prefixo cíclico
        rx_mat = rx.reshape((N+CP, S), order='F')[CP:,:]
        #Passo 8: FFT e serializa
        y = np.fft.fft(rx_mat, axis=0).reshape(-1, order='F')
        
        # Passo 9: demodulação por decisão hard
        bits_hat = modem.demodulate(y, 'hard')
        
        # Passo 10: calcula BER comparando com vetor de bits transmitidos
        errs = np.sum(bits != bits_hat)
        ber.append(errs / bits.size)


         # — bloco de espectro símbolo 0 —
        freq_axis = np.linspace(-0.5, 0.5, tx.size)
  
        t0  = tx[:, 0]
        r0  = rx_mat[:, 0]
        T0  = np.fft.fft(t0, tx.size)
        R0  = np.fft.fft(r0, tx.size)

        plt.figure(figsize=(6,6))
        plt.subplot(2,1,1)
        plt.plot(freq_axis, np.abs(np.fft.fftshift(T0)))
        plt.title(f'Espectro Antes (símbolo 0) – {M}-QAM')
        plt.xlabel('Frequência (Hz)'); plt.ylabel('Magnitude'); plt.grid(True)

        plt.subplot(2,1,2)
        plt.plot(freq_axis, np.abs(np.fft.fftshift(R0)))
        plt.title(f'Espectro Após AWGN ({snr_db} dB) – {M}-QAM')
        plt.xlabel('Frequência (Hz)'); plt.ylabel('Magnitude'); plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'espectro_{M}qam_{snr_db}dB.png')
        plt.close()
        # — fim bloco —
    return ber

# Executa simulações para QPSK, 16-QAM e 64-QAM
BER_QPSK  = simulate_ofdm_qpsk()
BER_16QAM = simulate_ofdm_qam(16)
BER_64QAM = simulate_ofdm_qam(64)

# Plota BER vs SNR
plt.figure(figsize=(8,5))
plt.semilogy(SNRs_dB, BER_QPSK,  'o-', label='QPSK')
plt.semilogy(SNRs_dB, BER_16QAM, 's-', label='16-QAM')
plt.semilogy(SNRs_dB, BER_64QAM, '^-', label='64-QAM')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('BER vs SNR para QPSK, 16-QAM e 64-QAM')
plt.grid(which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()



