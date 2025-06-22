import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from commpy.modulation import QAMModem

# Parâmetros OFDM
N, CP, S = 64, 16, 10000            # Número de subportadoras, comprimento do prefixo cíclico, símbolos OFDM
SNRs_dB  = np.arange(-10, 21, 5) # Faixa de SNR em dB
total    = N * S                 # Total de símbolos transmitidos
L = 10                           # número de taps do canal Rayleigh
# --- Canal Rayleigh puro ---
# h[n] = (hR + j hI)/sqrt(2), tamanho L
h = (np.random.randn(L) + 1j*np.random.randn(L)) / np.sqrt(2*L)

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
    
    ber_awgn = []
    ber_ray  = []
    ber_ray_awgn = []

    for snr_db in SNRs_dB:
        # Passo 6: canal AWGN - adiciona ruído com variância adequada
        snr = 10**(snr_db/10)
        noise_var = signal_var / snr
        # --- AWGN puro ---
        noise = np.sqrt(noise_var/2)*(np.random.randn(*tx_cp.shape) + 1j*np.random.randn(*tx_cp.shape))
        rx_awgn = tx_cp + noise

        
        # convolução tx_cp * h
        rx_ray = np.convolve(tx_cp, h, mode='full')[:tx_cp.size]

        # --- Rayleigh + AWGN ---
        noise2 = np.sqrt(noise_var/2)*(np.random.randn(*rx_ray.shape)
                                      +1j*np.random.randn(*rx_ray.shape))
        rx_ray_awgn = rx_ray + noise2
        
        # função auxiliar de recebimento, FFT, demod e BER
        def ofdm_recv(s,type):
            # Passo 7: paraleliza e remove prefixo cíclico
            mat = s.reshape((N+CP, S), order='F')[CP:,:]
            # Passo 8: FFT de volta para o domínio da frequência e serializa
            # sempre faz FFT
            Y = np.fft.fft(mat, axis=0)   # N×S

            # or type == 'ray'
            if type == 'awgn':
                # AWGN puro → não tem fading, basta serializar
                y = Y.reshape(-1, order='F')
            else:
                # Rayleigh puro ou Rayleigh+AWGN → equaliza
                H_fft = np.fft.fft(h, N)
                # evita divisões por valores muito pequenos
                H_fft[np.abs(H_fft) < 1e-3] = 1e-3
                Y_eq = Y / H_fft[:,None]
                y     = Y_eq.reshape(-1, order='F')


            # Passo 9: demodulação QPSK (decisão hard no sinal real/imaginário)
            b1  = (y.real<0).astype(int)
            b2  = (y.imag<0).astype(int)

            # — bloco de espectro símbolo 0 —
            freq_axis = np.linspace(-0.5, 0.5, N)
            
            t0  = tx[:, 0]
            r0  = mat[:, 0]
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

            # Passo 10: calcula BER comparando com bits transmitidos
            errs = np.sum(bits != np.vstack([b1,b2]))
            return errs/(bits.size)
        
        ber_awgn.append(ofdm_recv(rx_awgn,'awgn'))
        ber_ray.append(ofdm_recv(rx_ray,'ray'))
        ber_ray_awgn.append(ofdm_recv(rx_ray_awgn,'ray_awgn'))


    return ber_awgn, ber_ray, ber_ray_awgn


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
    
    ber_awgn = []
    ber_ray  = []
    ber_ray_awgn = []


    for snr_db in SNRs_dB:
        # Passo 6: canal AWGN
        snr = 10**(snr_db/10)
        noise_var = signal_var / snr
        # --- AWGN puro ---
        noise = np.sqrt(noise_var/2)*(np.random.randn(*tx_cp.shape) + 1j*np.random.randn(*tx_cp.shape))
        rx_awgn = tx_cp + noise

        # convolução tx_cp * h
        rx_ray = np.convolve(tx_cp, h, mode='full')[:tx_cp.size]

        # --- Rayleigh + AWGN ---
        noise2 = np.sqrt(noise_var/2)*(np.random.randn(*rx_ray.shape)
                                      +1j*np.random.randn(*rx_ray.shape))
        rx_ray_awgn = rx_ray + noise2
        
        # função auxiliar de recebimento, FFT, demod e BER
        def ofdm_recv(s,type):
            # Passo 7: paraleliza e remove prefixo cíclico
            mat = s.reshape((N+CP, S), order='F')[CP:,:]
            # Passo 8: FFT de volta para o domínio da frequência e serializa
            # sempre faz FFT
            Y = np.fft.fft(mat, axis=0)   # N×S

            # or type == 'ray'
            if type == 'awgn':
                # AWGN puro → não tem fading, basta serializar
                y = Y.reshape(-1, order='F')
            else:
                # Rayleigh puro ou Rayleigh+AWGN → equaliza
                H_fft = np.fft.fft(h, N)
                # evita divisões por valores muito pequenos
                H_fft[np.abs(H_fft) < 1e-3] = 1e-3
                Y_eq = Y / H_fft[:,None]
                y     = Y_eq.reshape(-1, order='F')
                

            # Passo 9: demodulação por decisão hard
            bits_hat = modem.demodulate(y, 'hard')

            # — bloco de espectro símbolo 0 —
            freq_axis = np.linspace(-0.5, 0.5, N)
            
            t0  = tx[:, 0]
            r0  = mat[:, 0]
            T0  = np.fft.fft(t0, N)
            R0  = np.fft.fft(r0, N)

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

            # Passo 10: calcula BER comparando com vetor de bits transmitidos
            errs = np.sum(bits != bits_hat)
            return errs / bits.size

        ber_awgn.append(ofdm_recv(rx_awgn,'awgn'))
        ber_ray.append(ofdm_recv(rx_ray,'ray'))
        ber_ray_awgn.append(ofdm_recv(rx_ray_awgn,'ray_awgn'))


    return ber_awgn, ber_ray, ber_ray_awgn

# Executa simulações para QPSK, 16-QAM e 64-QAM
BER_AWGN_QPSK, BER_RAY_QPSK, BER_RAY_AWGN_QPSK = simulate_ofdm_qpsk()
BER_AWGN_16QAM, BER_RAY_16QAM, BER_RAY_AWGN_16QAM = simulate_ofdm_qam(16)
BER_AWGN_64QAM, BER_RAY_64QAM, BER_RAY_AWGN_64QAM = simulate_ofdm_qam(64)

# 1) Canal AWGN
plt.figure(figsize=(8,5))
plt.semilogy(SNRs_dB, BER_AWGN_QPSK,  'o-', label='QPSK')
plt.semilogy(SNRs_dB, BER_AWGN_16QAM, 's-', label='16-QAM')
plt.semilogy(SNRs_dB, BER_AWGN_64QAM, '^-', label='64-QAM')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('BER vs SNR — Canal AWGN')
# fixa os ticks em potências de 10
yticks = [1, 1e-1, 1e-2, 1e-3, 1e-4,1e-5,1e-6]
ylabels = [r'$10^0$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$',r'$10^{-6}$']
plt.yticks(yticks, ylabels)
plt.grid(which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('ber_awgn.png')
plt.close()

# 2) Canal Rayleigh puro
plt.figure(figsize=(8,5))
plt.semilogy(SNRs_dB, BER_RAY_QPSK,  'o-', label='QPSK')
plt.semilogy(SNRs_dB, BER_RAY_16QAM, 's-', label='16-QAM')
plt.semilogy(SNRs_dB, BER_RAY_64QAM, '^-', label='64-QAM')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('BER vs SNR — Canal Rayleigh')
# fixa os ticks em potências de 10
yticks = [1, 1e-1, 1e-2, 1e-3]
ylabels = [r'$10^0$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$']
plt.yticks(yticks, ylabels)
plt.grid(which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('ber_ray.png')
plt.close()


# 2) Canal Rayleigh + AWGN
plt.figure(figsize=(8,5))
plt.semilogy(SNRs_dB, BER_RAY_AWGN_QPSK,  'o-', label='QPSK')
plt.semilogy(SNRs_dB, BER_RAY_AWGN_16QAM, 's-', label='16-QAM')
plt.semilogy(SNRs_dB, BER_RAY_AWGN_64QAM, '^-', label='64-QAM')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('BER vs SNR — Canal Rayleigh + AWGN')
# fixa os ticks em potências de 10
yticks = [1, 1e-1, 1e-2, 1e-3]
ylabels = [r'$10^0$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$']
plt.yticks(yticks, ylabels)
plt.grid(which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('ber_ray_awgn.png')
plt.close()




CP_list = [8, 16, 32]
mods = [
    (simulate_ofdm_qpsk, 'QPSK'),
    (lambda: simulate_ofdm_qam(16), '16-QAM'),
    (lambda: simulate_ofdm_qam(64), '64-QAM')
]
canals = [
    ('AWGN',    lambda awgn, ray, rawn: awgn),
    ('Rayleigh',lambda awgn, ray, rawn: ray),
    ('Rayleigh+AWGN', lambda awgn, ray, rawn: rawn)
]

for mod_func, mod_label in mods:
    for canal_key, select_ber in canals:
        plt.figure(figsize=(8,5))
        for CP in CP_list:
            globals()['CP'] = CP
            ber_awgn, ber_ray, ber_rawn = mod_func()
            ber = select_ber(ber_awgn, ber_ray, ber_rawn)
            plt.semilogy(
                SNRs_dB, ber, '-o',
                label=f'CP={CP}'
            )

        plt.title(f'{mod_label} — {canal_key}')
        plt.xlabel('SNR (dB)')
        plt.ylabel('BER')
        yticks = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        ylabels = [r'$10^0$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$']
        plt.yticks(yticks, ylabels)
        plt.grid(which='both', ls='--', alpha=0.6)
        plt.legend(title='Prefixo Cíclico', loc='upper right')
        plt.tight_layout()

        # em vez de plt.show(), salve o arquivo:
        filename = f'BER_{mod_label}_{canal_key}.png'.replace('+','p').replace(' ','_')
        # plt.savefig(filename, dpi=300)
        # plt.close()
        # print(f'Salvo: {filename}')
        plt.show()
        plt.close()

