import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from commpy.modulation import QAMModem

# ------------------ Configuração de Parâmetros ------------------
# Parâmetros OFDM
N, CP, S = 64, 16, 10000            # Número de subportadoras, comprimento do prefixo cíclico, símbolos OFDM
SNRs_dB  = np.arange(-10, 21, 5) # Faixa de SNR em dB
total    = N * S                 # Total de símbolos transmitidos
L =20                           # número de taps do canal Rayleigh
# --- Canal Rayleigh puro ---
# h[n] = (hR + j hI)/sqrt(2), tamanho L
h = (np.random.randn(L) + 1j*np.random.randn(L)) / np.sqrt(2*L)

# 1) Simulação OFDM com QPSK
# ------------------ 1) Simulação OFDM com QPSK ------------------
def simulate_ofdm_qpsk():
    k = 2  # bits por símbolo QPSK
    # Passo 2: gera bits aleatórios (2 linhas × total colunas)
    bits = np.random.randint(0, 2, size=(k, total))
    
    # Passo 3: mapeamento QPSK: s = (1 - 2*d1) + j*(1 - 2*d2)
    sym = (1 - 2*bits[0]) + 1j*(1 - 2*bits[1])
    # Aqui faz o S/P: converte esse vetor serial em uma matriz N×S,
    # onde cada coluna é um símbolo OFDM paralelo de N sub-portadoras.
    # Serial-to-parallel: reorganiza em matriz N subportadoras x S símbolos
    sym = sym.reshape((N, S), order='F')  # reorganiza em matriz N×S
    
    # Passo 4: IFFT para domínio do tempo
    tx = np.fft.ifft(sym, axis=0)
    
    # Passo 5: adiciona prefixo cíclico (últimas CP linhas) e serializa
    tx_cp = np.vstack([tx[-CP:], tx]).reshape(-1, order='F')
    # Estatísticas do sinal transmitido
    μ = np.mean(tx_cp)                                 # média complexa de tx_cp
    signal_var = np.mean(np.abs(tx_cp - μ)**2)         # var(tx_cp) = E[|x–μ|²]
    
    ber_awgn = [] # Lista para BER no canal AWGN
    ber_ray  = [] # Lista para BER no canal Rayleigh puro
    ber_ray_awgn = [] # Lista para BER no canal Rayleigh + AWGN

    for snr_db in SNRs_dB:
        # Passo 6: canal AWGN - adiciona ruído com variância adequada
        # Converte SNR de dB para razão linear e calcula variância de ruído
        snr = 10**(snr_db/10)
        noise_var = signal_var / snr
        # --- AWGN puro ---
        # Canal AWGN puro: adiciona ruído gaussiano complexo
        noise = np.sqrt(noise_var/2)*(np.random.randn(*tx_cp.shape) + 1j*np.random.randn(*tx_cp.shape))
        rx_awgn = tx_cp + noise

        
        # convolução tx_cp * h
        # Canal Rayleigh puro: convolução do sinal com resposta ao impulso do canal
        rx_ray = np.convolve(tx_cp, h, mode='full')[:tx_cp.size]

        # --- Rayleigh + AWGN ---
        # Canal Rayleigh + AWGN: aplica ruído ao sinal Rayleigh
        noise2 = np.sqrt(noise_var/2)*(np.random.randn(*rx_ray.shape)
                                      +1j*np.random.randn(*rx_ray.shape))
        rx_ray_awgn = rx_ray + noise2
        
        # função auxiliar de recebimento, FFT, demod e BER
        # Função interna para recepção, equalização e cálculo de BER
        def ofdm_recv(s,type):
            # Passo 7: paraleliza e remove prefixo cíclico
            # Remove prefixo cíclico e converte para matriz N x S
            mat = s.reshape((N+CP, S), order='F')[CP:,:]
            # Passo 8: FFT de volta para o domínio da frequência e serializa
            # sempre faz FFT
            # FFT para voltar ao domínio da frequência
            Y = np.fft.fft(mat, axis=0)   # N×S

            # or type == 'ray'
            # Equalização se canal Rayleigh
            if type == 'awgn' or type == 'ray':
                # AWGN puro → não tem fading, basta serializar
                y = Y.reshape(-1, order='F') # Sem equalização para AWGN ou Rayleigh sem ruído
            else:
                # Rayleigh+AWGN → equaliza
                H_fft = np.fft.fft(h, N) # FFT do canal
                # evita divisões por valores muito pequenos
                H_fft[np.abs(H_fft) < 1e-3] = 1e-3 # Evita divisão por zero
                Y_eq = Y / H_fft[:,None] # Equalização no domínio da frequência
                y     = Y_eq.reshape(-1, order='F')


            # Passo 9: demodulação QPSK (decisão hard no sinal real/imaginário)
            b1  = (y.real<0).astype(int)
            b2  = (y.imag<0).astype(int)

            # — bloco de espectro símbolo 0 —
            # Cálculo e plotagem do espectro do primeiro símbolo
            freq_axis = np.linspace(-0.5, 0.5, N)
            
            t0  = tx[:, 0] # Símbolo OFDM transmitido
            r0  = mat[:, 0] # Símbolo OFDM recebido
            T0  = np.fft.fft(t0, N)
            R0  = np.fft.fft(r0, N)

            if type == "awgn":
                text = "AWGN"
            if type == "ray":
                text = "Canal Rayleigh"
            if type == "ray_awgn":
                text = "Canal Rayleigh + AWGN"

            plt.figure(figsize=(6,6))
            plt.subplot(2,1,1)
            plt.plot(freq_axis, np.abs(np.fft.fftshift(T0)))
            plt.title(f'Espectro Antes (símbolo 0) – QPSK - {text}')
            plt.xlabel('Frequência (Hz)')
            plt.ylabel('Magnitude')
            plt.grid(True)

            plt.subplot(2,1,2)
            plt.plot(freq_axis, np.abs(np.fft.fftshift(R0)))
            plt.title(f'Espectro Após AWGN ({snr_db} dB) – QPSK - {text}')
            plt.xlabel('Frequência (Hz)')
            plt.ylabel('Magnitude')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f'espectro_qpsk_{snr_db}dB_{type}.png')
            plt.close()
            # — fim bloco —

            # Passo 10: calcula BER comparando com bits transmitidos
            # Compara bits estimados com originais para calcular BER
            errs = np.sum(bits != np.vstack([b1,b2]))
            return errs/(bits.size)
        
        # Executa recepção para cada tipo de canal
        ber_awgn.append(ofdm_recv(rx_awgn,'awgn'))
        ber_ray.append(ofdm_recv(rx_ray,'ray'))
        ber_ray_awgn.append(ofdm_recv(rx_ray_awgn,'ray_awgn'))


    return ber_awgn, ber_ray, ber_ray_awgn


# 2) Simulação genérica M-QAM usando CommPy QAMModem
def simulate_ofdm_qam(M):
    modem = QAMModem(M)                 # inicializa modem QAM
    k     = modem.num_bits_symbol       # bits por símbolo QAM
    
    # Passo 2: gera vetor de bits aleatório de comprimento k*total
    # Gera um vetor serial de bits aleatórios e converte em paralelo (serial-to-parallel)
    bits = np.random.randint(0, 2, size=(k, total)).reshape(-1, order='F')
     
    # Passo 3: modula bits em símbolos QAM e paraleliza
    # Modula bits em símbolos QAM e reorganiza em matriz N subportadoras × S símbolos
    sym = modem.modulate(bits).reshape((N, S), order='F')
    
    # Passo 4: IFFT
    # IFFT: converte os símbolos modulados para o domínio do tempo
    tx = np.fft.ifft(sym, axis=0)
    
    # Passo 5: adiciona prefixo cíclico (as últimas CP amostras) e serializa
    tx_cp = np.vstack([tx[-CP:], tx]).reshape(-1, order='F')
    # Estatísticas do sinal transmitido: média e variância
    μ = np.mean(tx_cp)                                 # média complexa de tx_cp
    signal_var = np.mean(np.abs(tx_cp - μ)**2)         # var(tx_cp) = E[|x–μ|²]  # variância do sinal
    
    # Listas para armazenar BER (Bit Error Rate) em cada tipo de canal
    ber_awgn = [] # para canal AWGN puro
    ber_ray  = [] # para canal Rayleigh puro
    ber_ray_awgn = [] # para canal Rayleigh + AWGN

    # Varre cada nível de SNR em dB
    for snr_db in SNRs_dB:
        # Passo 6: canal AWGN
        # Converte SNR de dB para escala linear e define variância do ruído
        snr = 10**(snr_db/10)
        noise_var = signal_var / snr
        # --- AWGN puro --- ---: adiciona ruído gaussiano complexo
        noise = np.sqrt(noise_var/2)*(np.random.randn(*tx_cp.shape) + 1j*np.random.randn(*tx_cp.shape))
        rx_awgn = tx_cp + noise

        # convolução tx_cp * h ---: convolução com resposta ao impulso h
        rx_ray = np.convolve(tx_cp, h, mode='full')[:tx_cp.size]

        # --- Rayleigh + AWGN --- adiciona ruído ao sinal Rayleigh
        noise2 = np.sqrt(noise_var/2)*(np.random.randn(*rx_ray.shape)
                                      +1j*np.random.randn(*rx_ray.shape))
        rx_ray_awgn = rx_ray + noise2
        
        # função auxiliar de recebimento, FFT, demod e BER
        # Função interna para recepção, equalização e cálculo do BER
        def ofdm_recv(s,type):
            # Passo 7: paraleliza e remove prefixo cíclico e retorna matriz N×S
            mat = s.reshape((N+CP, S), order='F')[CP:,:]
            # Passo 8: FFT de volta para o domínio da frequência e serializa
            # sempre faz FFT
            # FFT para converter de volta ao domínio da frequência
            Y = np.fft.fft(mat, axis=0)   # N×S

            # or type == 'ray'
            # Equalização apenas nos casos de Rayleigh+AWGN
            if type == 'awgn' or type == 'ray':
                # AWGN puro → não tem fading, basta serializar
                y = Y.reshape(-1, order='F') # sem equalização
            else:
                # Rayleigh+AWGN → equaliza
                H_fft = np.fft.fft(h, N) # FFT do canal
                # evita divisões por valores muito pequenos
                H_fft[np.abs(H_fft) < 1e-3] = 1e-3 # evita divisão por zero
                Y_eq = Y / H_fft[:,None]  # equaliza em cada subportadora
                y     = Y_eq.reshape(-1, order='F')
                

            # Passo 9: demodulação por decisão hard
            bits_hat = modem.demodulate(y, 'hard')

            # — bloco de espectro símbolo 0 —
            freq_axis = np.linspace(-0.5, 0.5, N)
            
            t0  = tx[:, 0]
            r0  = mat[:, 0]
            T0  = np.fft.fft(t0, N)
            R0  = np.fft.fft(r0, N)

            if type == "awgn":
                text = "AWGN"
            if type == "ray":
                text = "Canal Rayleigh"
            if type == "ray_awgn":
                text = "Canal Rayleigh + AWGN"

            plt.figure(figsize=(6,6))
            plt.subplot(2,1,1)
            plt.plot(freq_axis, np.abs(np.fft.fftshift(T0)))
            plt.title(f'Espectro Antes (símbolo 0) – {M}-QAM - {text}')
            plt.xlabel('Frequência (Hz)'); plt.ylabel('Magnitude'); plt.grid(True)

            plt.subplot(2,1,2)
            plt.plot(freq_axis, np.abs(np.fft.fftshift(R0)))
            plt.title(f'Espectro Após AWGN ({snr_db} dB) – {M}-QAM - {text}')
            plt.xlabel('Frequência (Hz)'); plt.ylabel('Magnitude'); plt.grid(True)

            plt.tight_layout()
            plt.savefig(f'espectro_{M}qam_{snr_db}dB_{type}.png')
            plt.close()
            # — fim bloco —

            # Passo 10: calcula BER comparando com vetor de bits transmitidos
            # Cálculo do BER comparando bits estimados vs. originais
            errs = np.sum(bits != bits_hat)
            return errs / bits.size
        
        # Aplica ofdm_recv a cada canal e armazena o BER resultante
        ber_awgn.append(ofdm_recv(rx_awgn,'awgn'))
        ber_ray.append(ofdm_recv(rx_ray,'ray'))
        ber_ray_awgn.append(ofdm_recv(rx_ray_awgn,'ray_awgn'))

     # Retorna as três curvas de BER para os diferentes canais
    return ber_awgn, ber_ray, ber_ray_awgn

#Executa simulações para QPSK, 16-QAM e 64-QAM
# ------------------ Execução das simulações iniciais ------------------
BER_AWGN_QPSK, BER_RAY_QPSK, BER_RAY_AWGN_QPSK = simulate_ofdm_qpsk()
BER_AWGN_16QAM, BER_RAY_16QAM, BER_RAY_AWGN_16QAM = simulate_ofdm_qam(16)
BER_AWGN_64QAM, BER_RAY_64QAM, BER_RAY_AWGN_64QAM = simulate_ofdm_qam(64)

# ------------------ Geração de gráficos BER vs SNR ------------------
# 1) Canal AWGN
plt.figure(figsize=(8,5))
plt.semilogy(SNRs_dB, BER_AWGN_QPSK,  'o-', label='QPSK')
plt.semilogy(SNRs_dB, BER_AWGN_16QAM, 's-', label='16-QAM')
plt.semilogy(SNRs_dB, BER_AWGN_64QAM, '^-', label='64-QAM')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('BER vs SNR — Canal AWGN')
# fixa os ticks em potências de 10
# yticks = [1, 1e-1, 1e-2, 1e-3, 1e-4,1e-5,1e-6]
# ylabels = [r'$10^0$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$',r'$10^{-6}$']
# plt.yticks(yticks, ylabels)
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
# yticks = [1, 1e-1, 1e-2, 1e-3]
# ylabels = [r'$10^0$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$']
# plt.yticks(yticks, ylabels)
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
# yticks = [1, 1e-1, 1e-2, 1e-3]
# ylabels = [r'$10^0$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$']
# plt.yticks(yticks, ylabels)
plt.grid(which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('ber_ray_awgn.png')
plt.close()



# ------------------ Varredura de prefixo cíclico ------------------
# Lista de diferentes comprimentos de prefixo cíclico a testar
CP_list = [8,16,32]
# Definição das modulações a simular: função de simulação + rótulo para legenda
mods = [
    (simulate_ofdm_qpsk, 'QPSK'),
    (lambda: simulate_ofdm_qam(16), '16-QAM'),
    (lambda: simulate_ofdm_qam(64), '64-QAM')
]
# Definição dos canais: chave + função que seleciona a curva BER correspondente
canals = [
    ('AWGN',    lambda awgn, ray, rawn: awgn), # apenas BER no canal AWGN
    ('Rayleigh',lambda awgn, ray, rawn: ray), # apenas BER no canal Rayleigh
    ('Rayleigh+AWGN', lambda awgn, ray, rawn: rawn) # apenas BER no canal Rayleigh+AWGN
]

# Loop principal: para cada modulação e cada tipo de canal...
for mod_func, mod_label in mods:
    for canal_key, select_ber in canals:
        # Cria nova figura para este par (modulação, canal)
        plt.figure(figsize=(8,5))
        # Para cada valor de CP, atualiza global e recalcula BER
        for CP in CP_list: 
            globals()['CP'] = CP # atualiza o prefixo cíclico usado pelas funções de simulação
            ber_awgn, ber_ray, ber_rawn = mod_func() # executa simulação completa
            ber = select_ber(ber_awgn, ber_ray, ber_rawn) # escolhe a curva de BER do canal atual
            # Plota BER vs SNR com marcador e legenda indicando o CP usado
            plt.semilogy(
                SNRs_dB, ber, '-o',
                label=f'CP={CP}'
            )
        # Ajusta títulos e rótulos do gráfico
        plt.title(f'{mod_label} — {canal_key}')
        plt.xlabel('SNR (dB)')
        plt.ylabel('BER')
        # yticks = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        # ylabels = [r'$10^0$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$']
        # plt.yticks(yticks, ylabels)
        plt.grid(which='both', ls='--', alpha=0.6)
        plt.legend(title='Prefixo Cíclico', loc='upper right')
        plt.tight_layout()
        # Salva o gráfico em arquivo em vez de exibir
        # em vez de plt.show(), salve o arquivo:
        filename = f'BER_{mod_label}_{canal_key}.png'.replace('+','p').replace(' ','_')
        plt.savefig(filename, dpi=300)
        plt.close() # fecha a figura para liberar memória
        print(f'Salvo: {filename}')
        # plt.show()
        # plt.close()

# ------------------ 3) Simulação com Código de Repetição ------------------
def simulate_ofdm_qpsk_rep(r=3, canal="rayleigh_awgn"):
    k = 2  # bits por símbolo QPSK
    # Gera matriz de bits aleatórios e repete cada bit r vezes (código de repetição)
    bits = np.random.randint(0, 2, size=(k, total))
    bits_rep = np.repeat(bits, r, axis=1)
    # Mapeamento QPSK nos bits repetidos
    sym = (1 - 2*bits_rep[0]) + 1j*(1 - 2*bits_rep[1])
    # Serial-to-parallel: organiza em matriz N subportadoras × (total*r / N) símbolos
    sym = sym.reshape((N, -1), order='F')

    # IFFT
    tx = np.fft.ifft(sym, axis=0)
    # Adição de prefixo cíclico e serialização
    tx_cp = np.vstack([tx[-CP:], tx]).reshape(-1, order='F')
     # Estatísticas do sinal transmitido
    μ = np.mean(tx_cp)
    signal_var = np.mean(np.abs(tx_cp - μ)**2)

    ber_rep = [] # lista de BER com repetição
    # Varre SNRs
    for snr_db in SNRs_dB:
        # Converte SNR dB → linear e define variância do ruído
        snr = 10**(snr_db/10)
        noise_var = signal_var / snr
        noise = np.sqrt(noise_var/2)*(np.random.randn(*tx_cp.shape) + 1j*np.random.randn(*tx_cp.shape))

        # Seleção do canal
        if canal == "awgn":
            rx = tx_cp + noise # AWGN puro
        elif canal == "rayleigh_awgn":
            rx = np.convolve(tx_cp, h, mode='full')[:tx_cp.size] + noise # Rayleigh + AWGN
        elif canal == "rayleigh":
            rx = np.convolve(tx_cp, h, mode='full')[:tx_cp.size] # Rayleigh puro
        else:
            raise ValueError("Canal deve ser 'awgn', 'rayleigh_awgn' ou 'rayleigh'.")

        # Remove o prefixo cíclico e reconstrói matriz N×?
        mat = rx.reshape((N+CP, -1), order='F')[CP:, :]
        # FFT para domínio da frequência
        Y = np.fft.fft(mat, axis=0)

        # Equalização no caso Rayleigh+AWGN
        if canal in ["rayleigh_awgn"]:
            H_fft = np.fft.fft(h, N)
            H_fft[np.abs(H_fft) < 1e-3] = 1e-3 # evita divisões problemáticas
            Y = Y / H_fft[:, None]

        # Serializa de volta para vetor
        y = Y.reshape(-1, order='F')
        # Decisão dura QPSK nos rads real e imaginário
        b1_rep = (y.real < 0).astype(int)
        b2_rep = (y.imag < 0).astype(int)
        # Ajusta tamanho para múltiplos de r e reshape para blocos de r repetições
        valid_len = (b1_rep.size // r) * r
        b1_rep = b1_rep[:valid_len].reshape(-1, r)
        b2_rep = b2_rep[:valid_len].reshape(-1, r)

        # Decisão por maioria: soma de cada bloco comparada a r/2
        b1_dec = (np.sum(b1_rep, axis=1) > r / 2).astype(int)
        b2_dec = (np.sum(b2_rep, axis=1) > r / 2).astype(int)
        # Referência: bits originais correspondentes
        b1_ref = bits[0, :len(b1_dec)]
        b2_ref = bits[1, :len(b2_dec)]
        # Conta erros e calcula BER
        err = np.sum(b1_dec != b1_ref) + np.sum(b2_dec != b2_ref)
        total_bits = b1_ref.size + b2_ref.size
        ber_rep.append(err / total_bits)

    return ber_rep # retorna curva de BER para o código de repetição

# Executa repetição para QPSK no canal AWGN e Rayleigh+AWGN
# Executa simulações de repetição (r=3) para QPSK em dois canais
BER_AWGN_QPSK_REP = simulate_ofdm_qpsk_rep(r=3, canal="awgn") # AWGN puro
BER_RAY_QPSK_REP = simulate_ofdm_qpsk_rep(r=3, canal="rayleigh") # Rayleigh puro
BER_RAY_AWGN_QPSK_REP = simulate_ofdm_qpsk_rep(r=3, canal="rayleigh_awgn") # Rayleigh + AWGN

# Gera gráficos comparativos de repetição vs sem repetição
# Canal AWGN
plt.figure(figsize=(8,5))
# Plota curva sem repetição (resultado de simulate_ofdm_qpsk em AWGN)
plt.semilogy(SNRs_dB, BER_AWGN_QPSK, 'o-', label='Sem Repetição - AWGN')
# Plota curva com repetição (resultado de simulate_ofdm_qpsk_rep em AWGN)
plt.semilogy(SNRs_dB, BER_AWGN_QPSK_REP, 's--', label='Com Repetição (r=3) - AWGN')

plt.xlabel('SNR (dB)') # rótulo eixo x
plt.ylabel('BER') # rótulo eixo y 
plt.title('QPSK — Código de Repetição — Canal AWGN') # título do gráfico
plt.grid(which='both', ls='--', alpha=0.6) # grade pontilhada
plt.legend() # legenda
plt.tight_layout() # ajusta layout para evitar cortes
plt.savefig('ber_qpsk_rep_awgn.png', dpi=300) # salva figura em arquivo
plt.close() # fecha figura para liberar memória

# QPSK — Rayleigh puro
plt.figure(figsize=(8,5))
plt.semilogy(SNRs_dB, BER_RAY_QPSK, 'o-', label='Sem Repetição - Rayleigh')
plt.semilogy(SNRs_dB, BER_RAY_QPSK_REP, 's--', label='Com Repetição (r=3) - Rayleigh')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('QPSK — Código de Repetição — Canal Rayleigh')
plt.grid(which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('ber_qpsk_rep_rayleigh.png', dpi=300)
plt.close()


# Canal Rayleigh + AWGN
plt.figure(figsize=(8,5))
# Plota curva sem repetição em Rayleigh+AWGN
plt.semilogy(SNRs_dB, BER_RAY_AWGN_QPSK, 'o-', label='Sem Repetição - Rayleigh+AWGN')
# Plota curva com repetição em Rayleigh+AWGN
plt.semilogy(SNRs_dB, BER_RAY_AWGN_QPSK_REP, 's--', label='Com Repetição (r=3) - Rayleigh+AWGN')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('QPSK — Código de Repetição — Canal Rayleigh + AWGN')
plt.grid(which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('ber_qpsk_rep_rayleigh_awgn.png', dpi=300)
plt.close()


def simulate_ofdm_qam_rep(M, r=3, canal="rayleigh_awgn"):
    # Inicializa o modem QAM de ordem M
    modem = QAMModem(M)
    # Número de bits por símbolo para essa modulação
    k = modem.num_bits_symbol
    # Gera vetor serial de bits aleatórios e converte para paralelo
    bits = np.random.randint(0, 2, size=(k, total)).reshape(-1, order='F')
    # Aplica repetição de cada bit r vezes (código de repetição)
    bits_rep = np.repeat(bits, r)
    # Mapeia bits repetidos em símbolos QAM e organiza em matriz N x ?
    sym = modem.modulate(bits_rep).reshape((N, -1), order='F')
    # IFFT: converte símbolos para domínio do tempo
    tx = np.fft.ifft(sym, axis=0)
    # Adiciona prefixo cíclico e serializa sinal transmitido
    tx_cp = np.vstack([tx[-CP:], tx]).reshape(-1, order='F')
    # Estatísticas do sinal transmitido
    μ = np.mean(tx_cp) # média complexa de tx_cp
    signal_var = np.mean(np.abs(tx_cp - μ)**2) # variância do sinal

    ber_rep = [] # lista para armazenar BER após repetição
    # Varre cada valor de SNR
    for snr_db in SNRs_dB:
        # Converte SNR de dB para escala linear e calcula variância de ruído
        snr = 10**(snr_db/10)
        noise_var = signal_var / snr
        # Gera ruído AWGN
        noise = np.sqrt(noise_var/2)*(np.random.randn(*tx_cp.shape) + 1j*np.random.randn(*tx_cp.shape))
        # Seleciona comportamento do canal
        if canal == "awgn":
            rx = tx_cp + noise # AWGN puro
        elif canal == "rayleigh_awgn":
            # Convolução com canal Rayleigh + AWGN
            rx = np.convolve(tx_cp, h, mode='full')[:tx_cp.size] + noise
        elif canal == "rayleigh":
            # Apenas canal Rayleigh (sem ruído adicional)
            rx = np.convolve(tx_cp, h, mode='full')[:tx_cp.size]
        else:
            raise ValueError("Canal deve ser 'awgn', 'rayleigh_awgn' ou 'rayleigh'.")
        # Remove prefixo cíclico e reconstrói matriz N x ?
        mat = rx.reshape((N+CP, -1), order='F')[CP:, :]
        # FFT para voltar ao domínio da frequência
        Y = np.fft.fft(mat, axis=0)
        # Equalização para caso Rayleigh+AWGN
        if canal in ["rayleigh_awgn"]:
            H_fft = np.fft.fft(h, N)
            H_fft[np.abs(H_fft) < 1e-3] = 1e-3 # evita divisões por zero
            Y = Y / H_fft[:, None]
        # Serializa de volta para vetor de símbolos
        y = Y.reshape(-1, order='F')
        # Demodula símbolos repetidos (hard decision)
        bits_hat_rep = modem.demodulate(y, 'hard')
        # Ajusta tamanho para múltiplos de r e organiza em blocos de r
        valid_len = (len(bits_hat_rep) // r) * r
        bits_hat_rep = bits_hat_rep[:valid_len]
        bits_hat_reshape = bits_hat_rep.reshape(-1, r)
        # Decisão por maioria em cada bloco
        bits_hat_dec = (np.sum(bits_hat_reshape, axis=1) > r / 2).astype(int)
        # Define bits de referência (originais)
        bits_ref = bits[:len(bits_hat_dec)]
        # Conta número de erros
        err = np.sum(bits_hat_dec != bits_ref)
        # Calcula e armazena BER para este SNR
        ber_rep.append(err / bits_ref.size)

    return ber_rep # retorna curva de BER com repetição para cada SNR

# Simulações de repetição em M-QAM com r=3 para distintos cenários de canal
# 64-QAM — Código de Repetição em AWGN e Rayleigh+AWGN
BER_AWGN_64QAM_REP = simulate_ofdm_qam_rep(64, r=3, canal="awgn") # 64-QAM no canal AWGN
BER_RAY_AWGN_64QAM_REP = simulate_ofdm_qam_rep(64, r=3, canal="rayleigh_awgn")  # 64-QAM no canal Rayleigh+AWGN
# Gráfico comparativo para AWGN puro
plt.figure(figsize=(8,5))
# Curva sem repetição (resultado de simulate_ofdm_qam)
plt.semilogy(SNRs_dB, BER_AWGN_64QAM, 'o-', label='Sem Repetição - AWGN')
# Curva com repetição (resultado de simulate_ofdm_qam_rep)
plt.semilogy(SNRs_dB, BER_AWGN_64QAM_REP, 's--', label='Com Repetição (r=3) - AWGN')
plt.xlabel('SNR (dB)') # Rótulo do eixo x
plt.ylabel('BER') # Rótulo do eixo y
plt.title('64-QAM — Código de Repetição — Canal AWGN') # Título do gráfico
plt.grid(which='both', ls='--', alpha=0.6) # Grade de fundo
plt.legend() # Exibe legenda
plt.tight_layout() # Ajusta layout
plt.savefig('ber_64qam_rep_awgn.png', dpi=300) # Salva figura
plt.close() # Fecha figura

# Gráfico comparativo para Rayleigh + AWGN
plt.figure(figsize=(8,5))
# Curva sem repetição
plt.semilogy(SNRs_dB, BER_RAY_AWGN_64QAM, 'o-', label='Sem Repetição - Rayleigh+AWGN')
# Curva com repetição
plt.semilogy(SNRs_dB, BER_RAY_AWGN_64QAM_REP, 's--', label='Com Repetição (r=3) - Rayleigh+AWGN')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('64-QAM — Código de Repetição — Canal Rayleigh + AWGN')
plt.grid(which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('ber_64qam_rep_rayleigh_awgn.png', dpi=300)
plt.close()

# 16-QAM — Código de Repetição em AWGN e Rayleigh+AWGN
BER_AWGN_16QAM_REP = simulate_ofdm_qam_rep(16, r=3, canal="awgn")
BER_RAY_AWGN_16QAM_REP = simulate_ofdm_qam_rep(16, r=3, canal="rayleigh_awgn")
# AWGN puro para 16-QAM
plt.figure(figsize=(8,5))
plt.semilogy(SNRs_dB, BER_AWGN_16QAM, 'o-', label='Sem Repetição - AWGN')
plt.semilogy(SNRs_dB, BER_AWGN_16QAM_REP, 's--', label='Com Repetição (r=3) - AWGN')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('16-QAM — Código de Repetição — Canal AWGN')
plt.grid(which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('ber_16qam_rep_awgn.png', dpi=300)
plt.close()
# Rayleigh + AWGN para 16-QAM
plt.figure(figsize=(8,5))
plt.semilogy(SNRs_dB, BER_RAY_AWGN_16QAM, 'o-', label='Sem Repetição - Rayleigh+AWGN')
plt.semilogy(SNRs_dB, BER_RAY_AWGN_16QAM_REP, 's--', label='Com Repetição (r=3) - Rayleigh+AWGN')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('16-QAM — Código de Repetição — Canal Rayleigh + AWGN')
plt.grid(which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('ber_16qam_rep_rayleigh_awgn.png', dpi=300)
plt.close()

# 16-QAM — Código de Repetição no canal Rayleigh puro
# 16-QAM — Rayleigh puro
BER_RAY_16QAM_REP = simulate_ofdm_qam_rep(16, r=3, canal="rayleigh")
plt.figure(figsize=(8,5))
plt.semilogy(SNRs_dB, BER_RAY_16QAM, 'o-', label='Sem Repetição - Rayleigh')
plt.semilogy(SNRs_dB, BER_RAY_16QAM_REP, 's--', label='Com Repetição (r=3) - Rayleigh')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('16-QAM — Código de Repetição — Canal Rayleigh')
plt.grid(which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('ber_16qam_rep_rayleigh.png', dpi=300)
plt.close()

# 64-QAM — Código de Repetição no canal Rayleigh puro
# 64-QAM — Rayleigh puro
BER_RAY_64QAM_REP = simulate_ofdm_qam_rep(64, r=3, canal="rayleigh")
plt.figure(figsize=(8,5))
plt.semilogy(SNRs_dB, BER_RAY_64QAM, 'o-', label='Sem Repetição - Rayleigh')
plt.semilogy(SNRs_dB, BER_RAY_64QAM_REP, 's--', label='Com Repetição (r=3) - Rayleigh')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('64-QAM — Código de Repetição — Canal Rayleigh')
plt.grid(which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('ber_64qam_rep_rayleigh.png', dpi=300)
plt.close()
