import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from astropy.coordinates import Angle
import astropy.units as u
from pyproj import Geod
from scipy.stats import norm

def dms_to_dd(deg, minutes, seconds, sign=1):
    """
    Converte DMS para decimal usando Astropy Angle.
    deg, minutes, seconds: componentes DMS
    sign: +1 para N/E, -1 para S/W
    """
    ang = Angle(f"{abs(deg)}d{minutes}m{seconds}s")
    return sign * ang.to(u.deg).value

# --- 2) coordenadas do transmissor (O) e boresight (B) --- (O) e boresight (B) ---
lat_O = dms_to_dd(22, 54, 9.83, sign=-1)
lon_O = dms_to_dd(43, 6, 57.63, sign=-1)
lat_B = dms_to_dd(22, 54, 16.03, sign=-1)
lon_B = dms_to_dd(43, 6, 46.11, sign=-1)

# pega o azimute de O para B usando uma lib pronta do python
geod = Geod(ellps='WGS84')
bearing_OB, back_az, dist = geod.inv(lon_O, lat_O, lon_B, lat_B)

# --- 3) leitura do CSV sem header ---
df = pd.read_csv(
    './Lat_long_dist_Pr (1).csv',
    header=None,
    names=['latitude','longitude','dist_km','Pr_dBm']
)

# --- 4) cálculo de θ, R(θ) e Gt(θ) ---
def calc_theta(row):
    # pega o azimute de O para A
    bOA, back_az, dist = geod.inv(lon_O, lat_O, row['longitude'], row['latitude'])
    #a diferenca do angulo para um ponto arbitrario para boresight para vai dar o angulo teta da imagem
    return abs(bOA - bearing_OB)

#aplica a funcao para achar o angulo teta
df['theta_deg'] = df.apply(calc_theta, axis=1)

#coeficientes do polinomio p R(teta)
coefs = np.array([
    6.68119099491645e-10,
   -1.10210602759622e-07,
    6.97748729121278e-06,
   -0.000211522819954194,
    0.00309474470699973,
   -0.0206164470242935,
   -0.00685676523907350,
   -0.000491338726848875
])
#dado no enunciado - pega os coeficientes e o angulo teta para cada ponto arbitrario e forma o polinomio de ordem 7 
df['R_theta'] = np.polyval(coefs, df['theta_deg'])
#Ganho da antena transmissora Gt
df['Gt_dBi'] = 14.1 + df['R_theta']   # ganho Tx corrigido


# --- 5) parâmetros do enunciado ---
Pt_dBm, Gr_dBi, f_MHz = -7.0, 2.0, 850.0

# --- 6) modelos FS (Friis), CI e AB ---
c = 3e8                     # velocidade da luz (m/s)
lam = c / (f_MHz * 1e6)     # comprimento de onda (m)

df['d_m'] = df['dist_km'] * 1000

# Modelo FS usando fórmula correta do slide
# PL(dB) = -10 log10( (Gt*Gr*λ^2) / (4πd)^2 )
# 1) coeficientes teóricos:
R0 = np.polyval(coefs, 0.0)

B_FS = 20.0
A_FS = (
    20 * np.log10(4 * math.pi / lam)
    #- (14.1 - R0)         # se variável por linha, fica vetor
    #- Gr_dBi
)

# 2) predição via log-distance:
df['PL_FS_pred'] = A_FS + B_FS * np.log10(df['d_m'])

# --- 7) path-loss medido para CI e AB ---
# defina antes as perdas e ganhos do seu sistema:
Lcab_tx_dB = 1.5    # perda no cabo TX (dB)
Lcab_rx_dB = 2*1.5    # perda no cabo RX (dB)
GLNA_dB    = 26.0   # ganho do LNA (dB)

df['PL_meas'] = (
    Pt_dBm
    + df['Gt_dBi']       # ganho da antena Tx
    - Lcab_tx_dB         # perda cabo Tx
    + Gr_dBi             # ganho da antena Rx
    - Lcab_rx_dB         # perda cabo Rx
    + GLNA_dB            # ganho do LNA
    - df['Pr_dBm']       # potência recebida medida
)

results = pd.DataFrame({
    'Modelo':          ['FS'],
    'Intercepto (dB)': [A_FS],
    'Declive':         [B_FS]
})
print(results.to_string(index=False))


# --- 8) Modelo CI (Close-In) ---
# Monta sistema y = B·x, com y = PL_meas − A_FS e x = log10(d/d0)
x_ci = np.log10(df['d_m']).values.reshape(-1, 1)
y_ci = (df['PL_meas'] - A_FS).values

# Resolve B pelo método dos mínimos-quadrados
B_CI, *_ = np.linalg.lstsq(x_ci, y_ci, rcond=None)

# Predição do modelo CI
df['PL_CI_pred'] = A_FS + B_CI[0] * np.log10(df['d_m'])

results = pd.DataFrame({
    'Modelo':          ['CI'],
    'Intercepto (dB)': [A_FS],
    'Declive':         [B_CI[0]]
})
print(results.to_string(index=False))


# --- Modelo AB (Floating Intercept) ---
# Monta a matriz X_ab com coluna de 1's (intercepto) e log10(d_m)
X_ab = np.column_stack([
    np.ones(len(df)), 
    np.log10(df['d_m'])
])
# y_ab é o vetor de PL medido
y_ab = df['PL_meas'].values

# Resolve [A_AB, B_AB] pelo método dos mínimos-quadrados
(A_AB, B_AB), *_ = np.linalg.lstsq(X_ab, y_ab, rcond=None)

# Predição do modelo AB
df['PL_AB_pred'] = A_AB + B_AB * np.log10(df['d_m'])

# Exibe tabela de coeficientes
results_ab = pd.DataFrame({
    'Modelo':          ['AB'],
    'Intercepto (dB)': [A_AB],
    'Declive':         [B_AB]
})
print(results_ab.to_string(index=False))

# --- 8) plot comparativo de path-loss ---
plt.figure()
plt.scatter(df['dist_km'], df['PL_meas'], label='Medido', s=15)
plt.plot(df['dist_km'], df['PL_FS_pred'], '--', label='FS', color='red')
plt.plot(df['dist_km'], df['PL_CI_pred'], '--', label='CI', color='green') 
plt.plot(df['dist_km'], df['PL_AB_pred'], '--', label='AB', color='orange')  
plt.xscale('log')
plt.xlabel('Distância (km)')
plt.ylabel('PL (dB)')
plt.legend()
plt.show()

# --- 1) calcular erros ---
error_FS = df['PL_meas'] - df['PL_FS_pred']
error_CI = df['PL_meas'] - df['PL_CI_pred']
error_AB = df['PL_meas'] - df['PL_AB_pred']

# --- 2) função de plotagem ---
def plot_error_with_gaussian(error, title):
    mu    = error.mean() #Calcula a media aritmetica do vetor error
    sigma = error.std(ddof=0)      #Calcula o desvio-padrao populacional de error
    # O parametro ddof=0 garante que dividimos por N, nao por 𝑁−1
    x     = np.linspace(error.min(), error.max(), 1000) #Gera um 
    #array x de 1000 pontos igualmente espaçados entre o valor minimo e o valor maximo de error. 
    # Esses pontos servem de “eixo x” para avaliar a curva normal.
    pdf   = norm.pdf(x, mu, sigma) # usa a pdf com mu e sigma para construcao

    plt.figure()
    plt.hist(error, bins=30, density=True, alpha=0.6)
    plt.plot(x, pdf, linewidth=2)
    plt.title(title)
    plt.xlabel('Erro (dB)')
    plt.ylabel('Densidade')
    plt.grid(True)
    plt.show()

# --- 3) gerar os três gráficos ---
plot_error_with_gaussian(error_FS, 'Distribuição de Erro – Modelo FS')
plot_error_with_gaussian(error_CI, 'Distribuição de Erro – Modelo CI')
plot_error_with_gaussian(error_AB, 'Distribuição de Erro – Modelo AB')

# calcula média e desvio-padrão para cada modelo
stats = pd.DataFrame({
    'Modelo': ['FS', 'CI', 'AB'],
    'Média do Erro (dB)': [
        error_FS.mean(),
        error_CI.mean(),
        error_AB.mean()
    ],
    'Desvio Padrão (dB)': [
        error_FS.std(ddof=0),   # ddof=0 para desvio padrão populacional
        error_CI.std(ddof=0),
        error_AB.std(ddof=0)
    ]
})

print(stats)