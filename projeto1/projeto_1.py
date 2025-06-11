import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from astropy.coordinates import Angle
import astropy.units as u
from pyproj import Geod

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
    bOA, back_az, dist = geod.inv(lon_O, lat_O, row['longitude'], row['latitude'])
    Δ = abs(bOA - bearing_OB)
    return Δ

df['theta_deg'] = df.apply(calc_theta, axis=1)

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
df['R_theta'] = np.polyval(coefs, df['theta_deg'])
df['Gt_dBi'] = 14.1 + df['R_theta']   # ganho Tx corrigido
df['Gt_dBi'].to_excel("teste.xlsx")

# --- 5) parâmetros do link budget ---
Pt_dBm, Gr_dBi, f_MHz = -7.0, 2.0, 850.0

# --- 6) modelos FS (Friis), CI e AB ---
c = 3e8                     # velocidade da luz (m/s)
lam = c / (f_MHz * 1e6)     # comprimento de onda (m)
# conversão de ganhos e distância em metros
df['Gt_lin'] = 10**(df['Gt_dBi']/10)
Gr_lin = 10**(Gr_dBi/10)
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

# --- 8) plot comparativo de path-loss ---
plt.figure()
plt.scatter(df['dist_km'], df['PL_meas'], label='Medido', s=15)
plt.plot(df['dist_km'], df['PL_FS_pred'], label='FS')
plt.plot(df['dist_km'], df['PL_CI_pred'], '--', label='CI')  # <— adicionado
plt.xscale('log')
plt.xlabel('Distância (km)')
plt.ylabel('PL (dB)')
plt.legend()
plt.show()

