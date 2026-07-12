

import numpy as np
import matplotlib.pyplot as plt

# --- 1. パラメータ設定 ---
N = 19               # シェル数
q = 2.0
k0 = 2.0**(-4)
nu = 1e-6            # 論文の最小粘性
beta = 0.5
f = 5e-3 * (1.0 + 1.0j)

n_arr = np.arange(1, N + 1)
k = k0 * (q ** n_arr)

# 係数 c_n の計算
c1, c2, c3 = np.zeros(N), np.zeros(N), np.zeros(N)
for i in range(N):
    if i < N - 2: c1[i] = k[i]
    if i > 0 and i < N - 1: c2[i] = -beta * k[i-1]
    if i > 1: c3[i] = (beta - 1.0) * k[i-2]

# --- 2. 支配方程式の右辺（非線形項＋外力）を計算する関数 ---
def governing_equation(u, include_forcing=True):
    nl = np.zeros(N, dtype=complex)
    for i in range(N):
        u_p1 = u[i+1] if i+1 < N else 0.0
        u_p2 = u[i+2] if i+2 < N else 0.0
        u_m1 = u[i-1] if i-1 >= 0 else 0.0
        u_m2 = u[i-2] if i-2 >= 0 else 0.0

        nl[i] = 1j * (
            c1[i] * np.conj(u_p1) * np.conj(u_p2) +
            c2[i] * np.conj(u_m1) * np.conj(u_p1) +
            c3[i] * np.conj(u_m2) * np.conj(u_m1)
        )
        if include_forcing and i == 3:  # 外力 (n=4)
            nl[i] += f
    return nl

# --- 3. エネルギーフラックス Π(k) を計算する関数 ---
def calculate_flux(u):
    # 純粋な非線形項による時間発展（外力はエネルギー転送関数に含めない）
    nl_pure = governing_equation(u, include_forcing=False)

    # 各シェルのエネルギー転送関数 T_n = d/dt (1/2 |u_n|^2) = Re(u_n^* * du_n/dt)
    # ※ 複素数変数モデルなので、実部をとることで物理的なエネルギー変化量になる
    T = np.real(np.conj(u) * nl_pure)

    # フラックス Π(k_n) は、シェル n より高波数側の T の総和
    Pi = np.zeros(N)
    for i in range(N - 1):
        Pi[i] = np.sum(T[i+1:])
    return Pi

# --- 4. 時間発展とサンプリング ---
dt = 0.0005
total_steps = 400000
avg_start_step = 300000

# 初期条件
u = np.zeros(N, dtype=complex)
for i in range(N):
    Ek0 = (k[i]**2) * np.exp(-(k[i]**2))
    u[i] = np.sqrt(2.0 * k[i] * Ek0) * np.exp(1j * np.random.uniform(0, 2*np.pi))

E_visc = np.exp(-nu * (k**2) * dt)
E_visc_half = np.exp(-nu * (k**2) * (dt / 2.0))

# 統計平均用の配列
E_kn_sum = np.zeros(N)
Pi_sum = np.zeros(N)
avg_count = 0

print("統計定常状態の計算中...")
for step in range(total_steps):
    # --- あなたが完璧に導出した IF-RK4 ステップ ---
    k1 = governing_equation(u, include_forcing=True)

    u_half1 = (u + 0.5 * dt * k1) * E_visc_half
    k2 = governing_equation(u_half1, include_forcing=True)

    u_half2 = (u * E_visc_half + 0.5 * dt * k2)
    k3 = governing_equation(u_half2, include_forcing=True)

    u_full = (E_visc * u + dt * E_visc_half * k3)
    k4 = governing_equation(u_full, include_forcing=True)

    u = (u * E_visc + (dt / 6.0) * (k1 * E_visc + 2.0 * k2 * E_visc_half + 2.0 * k3 * E_visc_half + k4))

    # --- 後半のサンプリング ---
    if step >= avg_start_step:
        # エネルギースペクトルの足し込み
        E_kn_sum += (np.abs(u)**2) / (2.0 * k)
        # エネルギーフラックスの足し込み（毎ステップの u からフラックスを計算）
        Pi_sum += calculate_flux(u)
        avg_count += 1

print("計算完了！プロットを作成します。")

# 時間平均値の算出
E_kn_avg = E_kn_sum / avg_count
Pi_avg = Pi_sum / avg_count

# --- 5. 散逸率 ε と Kolmogorov 波数 kd の計算 ---
# 論文の定義通り、エンストロフィー Q を経由して ε = 2 * ν * Q を計算
Q_enstrophy = np.sum((k**2) * (np.abs(u)**2) / 2.0) # 最終状態、または時間平均から算出
epsilon = 2.0 * nu * np.sum((k**3) * E_kn_avg)
kd = (epsilon / (nu**3))**(0.25)

# 無次元化
x_normalized = k / kd
y_normalized = Pi_avg / epsilon

# --- 6. 図2のプロット ---
plt.figure(figsize=(7, 5))
# フラックスは対数ではなく線形軸（片対数グラフ）
plt.semilogx(x_normalized[:-1], y_normalized[:-1], 'x', color='black', label=r'$\nu=10^{-6}$')

plt.xlim(1e-8, 1e2)
plt.ylim(-0.2, 1.2)
plt.xlabel(r'$k / k_d$')
plt.ylabel(r'$\Pi(k) / \varepsilon$')
plt.title('Fig. 2 Energy Flux Function (Reconstructed)')
plt.grid(True, which="both", ls=":", alpha=0.3)
plt.axhline(1.0, color='gray', linestyle='--', alpha=0.7) # 1.0の基準線
plt.axhline(0.0, color='gray', linestyle='-', alpha=0.5)
plt.legend(loc='lower left')
plt.show()
