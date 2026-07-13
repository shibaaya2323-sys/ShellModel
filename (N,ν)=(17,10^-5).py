
import numpy as np
import matplotlib.pyplot as plt

# --- 1. パラメータ設定と型の明示 ---
# 実数定数はすべて float64（倍精度実数）として扱う
N = 17
q = np.float64(2.0)
k0 = np.float64(2.0**(-4))
nu = np.float64(1.0e-5)
beta = np.float64(0.5)
f = np.complex128(5.0e-3 * (1.0 + 1.0j)) # 外力は明示的に complex128

# 配列の生成時にも必ず dtype を明示する
n_arr = np.arange(1, N + 1, dtype=np.float64)
n_k = k0 * (q ** n_arr)

# 係数 c_n の計算（実数配列なので float64）
n_c1 = np.zeros(N, dtype=np.float64)
n_c2 = np.zeros(N, dtype=np.float64)
n_c3 = np.zeros(N, dtype=np.float64)

for i in range(N - 2):
    n_c1[i] = n_k[i]

for i in range(1, n_c2.size - 1): # i = 1 から N-2
    n_c2[i] = -beta * n_k[i-1]

for i in range(2, n_c3.size):     # i = 2 から N-1
    n_c3[i] = (beta - 1.0) * n_k[i-2]


# --- 2. 非線形項のみを計算する関数（内部も倍精度で統一） ---
def n_nonlinear_n(n_u):
    # 戻り値の配列を明示的に complex128 で確保
    n_nl = np.zeros(N, dtype=np.complex128)

    # 端っこの例外処理（1j も明示的に複素数型にするとなお安全です）
    I_comp = np.complex128(1.0j)

    n_nl[0] = I_comp * (n_c1[0] * np.conj(n_u[1]) * np.conj(n_u[2]))
    n_nl[1] = I_comp * (n_c1[1] * np.conj(n_u[2]) * np.conj(n_u[3]) +
                        n_c2[1] * np.conj(n_u[0]) * np.conj(n_u[2]))

    # 安全な中間エリア
    for i in range(2, N - 2):
        n_nl[i] = I_comp * (
            n_c1[i] * np.conj(n_u[i+1]) * np.conj(n_u[i+2]) +
            n_c2[i] * np.conj(n_u[i-1]) * np.conj(n_u[i+1]) +
            n_c3[i] * np.conj(n_u[i-2]) * np.conj(n_u[i-1])
        )

    n_nl[N-2] = I_comp * (n_c2[N-2] * np.conj(n_u[N-3]) * np.conj(n_u[N-1]) +
                          n_c3[N-2] * np.conj(n_u[N-4]) * np.conj(n_u[N-3]))
    n_nl[N-1] = I_comp * (n_c3[N-1] * np.conj(n_u[N-3]) * np.conj(n_u[N-2]))

    # 外力の追加
    n_nl[3] += f

    return n_nl


# --- ⏰ 時間に関する設定 ---
dt = np.float64(0.0005)
total_time = np.float64(500.0)
avg_start_time = np.float64(350.0)

total_steps = int(round(total_time / dt))
avg_start_step = int(round(avg_start_time / dt))

# 乱数のシードを固定（前回のやつもしっかり組込！）
np.random.seed(42)

# 初期条件（流速は複素数なので complex128）
n_u = np.zeros(N, dtype=np.complex128)
for i in range(N):
    energy_initial = (n_k[i]**2) * np.exp(-(n_k[i]**2))
    phase = np.random.uniform(0.0,2.0 * np.pi)
    n_u[i] = np.sqrt(2.0 * n_k[i] * energy_initial) * np.exp(1.0j * phase)

# 積分因子法の係数事前計算（実数配列なので float64）
n_E_visc = np.array(np.exp(-nu * (n_k**2) * dt), dtype=np.float64)
n_E_visc_half = np.array(np.exp(-nu * (n_k**2) * (dt * 0.5)), dtype=np.float64)

# 平均化のための貯金箱
n_E_kn_sum = np.zeros(N, dtype=np.float64)
epsilon_sum = np.float64(0.0)
avg_count = 0

print(f"総ステップ数: {total_steps} 回")
print("倍精度（complex128）明示モードで計算中...")

for step in range(total_steps):
    # --- 改良型ルンゲ・クッタ（RK4）ステップ ---
    n_k1 = n_nonlinear_n(n_u)

    n_u_half1 = (n_u + 0.5 * dt * n_k1) * n_E_visc_half
    n_k2 = n_nonlinear_n(n_u_half1)

    n_u_half2 = n_u * n_E_visc_half + 0.5 * dt * n_k2
    n_k3 = n_nonlinear_n(n_u_half2)

    n_u_full = n_E_visc * n_u + dt * n_E_visc_half * n_k3
    n_k4 = n_nonlinear_n(n_u_full)

    # 次のステップの u を統合
    n_u_next = n_u * n_E_visc + (dt / 6.0) * (
        n_k1 * n_E_visc + 2.0 * n_k2 * n_E_visc_half + 2.0 * n_k3 * n_E_visc_half + n_k4
    )

    n_u = n_u_next

    # --- 後半の統計定常状態のデータをサンプリング ---
    if step >= avg_start_step:
        n_abs_u_sq = np.abs(n_u)**2
        n_E_kn_instant = (n_abs_u_sq / (2.0 * n_k))
        n_E_kn_sum += n_E_kn_instant
        epsilon_instant = nu * np.sum((n_k**2) * n_abs_u_sq)
        epsilon_sum += epsilon_instant
        avg_count += 1

print("計算完了！")

# 時間平均エネルギースペクトル
n_E_kn_avg = n_E_kn_sum / avg_count
epsilon_avg = epsilon_sum / avg_count

# --- 4. 論文通りの無次元化（正規化）処理 ---
k_d = (epsilon_avg / (nu**3))**(0.25)

n_x_normalized = n_k / k_d
n_y_normalized = n_E_kn_avg / ((epsilon_avg**(0.25)) * (nu**(1.25)))

# --- 5. プロット ---
plt.figure(figsize=(7, 6))
plt.loglog(n_x_normalized, n_y_normalized, 'x', label=r'$\nu=10^{-5}$ (Double Precision RK4)', color='black')

n_x_line = np.logspace(-6, -1, 100)
n_y_line = 0.5 * (n_x_line ** (-5.0/3.0))
plt.loglog(n_x_line, n_y_line, color='black', linewidth=1.5, label='slope $-5/3$')

plt.xlim(1.0e-8, 1.0e2)
plt.ylim(1.0e-10, 1.0e14)
plt.xlabel(r'$k / k_d$')
plt.ylabel(r'$E_e(k / k_d)$')
plt.title('Fig. 1 Reconstructed (Strictly Double Precision)')
plt.grid(True, which="both", ls=":", alpha=0.3)
plt.legend(loc='lower left')
plt.show()
