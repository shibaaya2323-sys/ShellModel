
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


# --- ⏰ 時間設定を変えて計算する関数 ---
def run_simulation(dt, total_time, avg_start_time):

    dt = np.float64(dt)
    total_time = np.float64(total_time)
    avg_start_time = np.float64(avg_start_time)

    total_steps = int(round(total_time / dt))
    avg_start_step = int(round(avg_start_time / dt))

    # 乱数のシードを固定
    np.random.seed(42)

    # 初期条件
    n_u = np.zeros(N, dtype=np.complex128)

    for i in range(N):
        energy_initial = (n_k[i]**2) * np.exp(-(n_k[i]**2))
        phase = np.random.uniform(0.0, 2.0 * np.pi)

        n_u[i] = (np.sqrt(2.0 * n_k[i] * energy_initial) * np.exp(1.0j * phase))

    # 積分因子
    n_E_visc = np.array(np.exp(-nu * (n_k**2) * dt),dtype=np.float64)

    n_E_visc_half = np.array(np.exp(-nu * (n_k**2) * (dt * 0.5)),dtype=np.float64)

    # 平均化用
    n_E_kn_sum = np.zeros(N, dtype=np.float64)
    epsilon_sum = np.float64(0.0)
    injection_sum = np.float64(0.0)
    avg_count = 0

    print(f"総ステップ数: {total_steps} 回")
    print("倍精度（complex128）明示モードで計算中...")

    # 時間発展
    for step in range(total_steps):

        n_k1 = n_nonlinear_n(n_u)

        n_u_half1 = (n_u + 0.5 * dt * n_k1) * n_E_visc_half

        n_k2 = n_nonlinear_n(n_u_half1)

        n_u_half2 = (n_u * n_E_visc_half + 0.5 * dt * n_k2)

        n_k3 = n_nonlinear_n(n_u_half2)

        n_u_full = (n_E_visc * n_u + dt * n_E_visc_half * n_k3)

        n_k4 = n_nonlinear_n(n_u_full)

        n_u_next = (n_u * n_E_visc + (dt / 6.0) * (n_k1 * n_E_visc + 2.0 * n_k2 * n_E_visc_half + 2.0 * n_k3 * n_E_visc_half + n_k4))

        n_u = n_u_next

        # 統計定常状態で平均
        if step >= avg_start_step:
            n_abs_u_sq = np.abs(n_u)**2
            n_E_kn_instant = (n_abs_u_sq / (2.0 * n_k))
            n_E_kn_sum += n_E_kn_instant
            epsilon_instant = (nu * np.sum((n_k**2) * n_abs_u_sq))
            epsilon_sum += epsilon_instant
            injection_instant = np.float64(np.real(0.5 * (f * np.conj(n_u[3]) + n_u[3] * np.conj(f))))
            injection_sum += injection_instant
            avg_count += 1

    print("計算完了！")

    # 時間平均
    n_E_kn_avg = n_E_kn_sum / avg_count
    epsilon_avg = epsilon_sum / avg_count
    injection_avg = injection_sum / avg_count

    ratio = injection_avg / epsilon_avg

    relative_error = (abs(injection_avg - epsilon_avg) / abs(epsilon_avg))

    print()
    print("--- 統計定常状態での時間平均 ---")
    print(f"平均エネルギー注入率 <I> "f"= {injection_avg:.10e}")
    print(
        f"平均エネルギー散逸率 <epsilon> "f"= {epsilon_avg:.10e}")
    print(f"<I> / <epsilon> "f"= {ratio:.10f}")
    print(f"相対誤差 "f"= {relative_error:.6e}")

    # 無次元化
    k_d = (epsilon_avg / (nu**3))**0.25

    n_x_normalized = n_k / k_d

    n_y_normalized = (n_E_kn_avg/ ((epsilon_avg**0.25) * (nu**1.25)))

    # 結果を返す
    return {
        "dt": dt,
        "total_time": total_time,
        "avg_start_time": avg_start_time,
        "injection_avg": injection_avg,
        "epsilon_avg": epsilon_avg,
        "ratio": ratio,
        "relative_error": relative_error,
        "x_normalized": n_x_normalized.copy(),
        "y_normalized": n_y_normalized.copy()
    }
