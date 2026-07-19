import numpy as np
from numba import njit


# ============================================================
# 1. パラメータ
# ============================================================

N = 27
q = np.float64(2.0)
k0 = np.float64(2.0**(-4))
nu = np.float64(1.0e-9)
beta = np.float64(0.5)
f = np.complex128(5.0e-3 * (1.0 + 1.0j))

n_arr = np.arange(1, N + 1, dtype=np.float64)
n_k = k0 * q**n_arr
n_k_sq = n_k**2

n_c1 = np.zeros(N, dtype=np.float64)
n_c2 = np.zeros(N, dtype=np.float64)
n_c3 = np.zeros(N, dtype=np.float64)

for i in range(N - 2):
    n_c1[i] = n_k[i]

for i in range(1, N - 1):
    n_c2[i] = -beta * n_k[i - 1]

for i in range(2, N):
    n_c3[i] = (beta - 1.0) * n_k[i - 2]


# ============================================================
# 2. 非線形項
# ============================================================

@njit
def n_nonlinear_numba(n_u, n_c1, n_c2, n_c3, f):

    N_local = n_u.size
    n_nl = np.zeros(N_local, dtype=np.complex128)

    n_nl[0] = 1.0j * (n_c1[0] * np.conj(n_u[1]) * np.conj(n_u[2]))

    n_nl[1] = 1.0j * (n_c1[1] * np.conj(n_u[2]) * np.conj(n_u[3]) +
                      n_c2[1] * np.conj(n_u[0]) * np.conj(n_u[2]))

    for i in range(2, N_local - 2):

        n_nl[i] = 1.0j * (n_c1[i] * np.conj(n_u[i + 1]) * np.conj(n_u[i + 2]) +
                          n_c2[i] * np.conj(n_u[i - 1]) * np.conj(n_u[i + 1]) +
                          n_c3[i] * np.conj(n_u[i - 2]) * np.conj(n_u[i - 1]))

    n_nl[N_local - 2] = 1.0j * (n_c2[N_local - 2] * np.conj(n_u[N_local - 3]) * np.conj(n_u[N_local - 1]) +
                                n_c3[N_local - 2] * np.conj(n_u[N_local - 4]) * np.conj(n_u[N_local - 3]))

    n_nl[N_local - 1] = 1.0j * (n_c3[N_local - 1] * np.conj(n_u[N_local - 3]) * np.conj(n_u[N_local - 2]))

    # 第4シェルへの外力
    n_nl[3] += f

    return n_nl


# ============================================================
# 3. 時間積分
# ============================================================

@njit
def run_simulation_numba(
    dt,
    total_time,
    avg_start_time,
    sample_interval,
    n_k,
    n_k_sq,
    n_c1,
    n_c2,
    n_c3,
    nu,
    f,
):

    total_steps = int(round(total_time / dt))
    avg_start_step = int(round(avg_start_time / dt))

    N_local = n_k.size

    # 初期条件
    np.random.seed(42)

    n_u = np.zeros(N_local, dtype=np.complex128)

    for i in range(N_local):

        energy_initial = (n_k_sq[i] * np.exp(-n_k_sq[i]))
        phase = np.random.uniform(0.0,2.0 * np.pi)
        n_u[i] = (np.sqrt(2.0 * n_k[i] * energy_initial) * np.exp(1.0j * phase))

    # 積分因子
    n_E_visc = np.exp(-nu * n_k_sq * dt)
    n_E_visc_half = np.exp(-nu * n_k_sq * dt * 0.5)

    # 平均化用
    n_E_kn_sum = np.zeros(N_local,dtype=np.float64)

    epsilon_sum = 0.0
    injection_sum = 0.0
    avg_count = 0

    # 時間積分
    for step in range(total_steps):

        n_k1 = n_nonlinear_numba(n_u,n_c1,n_c2,n_c3,f)
        n_u_half1 = (n_u + 0.5 * dt * n_k1) * n_E_visc_half

        n_k2 = n_nonlinear_numba(n_u_half1,n_c1,n_c2,n_c3,f)
        n_u_half2 = (n_u * n_E_visc_half + 0.5 * dt * n_k2)

        n_k3 = n_nonlinear_numba(n_u_half2,n_c1,n_c2,n_c3,f)
        n_u_full = (n_E_visc * n_u + dt * n_E_visc_half * n_k3)

        n_k4 = n_nonlinear_numba(n_u_full,n_c1,n_c2,n_c3,f)

        n_u = (n_u * n_E_visc +(dt / 6.0) * (n_k1 * n_E_visc + 2.0 * n_k2 * n_E_visc_half + 2.0 * n_k3 * n_E_visc_half + n_k4))

        # sample_intervalステップごとに平均を取る
        if (step >= avg_start_step and (step - avg_start_step) % sample_interval == 0):

            n_abs_u_sq = (n_u.real**2 + n_u.imag**2)
            n_E_kn_sum += n_abs_u_sq / (2.0 * n_k)
            epsilon_sum += (nu * np.sum(n_k_sq * n_abs_u_sq))
            injection_sum += np.real( f * np.conj(n_u[3]))
            avg_count += 1

    n_E_kn_avg = (n_E_kn_sum / avg_count)
    epsilon_avg = (epsilon_sum / avg_count)
    injection_avg = (injection_sum / avg_count)
    ratio = (injection_avg / epsilon_avg)
    relative_error = (abs(injection_avg - epsilon_avg) / abs(epsilon_avg))
    k_d = (epsilon_avg / nu**3)**0.25

    n_x_normalized = n_k / k_d
    n_y_normalized = (n_E_kn_avg / (epsilon_avg**0.25 * nu**1.25))

    return (
        injection_avg,
        epsilon_avg,
        ratio,
        relative_error,
        n_x_normalized,
        n_y_normalized,
        avg_count,
    )


# ============================================================
# 4. 実行用関数
# ============================================================

def run_simulation(
    dt,
    total_time,
    avg_start_time,
    sample_interval=10,
):

    total_steps = int(round(total_time / dt))

    print(f"総ステップ数: {total_steps:,} 回")
    print(f"平均値の取得間隔: "f"{sample_interval} ステップ")

    results = run_simulation_numba(
        np.float64(dt),
        np.float64(total_time),
        np.float64(avg_start_time),
        int(sample_interval),
        n_k,
        n_k_sq,
        n_c1,
        n_c2,
        n_c3,
        nu,
        f,
    )

    (
        injection_avg,
        epsilon_avg,
        ratio,
        relative_error,
        n_x_normalized,
        n_y_normalized,
        avg_count,
    ) = results

    print("計算完了！")
    print()
    print("--- 統計定常状態での時間平均 ---")
    print(f"平均エネルギー注入率 <I> "f"= {injection_avg:.10e}")
    print(f"平均エネルギー散逸率 <epsilon> "f"= {epsilon_avg:.10e}")
    print(f"<I> / <epsilon> "f"= {ratio:.10f}")
    print(f"相対誤差 "f"= {relative_error:.6e}")
    print(f"平均化に使用した点数 "f"= {avg_count:,}")

    return {
        "dt": np.float64(dt),
        "total_time": np.float64(total_time),
        "avg_start_time": np.float64(
            avg_start_time
        ),
        "sample_interval": sample_interval,
        "injection_avg": injection_avg,
        "epsilon_avg": epsilon_avg,
        "ratio": ratio,
        "relative_error": relative_error,
        "x_normalized":
            n_x_normalized.copy(),
        "y_normalized":
            n_y_normalized.copy(),
    }
