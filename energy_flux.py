import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# 1. パラメータ

N = 22
q = np.float64(2.0)
k0 = np.float64(2.0**(-4))
nu = np.float64(1.0e-7)
beta = np.float64(0.5)
f = np.complex128(5.0e-3 * (1.0 + 1.0j))

n_arr = np.arange(1, N + 1, dtype=np.float64)

n_k = k0 * q**n_arr
n_k_sq = n_k**2

# 2. 非線形項の係数

n_c1 = np.zeros(N, dtype=np.float64)
n_c2 = np.zeros(N, dtype=np.float64)
n_c3 = np.zeros(N, dtype=np.float64)

for i in range(N - 2):
    n_c1[i] = n_k[i]

for i in range(1, N - 1):
    n_c2[i] = -beta * n_k[i - 1]

for i in range(2, N):
    n_c3[i] = (beta - 1.0) * n_k[i - 2]

# 3. 非線形項

@njit
def nonlinear_numba(
    u,
    c1,
    c2,
    c3,
    forcing,
    include_forcing,
):
    """
    シェルモデルの非線形項を計算する。

    include_forcing=True:
        時間発展用。第4シェルへの外力を含む。

    include_forcing=False:
        エネルギーフラックス計算用。外力を含まない。
    """

    N_local = u.size

    nl = np.zeros(N_local, dtype=np.complex128)

    # 第1シェル
    nl[0] = 1.0j * (c1[0] * np.conj(u[1]) * np.conj(u[2]))

    # 第2シェル
    nl[1] = 1.0j * (c1[1] * np.conj(u[2]) * np.conj(u[3]) +
                    c2[1] * np.conj(u[0]) * np.conj(u[2]))

    # 中間のシェル
    for i in range(2, N_local - 2):

        nl[i] = 1.0j * (c1[i] * np.conj(u[i + 1]) * np.conj(u[i + 2]) +
                        c2[i] * np.conj(u[i - 1]) * np.conj(u[i + 1]) +
                        c3[i] * np.conj(u[i - 2]) * np.conj(u[i - 1]))

    # 最後から2番目のシェル
    nl[N_local - 2] = 1.0j * (c2[N_local - 2] * np.conj(u[N_local - 3]) * np.conj(u[N_local - 1]) +
                              c3[N_local - 2] * np.conj(u[N_local - 4]) * np.conj(u[N_local - 3]))

    # 最後のシェル
    nl[N_local - 1] = 1.0j * (c3[N_local - 1] * np.conj(u[N_local - 3]) * np.conj(u[N_local - 2]))

    # 第4シェルへの外力
    if include_forcing:
        nl[3] += forcing

    return nl

# 4. エネルギーフラックス

@njit
def calculate_flux_numba(
    u,
    c1,
    c2,
    c3,
    forcing,
):
    """
    非線形相互作用によるエネルギー転送率 T_n と
    エネルギーフラックス Pi(k_n) を計算する。

    T_n = Re[u_n^* N_n(u)]

    ここでは

        Pi(k_n) = sum_{m=n+1}^{N} T_m

    と定義する。
    """

    N_local = u.size

    # フラックスには外力を入れない
    nl_pure = nonlinear_numba(
        u,
        c1,
        c2,
        c3,
        forcing,
        False,
    )

    transfer = np.empty(N_local, dtype=np.float64)

    for i in range(N_local):
        transfer[i] = np.real(np.conj(u[i]) * nl_pure[i])

    flux = np.zeros(N_local, dtype=np.float64)

    # 高波数側から累積和を計算
    cumulative = 0.0

    for i in range(N_local - 1, 0, -1):
        cumulative += transfer[i]
        flux[i - 1] = cumulative

    # 最終シェルより高波数側は存在しない
    flux[N_local - 1] = 0.0

    return flux

# 5. 時間積分

@njit
def run_flux_simulation_numba(
    dt,
    total_time,
    avg_start_time,
    sample_interval,
    k,
    k_sq,
    c1,
    c2,
    c3,
    viscosity,
    forcing,
):
    """
    IF-RK4で時間発展させ、平均エネルギースペクトルと
    平均エネルギーフラックスを求める。
    """

    total_steps = int(round(total_time / dt))
    avg_start_step = int(round(avg_start_time / dt))

    N_local = k.size

    # 初期条件
  
    np.random.seed(42)

    u = np.zeros(N_local, dtype=np.complex128)

    for i in range(N_local):

        initial_energy = (k_sq[i] * np.exp(-k_sq[i]))

        phase = np.random.uniform(0.0,2.0 * np.pi,)

        u[i] = (np.sqrt(2.0 * k[i] * initial_energy) * np.exp(1.0j * phase))

    # 積分因子
    
    E_visc = np.exp(-viscosity * k_sq * dt)

    E_visc_half = np.exp(-viscosity * k_sq * dt * 0.5)

    # 時間平均用

    energy_spectrum_sum = np.zeros(N_local,dtype=np.float64,)

    flux_sum = np.zeros(N_local,dtype=np.float64,)

    dissipation_sum = 0.0
    injection_sum = 0.0

    avg_count = 0

    # 時間積分

    for step in range(total_steps):

        # ---------- IF-RK4 ----------

        k1 = nonlinear_numba(u,c1,c2,c3,forcing,True,)

        u_half1 = (u + 0.5 * dt * k1) * E_visc_half

        k2 = nonlinear_numba(u_half1,c1,c2,c3,forcing,True,)

        u_half2 = (u * E_visc_half + 0.5 * dt * k2)

        k3 = nonlinear_numba(u_half2,c1,c2,c3,forcing,True,)

        u_full = (E_visc * u + dt * E_visc_half * k3)

        k4 = nonlinear_numba(u_full,c1,c2,c3,forcing,True,)

        u = (u * E_visc + (dt / 6.0) * (k1 * E_visc + 2.0 * k2 * E_visc_half + 2.0 * k3 * E_visc_half + k4))

        # ---------- 時間平均 ----------

        if (
            step >= avg_start_step
            and
            (step - avg_start_step)
            % sample_interval == 0
        ):

            abs_u_sq = (u.real**2 + u.imag**2)

            # E(k_n) = |u_n|^2 / (2 k_n)
            energy_spectrum_sum += (abs_u_sq / (2.0 * k))

            # エネルギーフラックス
            flux_sum += calculate_flux_numba(u,c1,c2,c3,forcing,)

            # 瞬間散逸率
            # epsilon(t)
            # = nu sum_n k_n^2 |u_n|^2

            dissipation_sum += (viscosity * np.sum(k_sq * abs_u_sq))

            # 瞬間エネルギー注入率
            injection_sum += np.real(forcing * np.conj(u[3]))

            avg_count += 1

    # 平均値
   
    energy_spectrum_avg = (energy_spectrum_sum / avg_count)

    flux_avg = (flux_sum / avg_count)

    dissipation_avg = (dissipation_sum / avg_count)

    injection_avg = (injection_sum / avg_count)

    ratio = (injection_avg / dissipation_avg)

    relative_error = (abs(injection_avg - dissipation_avg) / abs(dissipation_avg))

    # Kolmogorov波数
    k_d = (dissipation_avg / viscosity**3)**0.25

    # 無次元化
    x_normalized = k / k_d
    flux_normalized = (flux_avg / dissipation_avg)

    return (
        energy_spectrum_avg,
        flux_avg,
        injection_avg,
        dissipation_avg,
        ratio,
        relative_error,
        k_d,
        x_normalized,
        flux_normalized,
        avg_count,
    )

# 6. 実行用関数

def run_flux_simulation(
    dt,
    total_time,
    avg_start_time,
    sample_interval=10,
):
    """
    エネルギーフラックス計算を実行する関数。
    """

    total_steps = int(round(total_time / dt))

    avg_start_step = int(round(avg_start_time / dt))

    if dt <= 0:
        raise ValueError("dtは正の値にしてください。")

    if total_time <= 0:
        raise ValueError("total_timeは正の値にしてください。")

    if avg_start_time < 0:
        raise ValueError("avg_start_timeは0以上にしてください。")

    if avg_start_time >= total_time:
        raise ValueError("avg_start_timeはtotal_timeより""小さくしてください。")

    if sample_interval < 1:
        raise ValueError("sample_intervalは1以上の整数にしてください。")

    estimated_count = ((total_steps - avg_start_step - 1) // sample_interval + 1)

    print(f"N = {N}")
    print(f"nu = {nu:.1e}")
    print(f"dt = {dt}")
    print(f"total_time = {total_time}")
    print(f"avg_start_time = {avg_start_time}")
    print(f"総ステップ数 = {total_steps:,}")
    print(f"サンプリング間隔 = "f"{sample_interval} ステップ")
    print(f"平均化点数の予定 = "f"{estimated_count:,}")
    print()
    print("計算中...")

    results = run_flux_simulation_numba(
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
        energy_spectrum_avg,
        flux_avg,
        injection_avg,
        dissipation_avg,
        ratio,
        relative_error,
        k_d,
        x_normalized,
        flux_normalized,
        avg_count,
    ) = results

    print("計算完了！")
    print()
    print("--- 統計定常状態での時間平均 ---")
    print(f"平均エネルギー注入率 <I> "f"= {injection_avg:.10e}")
    print(f"平均エネルギー散逸率 <epsilon> "f"= {dissipation_avg:.10e}")
    print(f"<I> / <epsilon> "f"= {ratio:.10f}")
    print(f"相対誤差 "f"= {relative_error:.6e}")
    print(f"Kolmogorov波数 k_d "f"= {k_d:.10e}")
    print(f"平均化に使用した点数 "f"= {avg_count:,}")

    return {
        "N": N,
        "nu": nu,
        "dt": np.float64(dt),
        "total_time": np.float64(total_time),
        "avg_start_time": np.float64(
            avg_start_time
        ),
        "sample_interval": int(
            sample_interval
        ),
        "energy_spectrum_avg":
            energy_spectrum_avg.copy(),
        "flux_avg":
            flux_avg.copy(),
        "injection_avg":
            injection_avg,
        "epsilon_avg":
            dissipation_avg,
        "ratio":
            ratio,
        "relative_error":
            relative_error,
        "k_d":
            k_d,
        "x_normalized":
            x_normalized.copy(),
        "flux_normalized":
            flux_normalized.copy(),
        "avg_count":
            avg_count,
    }
