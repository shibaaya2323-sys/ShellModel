import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# 1. パラメータ

def make_shell_parameters(N, nu):

    N = int(N)
    nu = np.float64(nu)

    if N < 4:
        raise ValueError("Nは4以上にしてください。")

    if nu <= 0:
        raise ValueError("nuは正の値にしてください。")

    q = np.float64(2.0)
    k0 = np.float64(2.0**(-4))
    beta = np.float64(0.5)
    f = np.complex128(5.0e-3 * (1.0 + 1.0j))

    n_arr = np.arange(1,N + 1,dtype=np.float64)

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

    return (
        n_k,
        n_k_sq,
        n_c1,
        n_c2,
        n_c3,
        nu,
        f,
    )

# 3. 非線形項

@njit
def nonlinear_numba(
    n_u,
    n_c1,
    n_c2,
    n_c3,
    f,
    include_forcing,
):
    """
    シェルモデルの非線形項を計算する。

    include_forcing=True:
        時間発展用。第4シェルへの外力を含む。

    include_forcing=False:
        エネルギーフラックス計算用。外力を含まない。
    """

    N_local = n_u.size

    n_nl = np.zeros(N_local, dtype=np.complex128)

    # 第1シェル
    n_nl[0] = 1.0j * (n_c1[0] * np.conj(n_u[1]) * np.conj(n_u[2]))

    # 第2シェル
    n_nl[1] = 1.0j * (n_c1[1] * np.conj(n_u[2]) * np.conj(n_u[3]) +
                      n_c2[1] * np.conj(n_u[0]) * np.conj(n_u[2]))

    # 中間のシェル
    for i in range(2, N_local - 2):

        n_nl[i] = 1.0j * (n_c1[i] * np.conj(n_u[i + 1]) * np.conj(n_u[i + 2]) +
                          n_c2[i] * np.conj(n_u[i - 1]) * np.conj(n_u[i + 1]) +
                          n_c3[i] * np.conj(n_u[i - 2]) * np.conj(n_u[i - 1]))

    # 最後から2番目のシェル
    n_nl[N_local - 2] = 1.0j * (n_c2[N_local - 2] * np.conj(n_u[N_local - 3]) * np.conj(n_u[N_local - 1]) +
                                n_c3[N_local - 2] * np.conj(n_u[N_local - 4]) * np.conj(n_u[N_local - 3]))

    # 最後のシェル
    n_nl[N_local - 1] = 1.0j * (n_c3[N_local - 1] * np.conj(n_u[N_local - 3]) * np.conj(n_u[N_local - 2]))

    # 第4シェルへの外力
    if include_forcing:
        n_nl[3] += f

    return n_nl

# 4. エネルギーフラックス

@njit
def calculate_flux_numba(
    n_u,
    n_c1,
    n_c2,
    n_c3,
    f,
):
    """
    非線形相互作用によるエネルギー転送率 T_n と
    エネルギーフラックス Pi(k_n) を計算する。

    T_n = Re[u_n^* N_n(u)]

    ここでは

        Pi(k_n) = sum_{m=n+1}^{N} T_m

    と定義する。
    """

    N_local = n_u.size

    # フラックスには外力を入れない
    n_nl_pure = nonlinear_numba(
        n_u,
        n_c1,
        n_c2,
        n_c3,
        f,
        False,
    )

    n_transfer = np.empty(N_local, dtype=np.float64)

    for i in range(N_local):
        n_transfer[i] = np.real(np.conj(n_u[i]) * n_nl_pure[i])

    n_flux = np.zeros(N_local, dtype=np.float64)

    # 高波数側から累積和を計算
    cumulative = 0.0

    for i in range(N_local - 1, 0, -1):
        cumulative += n_transfer[i]
        n_flux[i - 1] = cumulative

    # 最終シェルより高波数側は存在しない
    n_flux[N_local - 1] = 0.0

    return n_flux

# 5. 時間積分

@njit
def run_flux_simulation_numba(
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
    """
    IF-RK4で時間発展させ、平均エネルギースペクトルと
    平均エネルギーフラックスを求める。
    """

    total_steps = int(round(total_time / dt))
    avg_start_step = int(round(avg_start_time / dt))

    N_local = n_k.size

    # 初期条件
  
    np.random.seed(42)

    n_u = np.zeros(N_local, dtype=np.complex128)

    for i in range(N_local):

        initial_energy = (n_k_sq[i] * np.exp(-n_k_sq[i]))

        phase = np.random.uniform(0.0,2.0 * np.pi,)

        n_u[i] = (np.sqrt(2.0 * n_k[i] * initial_energy) * np.exp(1.0j * phase))

    # 積分因子
    
    n_E_visc = np.exp(-nu * n_k_sq * dt)

    n_E_visc_half = np.exp(-nu * n_k_sq * dt * 0.5)

    # 時間平均用

    n_energy_spectrum_sum = np.zeros(N_local,dtype=np.float64)

    n_flux_sum = np.zeros(N_local,dtype=np.float64)

    dissipation_sum = 0.0
    injection_sum = 0.0

    avg_count = 0

    # 時間積分

    for step in range(total_steps):

        # ---------- IF-RK4 ----------

        n_k1 = nonlinear_numba(n_u,n_c1,n_c2,n_c3,f,True,)

        n_u_half1 = (n_u + 0.5 * dt * n_k1) * n_E_visc_half

        n_k2 = nonlinear_numba(n_u_half1,n_c1,n_c2,n_c3,f,True,)

        n_u_half2 = (n_u * n_E_visc_half + 0.5 * dt * n_k2)

        n_k3 = nonlinear_numba(n_u_half2,n_c1,n_c2,n_c3,f,True,)

        n_u_full = (n_E_visc * n_u + dt * n_E_visc_half * n_k3)

        n_k4 = nonlinear_numba(n_u_full,n_c1,n_c2,n_c3,f,True,)

        n_u = (n_u * n_E_visc + (dt / 6.0) * (n_k1 * n_E_visc + 2.0 * n_k2 * n_E_visc_half + 2.0 * n_k3 * n_E_visc_half + n_k4))

        # ---------- 時間平均 ----------

        if (
            step >= avg_start_step
            and
            (step - avg_start_step)
            % sample_interval == 0
        ):

            n_abs_u_sq = (n_u.real**2 + n_u.imag**2)

            # E(k_n) = |u_n|^2 / (2 k_n)
            n_energy_spectrum_sum += (n_abs_u_sq / (2.0 * n_k))

            # エネルギーフラックス
            n_flux_sum += calculate_flux_numba(n_u,n_c1,n_c2,n_c3,f,)

            # 瞬間散逸率
            # epsilon(t)
            # = nu sum_n k_n^2 |u_n|^2

            dissipation_sum += (nu * np.sum(n_k_sq * n_abs_u_sq))

            # 瞬間エネルギー注入率
            injection_sum += np.real(f * np.conj(n_u[3]))

            avg_count += 1

    # 平均値
   
    n_energy_spectrum_avg = (n_energy_spectrum_sum / avg_count)

    n_flux_avg = (n_flux_sum / avg_count)

    dissipation_avg = (dissipation_sum / avg_count)

    injection_avg = (injection_sum / avg_count)

    ratio = (injection_avg / dissipation_avg)

    relative_error = (abs(injection_avg - dissipation_avg) / abs(dissipation_avg))

    # Kolmogorov波数
    k_d = (dissipation_avg / nu**3)**0.25

    # 無次元化
    n_x_normalized = n_k / k_d
    n_flux_normalized = (n_flux_avg / dissipation_avg)

    return (
        n_energy_spectrum_avg,
        n_flux_avg,
        injection_avg,
        dissipation_avg,
        ratio,
        relative_error,
        k_d,
        n_x_normalized,
        n_flux_normalized,
        avg_count,
    )

# 6. 実行用関数

def run_flux_simulation(
    N,
    nu,
    dt,
    total_time,
    avg_start_time,
    sample_interval=10,
):
    """
    エネルギーフラックス計算を実行する関数。
    """
    N = int(N)
    nu = np.float64(nu)
    dt = np.float64(dt)
    total_time = np.float64(total_time)
    avg_start_time = np.float64(avg_start_time)
    sample_interval = int(sample_interval)

    if N < 4:
        raise ValueError("Nは4以上にしてください。")

    if nu <= 0:
        raise ValueError("nuは正の値にしてください。")

    if dt <= 0:
        raise ValueError("dtは正の値にしてください。")

    if total_time <= 0:
        raise ValueError("total_timeは正の値にしてください。")

    if avg_start_time < 0:
        raise ValueError("avg_start_timeは0以上にしてください。")

    if avg_start_time >= total_time:
        raise ValueError("avg_start_timeはtotal_timeより小さくしてください。")

    if sample_interval < 1:
        raise ValueError("sample_intervalは1以上の整数にしてください。")

    (
        n_k,
        n_k_sq,
        n_c1,
        n_c2,
        n_c3,
        nu,
        f,
    ) = make_shell_parameters(
        N,
        nu,
    )

    total_steps = int(
        round(total_time / dt)
    )

    avg_start_step = int(
        round(avg_start_time / dt)
    )

    estimated_count = (
        (
            total_steps
            - avg_start_step
            - 1
        )
        // sample_interval
        + 1
    )

    if estimated_count <= 0:
        raise ValueError(
            "平均化に使用できる点がありません。"
        )

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
    )

    (
        n_energy_spectrum_avg,
        n_flux_avg,
        injection_avg,
        dissipation_avg,
        ratio,
        relative_error,
        k_d,
        n_x_normalized,
        n_flux_normalized,
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
            n_energy_spectrum_avg.copy(),
        "flux_avg":
            n_flux_avg.copy(),
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
            n_x_normalized.copy(),
        "flux_normalized":
            n_flux_normalized.copy(),
        "avg_count":
            avg_count,
    }
