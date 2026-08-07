import numpy as np
from numba import njit


# ============================================================
# 1. パラメータ
# ============================================================
# シェル数、シェル間隔、基準波数、動粘性係数、
# 非線形相互作用の係数、および外力を設定する。

N = 22
q = np.float64(2.0)
k0 = np.float64(2.0**(-4))
nu = np.float64(1.0e-7)
beta = np.float64(0.5)

# 第4シェルに加える複素外力
f = np.complex128(5.0e-3 * (1.0 + 1.0j))

# シェル番号 n=1,2,...,N に対応する配列を作成する
n_arr = np.arange(1, N + 1, dtype=np.float64)

# 各シェルの波数 k_n=k_0 q^n を計算する。
n_k = k0 * q**n_arr

# 粘性項の計算で使用する k_n^2 をあらかじめ計算する。
n_k_sq = n_k**2

# 非線形項に現れる3種類の係数を格納する配列を作成する。
n_c1 = np.zeros(N, dtype=np.float64)
n_c2 = np.zeros(N, dtype=np.float64)
n_c3 = np.zeros(N, dtype=np.float64)

# 第1項の係数 c_n^(1)=k_n を設定する。
# 配列の端では対応するシェルが存在しないため、係数を0のまま残す
for i in range(N - 2):
    n_c1[i] = n_k[i]

# 第2項の係数 c_n^(2)=-beta k_{n-1} を設定する。
for i in range(1, N - 1):
    n_c2[i] = -beta * n_k[i - 1]

# 第3項の係数 c_n^(3)=(beta-1)k_{n-2} を設定する。
for i in range(2, N):
    n_c3[i] = (beta - 1.0) * n_k[i - 2]


# ============================================================
# 2. 非線形項
# ============================================================
# 現在のシェル変数 u_n から、各シェルに作用する
# 非線形相互作用項 N_n(u) を計算する。

@njit
def n_nonlinear_numba(n_u, n_c1, n_c2, n_c3, f):
    
    # シェル数を入力された配列から取得する。
    N_local = n_u.size
    
    # 各シェルの非線形項を格納する複素配列を作成する。
    n_nl = np.zeros(N_local, dtype=np.complex128)
    
    # 第1シェルでは、n+1およびn+2のシェルとの相互作用だけが存在する。
    n_nl[0] = 1.0j * (n_c1[0] * np.conj(n_u[1]) * np.conj(n_u[2]))
    
    # 第2シェルでは、第1項と第2項の相互作用が存在する。
    n_nl[1] = 1.0j * (n_c1[1] * np.conj(n_u[2]) * np.conj(n_u[3]) +
                      n_c2[1] * np.conj(n_u[0]) * np.conj(n_u[2]))
    
    # 内部のシェルでは、3種類すべての非線形相互作用を計算する。
    for i in range(2, N_local - 2):

        n_nl[i] = 1.0j * (n_c1[i] * np.conj(n_u[i + 1]) * np.conj(n_u[i + 2]) +
                          n_c2[i] * np.conj(n_u[i - 1]) * np.conj(n_u[i + 1]) +
                          n_c3[i] * np.conj(n_u[i - 2]) * np.conj(n_u[i - 1]))
        
    # 最後から2番目のシェルでは、n+2のシェルが存在しないため、
    # 第2項と第3項だけを計算する。
    n_nl[N_local - 2] = 1.0j * (n_c2[N_local - 2] * np.conj(n_u[N_local - 3]) * np.conj(n_u[N_local - 1]) +
                                n_c3[N_local - 2] * np.conj(n_u[N_local - 4]) * np.conj(n_u[N_local - 3]))
    
    # 最後のシェルでは、第3項だけが存在する。
    n_nl[N_local - 1] = 1.0j * (n_c3[N_local - 1] * np.conj(n_u[N_local - 3]) * np.conj(n_u[N_local - 2]))

    # 第4シェルへの外力
    n_nl[3] += f

    return n_nl


# ============================================================
# 3. 時間積分
# ============================================================
# シェルモデル方程式を時間積分し、
# 統計定常状態におけるエネルギースペクトル、
# エネルギー注入率およびエネルギー散逸率を計算する。

@njit
def run_simulation_numba(
    dt,
    total_time,
    transient_time,
    sample_interval,
    n_k,
    n_k_sq,
    n_c1,
    n_c2,
    n_c3,
    nu,
    f,
):
    
    # 総時間を時間刻みで割り、時間積分の総ステップ数を求める。
    total_steps = int(round(total_time / dt))

    # 時間平均を開始する時刻をステップ番号に変換する。
    transient_step = int(round(transient_time / dt))
    
    # シェル数を波数配列から取得する。
    N_local = n_k.size

    # 初期条件
    # 計算を再現可能にするため、乱数のシードを固定する。
    np.random.seed(42)
    
    # 各シェルの複素振幅 u_n を格納する配列を作成する。
    n_u = np.zeros(N_local, dtype=np.complex128)

    for i in range(N_local):
        
        # 初期エネルギースペクトル E(k_n)=k_n^2 exp(-k_n^2)
        energy_initial = (n_k_sq[i] * np.exp(-n_k_sq[i]))
        
        # 各シェルの初期位相を0から2piの一様乱数で与える。
        phase = np.random.uniform(0.0,2.0 * np.pi)
        
        # E(k_n)=|u_n|^2/(2k_n) の関係からu_nを決定する。
        n_u[i] = (np.sqrt(2.0 * n_k[i] * energy_initial) * np.exp(1.0j * phase))

    # 積分因子
    # 粘性項 -nu k_n^2 u_n を積分因子で処理する。
    # 1ステップ分と半ステップ分の減衰因子を計算する。
    n_E_visc = np.exp(-nu * n_k_sq * dt)
    n_E_visc_half = np.exp(-nu * n_k_sq * dt * 0.5)

    # 時間平均を計算するための変数
    # エネルギースペクトルの累積値を格納する。
    n_E_kn_sum = np.zeros(N_local,dtype=np.float64)

    epsilon_sum = 0.0
    injection_sum = 0.0
    avg_count = 0

    # 時間積分
    # 積分因子を組み込んだ4次のRunge--Kutta法により、
    # シェル変数 u_n を時間発展させる。
    for step in range(total_steps):

        n_k1 = n_nonlinear_numba(n_u,n_c1,n_c2,n_c3,f)
        n_u_half1 = (n_u + 0.5 * dt * n_k1) * n_E_visc_half

        n_k2 = n_nonlinear_numba(n_u_half1,n_c1,n_c2,n_c3,f)
        n_u_half2 = (n_u * n_E_visc_half + 0.5 * dt * n_k2)

        n_k3 = n_nonlinear_numba(n_u_half2,n_c1,n_c2,n_c3,f)
        n_u_full = (n_E_visc * n_u + dt * n_E_visc_half * n_k3)

        n_k4 = n_nonlinear_numba(n_u_full,n_c1,n_c2,n_c3,f)

        n_u = (n_u * n_E_visc +(dt / 6.0) * (n_k1 * n_E_visc + 2.0 * n_k2 * n_E_visc_half + 2.0 * n_k3 * n_E_visc_half + n_k4))

        # 過渡時間の経過後、sample_intervalステップごとに
        # エネルギースペクトルなどをサンプリングする。
        if (step >= transient_step and (step - transient_step) % sample_interval == 0):
            
            # 各シェルの |u_n|^2 を計算する。
            n_abs_u_sq = (n_u.real**2 + n_u.imag**2)
            
            # E(k_n)=|u_n|^2/(2k_n) を累積する。
            n_E_kn_sum += n_abs_u_sq / (2.0 * n_k)

            # エネルギー散逸率epsilon=nu sum k_n^2 |u_n|^2を累積する。
            epsilon_sum += (nu * np.sum(n_k_sq * n_abs_u_sq))

            # 第4シェルにおけるエネルギー注入率I=Re(f u_4^*) を累積する。
            injection_sum += np.real( f * np.conj(n_u[3]))
            avg_count += 1

    
    # 累積値をサンプリング回数で割り、
    # 統計定常状態での時間平均を求める。
    n_E_kn_avg = (n_E_kn_sum / avg_count)
    epsilon_avg = (epsilon_sum / avg_count)
    injection_avg = (injection_sum / avg_count)

    # 注入率と散逸率の比を計算する。
    # 統計定常状態では、この値が1に近いことが期待される。
    ratio = (injection_avg / epsilon_avg)

    # 注入率と散逸率の相対誤差を計算する。
    relative_error = (abs(injection_avg - epsilon_avg) / abs(epsilon_avg))

    # Kolmogorov波数 k_d=(epsilon/nu^3)^(1/4)
    k_d = (epsilon_avg / nu**3)**0.25

    # 波数をKolmogorov波数で規格化する
    n_x_normalized = n_k / k_d

    # エネルギースペクトルを epsilon^(1/4) nu^(5/4) で規格化する。
    n_y_normalized = (n_E_kn_avg / (epsilon_avg**0.25 * nu**1.25))

    # 計算結果を呼び出し元に返す。
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
# 入力された計算条件で時間積分を実行し、
# 結果を画面に表示するとともに、辞書形式で返す。

def run_simulation(
    dt,
    total_time,
    transient_time,
    sample_interval=10,
):
    
    # 入力値を浮動小数点数に変換する。
    dt = np.float64(dt)
    total_time = np.float64(total_time)
    transient_time = np.float64(transient_time)

    # 平均を取る時間を計算する。
    avg_time = total_time - transient_time

    # 入力値を確認する。
    if dt <= 0.0:
        raise ValueError("dtは正の値にしてください。")

    if total_time <= 0.0:
        raise ValueError("total_timeは正の値にしてください。")

    if transient_time < 0.0:
        raise ValueError("transient_timeは0以上にしてください。")

    if transient_time >= total_time:
        raise ValueError("transient_timeはtotal_timeより小さくしてください。")

    if sample_interval < 1:
        raise ValueError("sample_intervalは1以上の整数にしてください。")

    # 総ステップ数を計算する。
    total_steps = int(round(total_time / dt))

    # 過渡時間に対応するステップ数を計算する。
    transient_steps = int(round(transient_time / dt))

    # 時間平均を取る区間のステップ数を計算する。
    avg_steps = total_steps - transient_steps

    # 計算条件を画面に表示する。
    print("--- 計算条件 ---")
    print(f"時間刻み dt: {dt}")
    print(f"全時間 total_time: {total_time}")
    print(f"過渡時間 transient_time: {transient_time}")
    print(f"平均を取る時間 avg_time: {avg_time}")
    print(f"総ステップ数: {total_steps:,} 回")
    print(f"過渡時間のステップ数: {transient_steps:,} 回")
    print(f"平均区間のステップ数: {avg_steps:,} 回")
    print(f"平均値の取得間隔: {sample_interval} ステップ")
    print()   
    
    # Numbaで高速化した時間積分関数を実行する。
    results = run_simulation_numba(
        dt,
        total_time,
        transient_time,
        int(sample_interval),
        n_k,
        n_k_sq,
        n_c1,
        n_c2,
        n_c3,
        nu,
        f,
    )
    
    # 戻り値を各変数に分ける。
    (
        injection_avg,
        epsilon_avg,
        ratio,
        relative_error,
        n_x_normalized,
        n_y_normalized,
        avg_count,
    ) = results
    
    # 計算結果を画面に表示する。
    print("計算完了！")
    print()
    print("--- 統計定常状態での時間平均 ---")
    print(f"平均を取った時間: {avg_time}")
    print(f"平均エネルギー注入率 <I> "f"= {injection_avg:.10e}")
    print(f"平均エネルギー散逸率 <epsilon> "f"= {epsilon_avg:.10e}")
    print(f"<I> / <epsilon> "f"= {ratio:.10f}")
    print(f"相対誤差 "f"= {relative_error:.6e}")
    print(f"平均化に使用した点数 "f"= {avg_count:,}")
    
    # 計算条件と計算結果を辞書形式にまとめて返す。
    return {
        "dt": dt,
        "total_time": total_time,
        "transient_time": transient_time,
        "avg_time": avg_time,
        "sample_interval": int(sample_interval),
        "injection_avg": injection_avg,
        "epsilon_avg": epsilon_avg,
        "ratio": ratio,
        "relative_error": relative_error,
        "avg_count": avg_count,
        "x_normalized": n_x_normalized.copy(),
        "y_normalized": n_y_normalized.copy(),
    }
