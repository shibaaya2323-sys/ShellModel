import numpy as np
from numba import njit


# ============================================================
# 1. シェルモデルのパラメータを作成する関数
# ============================================================

def make_shell_parameters(N):
    """
    指定されたシェル数 N に対して、波数、非線形項の係数、
    および外力を作成する。
    """
    # シェル数を整数に変換
    N = int(N)
    q = np.float64(2.0)
    k0 = np.float64(2.0**(-4))
    beta = np.float64(0.5)
    f = np.complex128(5.0e-3 * (1.0 + 1.0j))

    # シェル番号 n=1,2,...,N
    n_arr = np.arange(1,N + 1,dtype=np.float64)

    # 各シェルの波数 k_n=k_0 q^n
    n_k = k0 * q**n_arr

    # 粘性項の計算に用いる k_n^2
    n_k_sq = n_k**2

    # 非線形項に現れる3種類の係数
    n_c1 = np.zeros(N, dtype=np.float64)
    n_c2 = np.zeros(N, dtype=np.float64)
    n_c3 = np.zeros(N, dtype=np.float64)

     # 第1項の係数 c_n^(1)=k_n
    for i in range(N - 2):
        n_c1[i] = n_k[i]

    # 第2項の係数 c_n^(2)=-beta k_{n-1}
    for i in range(1, N - 1):
        n_c2[i] = -beta * n_k[i - 1]

    # 第3項の係数 c_n^(3)=(beta-1)k_{n-2}
    for i in range(2, N):
        n_c3[i] = (beta - 1.0) * n_k[i - 2]

    return (
        n_k,
        n_k_sq,
        n_c1,
        n_c2,
        n_c3,
        f,
    )

# ============================================================
# 2. 非線形項を計算する関数
# ============================================================

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
    各シェルの複素変数 u_n から、
    シェルモデルの非線形項 N_n(u) を計算する。

    include_forcing=True:
        時間発展用。第4シェルへの外力を加える。

    include_forcing=False:
        エネルギーフラックス計算用。
        非線形相互作用だけを取り出すため、外力を加えない。
    """
    # シェル数
    N_local = n_u.size
    
    # 非線形項を格納する複素配列
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
    # 第2項と第3項だけを計算する
    n_nl[N_local - 2] = 1.0j * (n_c2[N_local - 2] * np.conj(n_u[N_local - 3]) * np.conj(n_u[N_local - 1]) +
                                n_c3[N_local - 2] * np.conj(n_u[N_local - 4]) * np.conj(n_u[N_local - 3]))

    # 最後のシェルでは、第3項だけが存在する。
    n_nl[N_local - 1] = 1.0j * (n_c3[N_local - 1] * np.conj(n_u[N_local - 3]) * np.conj(n_u[N_local - 2]))

    # 第4シェルへの外力
    if include_forcing:
        n_nl[3] += f

    return n_nl

# ============================================================
# 3. 瞬間エネルギーフラックスを計算する関数
# ============================================================

@njit
def calculate_flux_numba(
    n_u,
    n_c1,
    n_c2,
    n_c3,
    f,
):
    """
    非線形相互作用による各シェルのエネルギー転送率 T_n と、
    エネルギーフラックス Pi(k_n) を計算する。

    T_n = Re[u_n^* N_n(u)]

    Pi(k_n) = sum_{m=n}^{N} T_m
    """
    
    # シェル数
    N_local = n_u.size

    # フラックス計算では外力を除き、
    # 非線形相互作用だけを取り出す
    n_nl_pure = nonlinear_numba(
        n_u,
        n_c1,
        n_c2,
        n_c3,
        f,
        False,
    )
    
    # 各シェルのエネルギー転送率 T_n
    n_transfer = np.empty(N_local, dtype=np.float64)
    
    # T_n=Re[u_n^* N_n(u)] を計算
    for i in range(N_local):
        n_transfer[i] = np.real(np.conj(n_u[i]) * n_nl_pure[i])
        
    # 各波数境界のフラックス
    n_flux = np.zeros(N_local, dtype=np.float64)

    # 高波数側から転送率を累積する
    cumulative = 0.0

    for i in range(N_local - 1, -1, -1):
        cumulative += n_transfer[i]
        n_flux[i] = cumulative

        # 全シェルの非線形転送率の総和
        # 非線形項がエネルギーを保存する場合、ほぼ0になる
        transfer_total = np.sum(n_transfer)

    return n_flux, transfer_total

# ============================================================
# 4. 時間積分と時間平均を行う関数
# ============================================================

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
    積分因子を組み込んだ4次Runge--Kutta法で時間発展させる。

    統計定常状態に達した後のデータから、

    ・平均エネルギーフラックス
    ・平均エネルギー注入率
    ・平均エネルギー散逸率
    ・非線形エネルギー保存の誤差

    を求める。
    """

    # 総時間を時間刻みで割り、
    # 時間積分の総ステップ数を求める

    total_steps = int(round(total_time / dt))
    avg_start_step = int(round(avg_start_time / dt))

    # シェル数
    N_local = n_k.size

    # 初期条件
    # 計算結果を再現できるように乱数シードを固定する
    np.random.seed(42)

    # 各シェルの複素変数 u_n を格納する配列
    n_u = np.zeros(N_local, dtype=np.complex128)

    for i in range(N_local):
        
        # 初期エネルギースペクトル E(k_n)=k_n^2 exp(-k_n^2)
        initial_energy = (n_k_sq[i] * np.exp(-n_k_sq[i]))
        
        # 各シェルの初期位相を0から2piまでの一様乱数で与える
        phase = np.random.uniform(0.0,2.0 * np.pi,)
        
        # E(k_n)=|u_n|^2/(2k_n) の関係からu_nを決定する
        n_u[i] = (np.sqrt(2.0 * n_k[i] * initial_energy) * np.exp(1.0j * phase))

    # 積分因子
    # 粘性項 -nu k_n^2 u_n を積分因子で処理する。
    # 1ステップ分と半ステップ分の減衰因子を計算する。
    n_E_visc = np.exp(-nu * n_k_sq * dt)
    n_E_visc_half = np.exp(-nu * n_k_sq * dt * 0.5)

    # 時間平均用の変数
    # 各波数境界における
    # エネルギーフラックスの累積値
    n_flux_sum = np.zeros(N_local,dtype=np.float64)

    dissipation_sum = 0.0
    injection_sum = 0.0
    transfer_total_sum = 0.0
    transfer_total_abs_max = 0.0

    avg_count = 0

    # 時間積分
    # 積分因子を組み込んだ4次のRunge--Kutta法により、
    # シェル変数 u_n を時間発展させる。

    for step in range(total_steps):

        n_k1 = nonlinear_numba(n_u,n_c1,n_c2,n_c3,f,True)
        n_u_half1 = (n_u + 0.5 * dt * n_k1) * n_E_visc_half

        n_k2 = nonlinear_numba(n_u_half1,n_c1,n_c2,n_c3,f,True)
        n_u_half2 = (n_u * n_E_visc_half + 0.5 * dt * n_k2)

        n_k3 = nonlinear_numba(n_u_half2,n_c1,n_c2,n_c3,f,True)
        n_u_full = (n_E_visc * n_u + dt * n_E_visc_half * n_k3)

        n_k4 = nonlinear_numba(n_u_full,n_c1,n_c2,n_c3,f,True)

        n_u = (n_u * n_E_visc + (dt / 6.0) * (n_k1 * n_E_visc + 2.0 * n_k2 * n_E_visc_half + 2.0 * n_k3 * n_E_visc_half + n_k4))

       
        # avg_start_time以降で、sample_intervalステップごとに
        # エネルギーフラックスなどをサンプリングする。
        if (step >= avg_start_step and (step - avg_start_step) % sample_interval == 0):
            
            # 各シェルの |u_n|^2 を計算する。
            n_abs_u_sq = (n_u.real**2 + n_u.imag**2)

            # 現在の u_n から、
            # 各波数境界の瞬間フラックスを計算する
            n_flux_instant, transfer_total = calculate_flux_numba(n_u,n_c1,n_c2,n_c3,f)
            
            # 瞬間フラックスを累積する
            n_flux_sum += n_flux_instant
            
            # 全シェルの転送率の和を累積する
            transfer_total_sum += transfer_total
            
            # 非線形エネルギー保存誤差の
            # 絶対値の最大値を記録する
            if abs(transfer_total) > transfer_total_abs_max:
                transfer_total_abs_max = abs(transfer_total)

           
            # エネルギー散逸率epsilon=nu sum k_n^2 |u_n|^2を累積する。
            dissipation_sum += (nu * np.sum(n_k_sq * n_abs_u_sq))

            # 第4シェルにおけるエネルギー注入率I=Re(f u_4^*) を累積する。
            injection_sum += np.real(f * np.conj(n_u[3]))
            avg_count += 1

    # 時間平均値の計算
    # 平均エネルギーフラックス
    n_flux_avg = (n_flux_sum / avg_count)

    # 平均エネルギー散逸率
    dissipation_avg = (dissipation_sum / avg_count)

    # 平均エネルギー注入率
    injection_avg = (injection_sum / avg_count)

    # 全シェルの非線形転送率の和の平均
    transfer_total_avg = transfer_total_sum / avg_count

    # 注入率と散逸率の比
    # 統計定常状態では1に近いことが期待される
    ratio = (injection_avg / dissipation_avg)

    # 注入率と散逸率の相対誤差
    relative_error = (abs(injection_avg - dissipation_avg) / abs(dissipation_avg))

    # Kolmogorov波数
    k_d = (dissipation_avg / nu**3)**0.25

    # 波数とフラックスの無次元化
    # 横軸 k_n/k_d
    n_x_normalized = n_k / k_d

    # 縦軸 Pi(k_n)/epsilon
    n_flux_normalized = (n_flux_avg / dissipation_avg)

    # 計算結果を呼び出し元へ返す
    return (
        n_flux_avg,
        injection_avg,
        dissipation_avg,
        ratio,
        relative_error,
        k_d,
        n_x_normalized,
        n_flux_normalized,
        avg_count,
        transfer_total_avg,
        transfer_total_abs_max,
    )

# ============================================================
# 5. 実行用関数
# ============================================================

def run_flux_simulation(
    N,
    nu,
    dt,
    total_time,
    avg_start_time,
    sample_interval=10,
):
    """
    エネルギーフラックス計算を実行する。

    入力値を確認した後にシェルモデルのパラメータを作成し、
    Numbaで高速化した時間積分関数を呼び出す。

    計算結果は画面に表示するとともに、
    辞書形式で呼び出し元へ返す。
    """
    N = int(N)
    nu = np.float64(nu)
    dt = np.float64(dt)
    total_time = np.float64(total_time)
    avg_start_time = np.float64(avg_start_time)
    sample_interval = int(sample_interval)
    
    # 非線形項の端点処理には少なくとも4個のシェルが必要
    if int(N) < 4:
        raise ValueError("Nは4以上にしてください。")
        
    # 動粘性係数は正でなければならない
    if nu <= 0:
        raise ValueError("nuは正の値にしてください。")
        
    # 時間刻みは正でなければならない
    if dt <= 0:
        raise ValueError("dtは正の値にしてください。")
        
    # 総計算時間は正でなければならない
    if total_time <= 0:
        raise ValueError("total_timeは正の値にしてください。")
        
    # 平均開始時刻は0以上でなければならない
    if avg_start_time < 0:
        raise ValueError("avg_start_timeは0以上にしてください。")
        
    # 平均開始時刻は総計算時間より前でなければならない
    if avg_start_time >= total_time:
        raise ValueError("avg_start_timeはtotal_timeより小さくしてください。")
    # サンプリング間隔は1以上の整数でなければならない
    if sample_interval < 1:
        raise ValueError("sample_intervalは1以上の整数にしてください。")

    # --------------------------------------------------------
    # シェルモデルのパラメータを作成する
    # --------------------------------------------------------

    (
        n_k,
        n_k_sq,
        n_c1,
        n_c2,
        n_c3,
        f,
    ) = make_shell_parameters(N=int(N))

    # --------------------------------------------------------
    # 計算回数と平均化点数を求める
    # --------------------------------------------------------
    
    # 総ステップ数
    total_steps = int(round(total_time / dt))
    
    # 時間平均を開始するステップ番号
    avg_start_step = int(round(avg_start_time / dt))
    
    # 平均化に使用する予定のデータ点数
    estimated_count = ((total_steps - avg_start_step - 1) // sample_interval + 1)
    
    # 平均化できる点が存在するか確認する
    if estimated_count <= 0:
        raise ValueError("平均化に使用できる点がありません。")
        
    # --------------------------------------------------------
    # 計算条件を画面に表示する
    # --------------------------------------------------------
    
    print(f"N = {int(N)}")
    print(f"nu = {nu:.1e}")
    print(f"dt = {dt}")
    print(f"total_time = {total_time}")
    print(f"avg_start_time = {avg_start_time}")
    print(f"総ステップ数 = {total_steps:,}")
    print(f"サンプリング間隔 = "f"{sample_interval} ステップ")
    print(f"平均化点数の予定 = "f"{estimated_count:,}")
    print()
    print("計算中...")

    # --------------------------------------------------------
    # Numbaで高速化した時間積分を実行する
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 戻り値を各変数に分ける
    # --------------------------------------------------------

    (
        n_flux_avg,
        injection_avg,
        dissipation_avg,
        ratio,
        relative_error,
        k_d,
        n_x_normalized,
        n_flux_normalized,
        avg_count,
        transfer_total_avg,
        transfer_total_abs_max,
    ) = results

    # --------------------------------------------------------
    # 計算結果を画面に表示する
    # --------------------------------------------------------

    print("計算完了！")
    print()
    print("--- 統計定常状態での時間平均 ---")
    print(f"平均エネルギー注入率 <I> = {injection_avg:.10e}")
    print(f"平均エネルギー散逸率 <epsilon> = {dissipation_avg:.10e}")
    print(f"<I> / <epsilon> = {ratio:.10f}")
    print(f"相対誤差 = {relative_error:.6e}")
    print(f"Kolmogorov波数 k_d = {k_d:.10e}")
    print(f"平均化に使用した点数 = {avg_count:,}")
    print()
    print("--- 非線形エネルギー保存の確認 ---")
    print(f"<sum T_n> = {transfer_total_avg:.10e}")
    print(f"max |sum T_n| = {transfer_total_abs_max:.10e}")

    # --------------------------------------------------------
    # 計算条件と結果を辞書形式で返す
    # --------------------------------------------------------
    # この辞書は、グラフ作成や
    # 異なる粘性係数の結果を比較するときに使用する

    return {
    "N": int(N),
    "nu": nu,
    "dt": dt,
    "total_time": total_time,
    "avg_start_time": avg_start_time,
    "sample_interval": sample_interval,
    "flux_avg": n_flux_avg.copy(),
    "injection_avg": injection_avg,
    "epsilon_avg": dissipation_avg,
    "ratio": ratio,
    "relative_error": relative_error,
    "k_d": k_d,
    "x_normalized": n_x_normalized.copy(),
    "flux_normalized": n_flux_normalized.copy(),
    "avg_count": avg_count,
    "transfer_total_avg": transfer_total_avg,
    "transfer_total_abs_max": transfer_total_abs_max,
}
