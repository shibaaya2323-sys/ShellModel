
import numpy as np


def n_nonlinear_n(
    n_u,
    N,
    n_c1,
    n_c2,
    n_c3,
    f,
):
    """
    シェルモデルの非線形項と外力を計算する関数。
    """

    # 戻り値をcomplex128で用意
    n_nl = np.zeros(N, dtype=np.complex128)

    I_comp = np.complex128(1.0j)

    # i = 0
    n_nl[0] = I_comp * (
        n_c1[0]
        * np.conj(n_u[1])
        * np.conj(n_u[2])
    )

    # i = 1
    n_nl[1] = I_comp * (
        n_c1[1]
        * np.conj(n_u[2])
        * np.conj(n_u[3])
        +
        n_c2[1]
        * np.conj(n_u[0])
        * np.conj(n_u[2])
    )

    # i = 2 ～ N-3
    for i in range(2, N - 2):
        n_nl[i] = I_comp * (
            n_c1[i]
            * np.conj(n_u[i + 1])
            * np.conj(n_u[i + 2])
            +
            n_c2[i]
            * np.conj(n_u[i - 1])
            * np.conj(n_u[i + 1])
            +
            n_c3[i]
            * np.conj(n_u[i - 2])
            * np.conj(n_u[i - 1])
        )

    # i = N-2
    n_nl[N - 2] = I_comp * (
        n_c2[N - 2]
        * np.conj(n_u[N - 3])
        * np.conj(n_u[N - 1])
        +
        n_c3[N - 2]
        * np.conj(n_u[N - 4])
        * np.conj(n_u[N - 3])
    )

    # i = N-1
    n_nl[N - 1] = I_comp * (
        n_c3[N - 1]
        * np.conj(n_u[N - 3])
        * np.conj(n_u[N - 2])
    )

    # 第4シェルに外力を追加
    n_nl[3] += f

    return n_nl



