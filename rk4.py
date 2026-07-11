
import numpy as np

from nonlinear import n_nonlinear_n


def modified_rk4_step(
    n_u,
    dt,
    n_E_visc,
    n_E_visc_half,
    N,
    n_c1,
    n_c2,
    n_c3,
    f,
):
    """
    積分因子法を組み込んだ改良型RK4で、
    n_uを1ステップだけ進める。
    """

    # --- 第1段 ---
    n_k1 = n_nonlinear_n(
        n_u,
        N,
        n_c1,
        n_c2,
        n_c3,
        f,
    )

    # --- 第2段 ---
    n_u_half1 = (
        n_u + 0.5 * dt * n_k1
    ) * n_E_visc_half

    n_k2 = n_nonlinear_n(
        n_u_half1,
        N,
        n_c1,
        n_c2,
        n_c3,
        f,
    )

    # --- 第3段 ---
    n_u_half2 = (
        n_u * n_E_visc_half
        + 0.5 * dt * n_k2
    )

    n_k3 = n_nonlinear_n(
        n_u_half2,
        N,
        n_c1,
        n_c2,
        n_c3,
        f,
    )

    # --- 第4段 ---
    n_u_full = (
        n_E_visc * n_u
        + dt * n_E_visc_half * n_k3
    )

    n_k4 = n_nonlinear_n(
        n_u_full,
        N,
        n_c1,
        n_c2,
        n_c3,
        f,
    )

    # --- 次の時刻のn_u ---
    n_u_next = (
        n_u * n_E_visc
        + (dt / 6.0)
        * (
            n_k1 * n_E_visc
            + 2.0 * n_k2 * n_E_visc_half
            + 2.0 * n_k3 * n_E_visc_half
            + n_k4
        )
    )

    return np.asarray(
        n_u_next,
        dtype=np.complex128,
    )
