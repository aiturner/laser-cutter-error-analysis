import numpy as np

def fit_affine_2d(g, q):
    """
    Fit q ≈ B @ [gx, gy, 1]^T using least squares.

    Parameters
    ----------
    g : (N, 2) array
        Commanded galvo coordinates: [gx, gy]
    q : (N, 2) array
        Measured corrected coordinates in platform frame: [qx, qy]

    Returns
    -------
    B : (2, 3) array
        Affine matrix:
        [ a  b  tx ]
        [ c  d  ty ]
    residuals : (N, 2) array
        q - q_fit
    """

    g = np.asarray(g, dtype=float)
    q = np.asarray(q, dtype=float)

    if g.ndim != 2 or g.shape[1] != 2:
        raise ValueError("g must have shape (N, 2)")
    if q.shape != g.shape:
        raise ValueError("q must have the same shape as g")

    N = g.shape[0]
    M = np.column_stack([g, np.ones(N)])

    #  M = 
    # [gx1 gy1 1]
    # |gx2 gy2 1|
    # | ........|
    # [gxN gyN 1]

    # qx ​= a*gx + b*gy + tx
    # find a,b, tx via least squares
    params_x, *_ = np.linalg.lstsq(M, q[:, 0], rcond=None)

    # qx ​= c*gx + d*gy + ty
    # find c,d, ty via least squares
    params_y, *_ = np.linalg.lstsq(M, q[:, 1], rcond=None)

    print(params_x , params_y)

    B = np.array([
        params_x, 
        params_y
    ])

    q_fit = M @ B.T #fitted points

    residuals = q - q_fit #not explained by affine fitting

    return B, residuals

def pixels_to_mm(du_dv, k):
    """
    Convert pixel offsets to physical offsets.

    du_dv : (N, 2)
        [du, dv]
    k : float
        mm per pixel
    """
    du_dv = np.asarray(du_dv, dtype=float)
    return k * du_dv


def build_q_from_measurements(p, du_dv, k):
    """
    Identity-rotation assumption:
        R_A = I

    So:
        q = p + (dx, dy)

    Parameters
    ----------
    p : (N, 2)
        Platform reference positions
    du_dv : (N, 2)
        Measured pixel offsets
    k : float
        mm per pixel
    """
    p = np.asarray(p, dtype=float)
    delta_xy = pixels_to_mm(du_dv, k)
    q = p + delta_xy
    return q

def calibrate_galvo(g, p, du_dv, k):
    " "
    q = build_q_from_measurements(p, du_dv, k)
    B, residuals = fit_affine_2d(g, q)
    return {
        "B": B,
        "q": q,
        "residuals": residuals,
    }


def predict_q(g, B):
    """
    Predict corrected position from galvo command using fitted affine matrix.
    """
    g = np.asarray(g, dtype=float)
    if g.ndim == 1:
        g = g.reshape(1, 2)

    M = np.column_stack([g, np.ones(g.shape[0])])  # (N, 3)
    return M @ B.T

if __name__ == "__main__":
    # Example data
    g = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    p = np.array([
        [10.0, 10.0],
        [11.0, 10.0],
        [10.0, 11.0],
        [11.0, 11.0],
    ])

    du_dv = np.array([
        [2.0, -1.0],
        [2.2, -0.8],
        [1.8, -1.1],
        [2.1, -0.9],
    ])

    k = 0.01  # mm per pixel

    result = calibrate_galvo(g, p, du_dv, k)

    B = result["B"]
    residuals = result["residuals"]

    print("Affine matrix B:")
    print(B)
    print("\nResiduals:")
    print(residuals)