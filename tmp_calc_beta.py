import numpy as np, json, os, sys

# Load robot preset (v14)
preset_path = r'C:\Users\user\OneDrive - 충북과학고등학교\원드라이브 동기화 폴더\코딩 작업 폴더\slackline_valancing\v14_mujoco\preset_robot.json'
with open(preset_path) as f:
    cfg = json.load(f)
# Use values from preset (m1,m2,l1,l2, etc.) but model uses its own physical params defined later.
# We'll use the same physical constants as in v15 transfer_matrix_analysis (R, h, M_total, I_inv)
R = 1.0
h = 0.85
M_total = 75.0
I_inv = 20.0
g = 9.81
omega = np.sqrt(g / (R + h))
lam = np.sqrt(M_total * g * h / I_inv)
# Folding torque parameters (same as v15 script)
tau_fold = 100.0
I_delta = 5.0
a_f = tau_fold / I_delta
c_foot = 0.280
f_phi = (c_foot / R) * a_f
f_beta = (c_foot / h) * a_f
beta0 = np.radians(5)
# Choose a representative T_f and T_w (use the values that gave best r in v15 sweep)
T_quarter = np.pi / (2 * omega)
T_f = 0.10
T_w = 0.15 * T_quarter

def make_A(om, la, T):
    A = np.zeros((4,4))
    c, s = np.cos(om*T), np.sin(om*T)
    A[0,0], A[0,1] = c, s/om
    A[1,0], A[1,1] = -om*s, c
    ch, sh = np.cosh(la*T), np.sinh(la*T)
    A[2,2], A[2,3] = ch, sh/la
    A[3,2], A[3,3] = la*sh, ch
    return A

def make_b_fold(om, la, T_f, fp, fb):
    C = np.cos(om * T_f)
    S = np.sin(om * T_f)
    Ch = np.cosh(la * T_f)
    Sh = np.sinh(la * T_f)
    return np.array([
        fp/om**2 * (1 - C),
        fp/om * S,
        -fb/la**2 * (Ch - 1),
        -fb/la * Sh
    ])

def one_cycle(T_f, T_w):
    A_F = make_A(omega, lam, T_f)
    A_W = make_A(omega, lam, T_w)
    A_E = A_F
    b_F = make_b_fold(omega, lam, T_f, f_phi, f_beta)
    b_E = -b_F
    M = A_E @ A_W @ A_F
    d = A_E @ A_W @ b_F + b_E
    return M, d

M, d = one_cycle(T_f, T_w)
# eigenvalues of beta submatrix (rows 2,3 cols 2,3)
M_beta = M[2:4, 2:4]
vals = np.linalg.eigvals(M_beta)
print('omega =', omega)
print('lam =', lam)
print('T_f =', T_f, 'T_w =', T_w)
print('M_beta eigenvalues =', vals)
print('spectral radius =', max(abs(vals)))
# state after one cycle
v0 = np.array([0.0, 0.0, beta0, 0.0])
vf = M @ v0 + d
beta_f = vf[2]
print('beta_f (deg) =', np.degrees(beta_f))
print('r =', abs(beta_f / beta0))
