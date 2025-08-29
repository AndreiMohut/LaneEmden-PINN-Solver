# ===========================================
# Multi-case PINN + Variational vs Analytical (no numerical solver)
# ODE: x'' + 2ζω0 x' + ω0^2 x = 0,  x(0)=1, x'(0)=0
# Train on 8 ω0 values and evaluate 5 ω0 values (same as user params).
# Saves one PNG with two subplots: TRAIN (left) and EVAL (right),
# each showing Analytical (solid), PINN (dashed), Variational (dotted).
# ===========================================

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- import your StellarNet implementation ---
from pinn_architecture.pinn_architecture import StellarNet

# Repro & device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------
# Problem setup (same as your code)
# -----------------------------
zeta = 0.20            # damping ratio ζ (constant)
omega0_train = np.linspace(0.8, 1.5, 8, dtype=np.float32)                 # 8 training ω0
omega0_test  = np.array([0.85, 0.95, 1.05, 1.15, 1.25], dtype=np.float32)  # 5 evaluation ω0

t_max  = 10.0          # time horizon
Ncol_per_case = 2000   # collocation points per training case
lr     = 1e-3
epochs = 30000

# -----------------------------
# Collocation grid over (τ, ω0) — TRAIN only
# τ = t / t_max in [0,1]
# -----------------------------
num_train = len(omega0_train)
tau_1case = torch.linspace(0.0, 1.0, Ncol_per_case, device=device).view(-1, 1)
tau = tau_1case.repeat(num_train, 1).detach().clone().requires_grad_(True)

omega0_col = torch.tensor(
    np.repeat(omega0_train, Ncol_per_case),
    dtype=torch.float32, device=device
).view(-1, 1)

# -----------------------------
# Model & hard-IC ansatz
# x(τ,ω0) = 1 + τ^2 fθ(τ,ω0)  ⇒ x(0)=1, x'(0)=0
# -----------------------------
model = StellarNet(input_dim=2, hidden_dim=256, num_blocks=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=100
)

def forward_x(tau_tensor, omega0_tensor):
    inp = torch.cat([tau_tensor, omega0_tensor], dim=1)  # (N,2)
    f = model(inp)
    return 1.0 + (tau_tensor**2) * f

# Chain rule scaling: t = τ * t_max
inv_T  = 1.0 / t_max
inv_T2 = inv_T * inv_T

# -----------------------------
# Training loop (physics loss)
# -----------------------------
model.train()
for ep in range(epochs + 1):
    optimizer.zero_grad()

    x_tau = forward_x(tau, omega0_col)

    dx_dtau = torch.autograd.grad(
        outputs=x_tau, inputs=tau,
        grad_outputs=torch.ones_like(x_tau),
        create_graph=True
    )[0]

    d2x_dtau2 = torch.autograd.grad(
        outputs=dx_dtau, inputs=tau,
        grad_outputs=torch.ones_like(dx_dtau),
        create_graph=True
    )[0]

    dx_dt   = inv_T  * dx_dtau
    d2x_dt2 = inv_T2 * d2x_dtau2

    # Physics residual: x'' + 2ζω0 x' + ω0^2 x = 0
    res = d2x_dt2 + 2.0 * zeta * omega0_col * dx_dt + (omega0_col**2) * x_tau
    loss = torch.mean(res**2)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step(loss.item())

    if ep % 50 == 0:
        print(f"[{ep:5d}/{epochs}] loss={loss.item():.6e}")

# -----------------------------
# Analytical solution (closed-form)
# -----------------------------
def analytical_linear_solution(zeta, omega0, t_eval):
    """Closed-form solution of x'' + 2ζω0 x' + ω0^2 x = 0, x(0)=1, x'(0)=0."""
    z = float(zeta); w0 = float(omega0); t = np.asarray(t_eval)
    if z < 1.0:  # underdamped
        wd = w0 * np.sqrt(1.0 - z**2)
        return np.exp(-z*w0*t) * (np.cos(wd*t) + (z*w0/wd)*np.sin(wd*t))
    elif np.isclose(z, 1.0):  # critical
        return np.exp(-w0*t) * (1.0 + w0*t)
    else:  # overdamped
        s1 = -z*w0 + w0*np.sqrt(z**2 - 1.0)
        s2 = -z*w0 - w0*np.sqrt(z**2 - 1.0)
        C1 = -s2/(s1 - s2)
        C2 =  s1/(s1 - s2)
        return C1*np.exp(s1*t) + C2*np.exp(s2*t)

def variational_linear_resonator(
    zeta, omega0, t_eval, t_max,
    M=24, Nq=800, segments=6, use_envelope=True, ridge=0.0
):
    """
    Stable least-squares variational (weighted-residual) in time with multi-elements.
    - Trial on each element [tau_a, tau_b], local s∈[0,1]:
        x(s) = x0 + (h*v0_tau)*s + Σ_{k=0..M} a_k ψ_k(s)
      ψ_k(s) = s^2 cos(kπ s) * E(s),   E(s)=exp(-α*(tau_a + h*s)) if use_envelope else 1,
      α = zeta*omega0*t_max.
    - Choose a_k to minimize ∫ (R)^2 dτ using *discrete* LS:
        minimize || sqrt(w*h) * (R_base + R_k a) ||_2
      solved by column-scaled QR/`lstsq` (optionally with ridge).
    - Propagate (x, x_τ) between elements to enforce C^1 continuity.
    """
    import numpy as np

    # quadrature on [0,1]
    def gl01(N):
        xi, wi = np.polynomial.legendre.leggauss(N)
        s = 0.5*(xi + 1.0); w = 0.5*wi
        return s, w

    # basis and derivatives on local s
    def basis_block(s, kvec, alpha, tau_a, h, use_env):
        s = s[:, None]                               # (Nq,1)
        KPI = np.pi * kvec[None, :]                  # (1,M+1)
        cosk = np.cos(KPI*s); sink = np.sin(KPI*s)

        g    = (s**2)*cosk
        g_s  = 2*s*cosk - (s**2)*KPI*sink
        g_ss = 2*cosk - 4*s*KPI*sink - (s**2)*(KPI**2)*cosk

        if use_env:
            E    = np.exp(-alpha*(tau_a + h*s))
            E_s  = -alpha*h * E
            E_ss = (alpha*h)**2 * E
            psi    = g*E
            psi_s  = g_s*E + g*E_s
            psi_ss = g_ss*E + 2*g_s*E_s + g*E_ss
        else:
            psi, psi_s, psi_ss = g, g_s, g_ss
        return psi, psi_s, psi_ss

    T = float(t_max)
    tau_eval = np.clip(np.asarray(t_eval)/T, 0.0, 1.0)
    seg_edges = np.linspace(0.0, 1.0, segments+1)
    kvec = np.arange(0, M+1, dtype=float)
    s_q, w_q = gl01(Nq)
    alpha = float(zeta*omega0*T)

    # store element parameters
    elems = []

    # initial conditions at tau=0
    x0 = 1.0
    v0_tau = 0.0

    for j in range(segments):
        tau_a = seg_edges[j]; tau_b = seg_edges[j+1]
        h = tau_b - tau_a

        psi, psi_s, psi_ss = basis_block(s_q, kvec, alpha, tau_a, h, use_envelope)

        inv_T  = 1.0/T
        inv_h  = 1.0/h
        dxdt   = inv_T*inv_h
        d2xdt2 = (inv_T**2)*(inv_h**2)

        # residual pieces
        # base part: x_base(s) = x0 + (h*v0_tau)*s  ->  x_t_base = v0_tau/T
        R_base = (2*zeta*omega0)*(v0_tau/T) + (omega0**2)*(x0 + (h*v0_tau)*s_q)     # (Nq,)
        R_k    = d2xdt2*psi_ss + 2*zeta*omega0*dxdt*psi_s + (omega0**2)*psi         # (Nq,M+1)

        # weighted design matrix and rhs: minimize || W^(1/2)(R_base + R_k a) ||_2
        wsqrt = np.sqrt(w_q*h)[:, None]                    # (Nq,1)
        Phi   = wsqrt * R_k                                # (Nq,M+1)
        y     = - (wsqrt[:,0] * R_base)                    # (Nq,)

        # column scaling to unit 2-norm
        col_norm = np.linalg.norm(Phi, axis=0) + 1e-14
        Phi_s = Phi / col_norm

        # optional tiny ridge for safety
        if ridge > 0.0:
            # solve (Phi_s^T Phi_s + λI) a_s = Phi_s^T y
            AtA = Phi_s.T @ Phi_s
            rhs = Phi_s.T @ y
            a_s = np.linalg.solve(AtA + ridge*np.eye(AtA.shape[0]), rhs)
        else:
            a_s, *_ = np.linalg.lstsq(Phi_s, y, rcond=None)

        a = a_s / col_norm

        elems.append({"x0": x0, "v0_tau": v0_tau, "a": a, "h": h, "tau_a": tau_a})

        # propagate state to next element (evaluate at s=1)
        s1 = 1.0
        KPI = np.pi * kvec
        cos1 = np.cos(KPI*s1); sin1 = np.sin(KPI*s1)

        if use_envelope:
            E1   = np.exp(-alpha*(tau_a + h*s1))
            E1_s = -alpha*h*E1
            g1   = (s1**2)*cos1
            g1_s = 2*s1*cos1 - (s1**2)*KPI*sin1
            psi1   = g1*E1
            psi1_s = g1_s*E1 + g1*E1_s
        else:
            psi1   = (s1**2)*cos1
            psi1_s = 2*s1*cos1 - (s1**2)*KPI*sin1

        x_end  = x0 + (h*v0_tau)*s1 + float(psi1 @ a)
        xs_end = (h*v0_tau)        + float(psi1_s @ a)     # d/ds at s=1
        v0_tau = xs_end * (1.0/h)                          # convert to d/dtau
        x0     = x_end

    # evaluate on t_eval
    x_out = np.empty_like(t_eval, dtype=float)
    for j in range(segments):
        tau_a = seg_edges[j]; tau_b = seg_edges[j+1]; h = tau_b - tau_a
        mask = (tau_eval >= tau_a) & (tau_eval <= (tau_b if j==segments-1 else tau_b-1e-12))
        if not np.any(mask): continue
        s = (tau_eval[mask] - tau_a)/h
        s_col = s[:, None]
        KPI = np.pi * kvec[None,:]
        cosk = np.cos(KPI*s_col)

        if use_envelope:
            E = np.exp(-alpha*(tau_a + h*s_col))
            psi_e = (s_col**2)*cosk*E
        else:
            psi_e = (s_col**2)*cosk

        p = elems[j]
        x_out[mask] = p["x0"] + (h*p["v0_tau"])*s + (psi_e @ p["a"]).ravel()
    return x_out

# -----------------------------
# Evaluation grids & helpers
# -----------------------------
t_eval = np.linspace(0.0, t_max, 2001)

def metrics(a, b):
    err = a - b
    mae  = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    maxe = np.max(np.abs(err))
    return mae, rmse, maxe

model.eval()
train_metrics = []
test_metrics  = []
train_results = {}
test_results  = {}

def evaluate_case(omega0_case):
    # Analytical
    x_ana = analytical_linear_solution(zeta, float(omega0_case), t_eval)
    # PINN
    with torch.no_grad():
        tau_eval = torch.tensor(t_eval / t_max, dtype=torch.float32, device=device).view(-1, 1)
        omega_eval = torch.full_like(tau_eval, float(omega0_case))
        x_pinn = forward_x(tau_eval, omega_eval).cpu().numpy().flatten()
    # Variational
    x_var = variational_linear_resonator(zeta, float(omega0_case), t_eval, t_max, M=10, Nq=400)
    return x_ana, x_pinn, x_var

# Train-set cases
for w0 in omega0_train:
    x_ana, x_pinn, x_var = evaluate_case(w0)
    mae_p, rmse_p, maxe_p = metrics(x_pinn, x_ana)
    mae_v, rmse_v, maxe_v = metrics(x_var,  x_ana)
    train_metrics.append((w0, mae_p, rmse_p, maxe_p, mae_v, rmse_v, maxe_v))
    train_results[float(w0)] = (x_ana, x_pinn, x_var)
    print(f"[TRAIN] ω0={w0:.2f} | PINN MAE={mae_p:.2e} RMSE={rmse_p:.2e} | "
          f"VAR MAE={mae_v:.2e} RMSE={rmse_v:.2e}")

# Eval (test) cases
for w0 in omega0_test:
    x_ana, x_pinn, x_var = evaluate_case(w0)
    mae_p, rmse_p, maxe_p = metrics(x_pinn, x_ana)
    mae_v, rmse_v, maxe_v = metrics(x_var,  x_ana)
    test_metrics.append((w0, mae_p, rmse_p, maxe_p, mae_v, rmse_v, maxe_v))
    test_results[float(w0)] = (x_ana, x_pinn, x_var)
    print(f"[EVAL ] ω0={w0:.2f} | PINN MAE={mae_p:.2e} RMSE={rmse_p:.2e} | "
          f"VAR MAE={mae_v:.2e} RMSE={rmse_v:.2e}")

# -----------------------------
# Single PNG with 2 subplots (Analytical: solid, PINN: circles, Variational: dotted)
# -----------------------------
all_w0 = np.concatenate([omega0_train, omega0_test])
w0_min, w0_max = float(all_w0.min()), float(all_w0.max())
def color_for_w0(w0):
    u = (float(w0) - w0_min) / (w0_max - w0_min + 1e-12)
    return plt.cm.viridis(u)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
ax_left, ax_right = axes

# place a marker roughly every ~60 samples
markevery = max(1, len(t_eval) // 60)

# TRAIN (left)
for w0 in omega0_train:
    c = color_for_w0(w0)
    x_ana, x_pinn, x_var = train_results[float(w0)]
    ax_left.plot(t_eval, x_ana, '-',  color=c, alpha=0.95, linewidth=1.8, zorder=1)
    ax_left.plot(t_eval, x_var,  ':',  color=c, alpha=0.95, linewidth=1.6, zorder=2)
    ax_left.plot(
        t_eval, x_pinn,
        linestyle='None', marker='o', color=c,
        markersize=4, markerfacecolor='none', markeredgewidth=1.0,
        markevery=markevery, zorder=3
    )
ax_left.set_title(f"Training (8 ω₀): {np.array2string(omega0_train, precision=2, separator=', ')}")
ax_left.set_xlabel("t"); ax_left.set_ylabel("x(t)")
ax_left.grid(True)

# EVAL (right)
for w0 in omega0_test:
    c = color_for_w0(w0)
    x_ana, x_pinn, x_var = test_results[float(w0)]
    ax_right.plot(t_eval, x_ana, '-',  color=c, alpha=0.95, linewidth=1.8, zorder=1)
    ax_right.plot(t_eval, x_var,  ':',  color=c, alpha=0.95, linewidth=1.6, zorder=2)
    ax_right.plot(
        t_eval, x_pinn,
        linestyle='None', marker='o', color=c,
        markersize=4, markerfacecolor='none', markeredgewidth=1.0,
        markevery=markevery, zorder=3
    )
ax_right.set_title(f"Evaluation (5 ω₀): {np.array2string(omega0_test, precision=2, separator=', ')}")
ax_right.set_xlabel("t")
ax_right.grid(True)

# Legend proxies
legend_lines = [
    Line2D([0], [0], color="k", linestyle='-',  linewidth=2, label="Analytical"),
    Line2D([0], [0], marker='o', linestyle='None', color="k",
           markersize=5, markerfacecolor='none', markeredgewidth=1.5, label="PINN"),
    Line2D([0], [0], color="k", linestyle=':',  linewidth=2, label="Variational"),
]
ax_left.legend(handles=legend_lines, loc='upper right')
ax_right.legend(handles=legend_lines, loc='upper right')

plt.suptitle(f"Linear Damped Resonator  (ζ={zeta})", y=0.99)
plt.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig("linres_train_vs_eval_analytical_pinn_variational.png", dpi=300)
plt.show()
