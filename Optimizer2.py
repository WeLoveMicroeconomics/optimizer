# app.py  —  Compact Matplotlib-only constrained optimization visualizer
import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sympy import symbols, sympify, lambdify, Eq, solve
from scipy.optimize import minimize

st.set_page_config(layout="centered")
st.title("Constrained Optimization (Compact • Matplotlib)")

# ========= Inputs =========
colL, colR = st.columns([2, 1])

with colL:
    f_expr = st.text_input("f(x, y):", value="min(x, y)")
    constraint = st.text_input("Constraint (e.g. x + y <= 10 or x + y == 10):", value="x + y <= 10")
    c1, c2 = st.columns(2)
    with c1:
        xmin = st.number_input("x min", value=0.0)
        ymin = st.number_input("y min", value=0.0)
    with c2:
        xmax = st.number_input("x max", value=10.0)
        ymax = st.number_input("y max", value=10.0)

with colR:
    n_levels = st.slider("Contour levels", 5, 20, 9, 1)
    show_feasible = st.toggle("Show feasible fill", value=True)
    equal_axes = st.toggle("Equal axis scale", value=True)
    base_fs = st.slider("Label font size", 8, 18, 10)
    render_width_px = st.slider("Rendered width (px)", 240, 720, 360, 10)

# ========= Validation =========
if xmin >= xmax or ymin >= ymax:
    st.error("Ensure x/y min < max.")
    st.stop()

# ========= Build functions =========
x_sym, y_sym = symbols("x y")
try:
    f_sym = sympify(f_expr, evaluate=False)
except Exception as e:
    st.error(f"Could not parse f(x,y): {e}")
    st.stop()

if f_expr.strip().lower().startswith("min"):
    def f_func(xv, yv): return np.minimum(xv, yv)
else:
    try:
        f_func = lambdify((x_sym, y_sym), f_sym, "numpy")
    except Exception as e:
        st.error(f"Could not lambdify f(x,y): {e}")
        st.stop()

# Constraint parse
if "<=" in constraint:
    lhs_str, rhs_str = constraint.split("<="); operator = "<="
elif "==" in constraint:
    lhs_str, rhs_str = constraint.split("=="); operator = "=="
else:
    st.error("Only '<=' and '==' constraints are supported.")
    st.stop()

try:
    lhs_sym = sympify(lhs_str.strip(), evaluate=False)
    rhs_val = float(rhs_str.strip())
    lhs_func = lambdify((x_sym, y_sym), lhs_sym, "numpy")
except Exception as e:
    st.error(f"Could not parse constraint: {e}")
    st.stop()

# ========= Optimization (maximize f by minimizing -f) =========
bounds = [(xmin, xmax), (ymin, ymax)]
if operator == "<=":
    cons = {'type': 'ineq', 'fun': lambda v: rhs_val - lhs_func(v[0], v[1])}
else:
    cons = {'type': 'eq', 'fun': lambda v: lhs_func(v[0], v[1]) - rhs_val}

def objective(v):
    try:
        val = f_func(v[0], v[1])
        if isinstance(val, np.ndarray): val = val.item()
        if np.isnan(val): return 1e10
        return -val
    except Exception:
        return 1e10

x0 = [(xmin + xmax) / 2, (ymin + ymax) / 2]
res = minimize(objective, x0, bounds=bounds, constraints=cons)
if not res.success:
    st.error(f"Optimization failed: {res.message}")
    st.stop()

max_point = res.x
max_val = -res.fun

# Numerical gradient of f at optimum
eps = 1e-6
def num_grad(func, v):
    g = np.zeros_like(v)
    for i in range(len(v)):
        v1, v2 = v.copy(), v.copy()
        v1[i] += eps; v2[i] -= eps
        g[i] = (func(v1) - func(v2)) / (2 * eps)
    return g
grad = -num_grad(objective, max_point)  # gradient of f

# ========= Grid & fields =========
x_vals = np.linspace(xmin, xmax, 400)
y_vals = np.linspace(ymin, ymax, 400)
X, Y = np.meshgrid(x_vals, y_vals)

Z = f_func(X, Y)
if isinstance(Z, (int, float)): Z = Z * np.ones_like(X, dtype=float)

zmin = float(np.nanmin(Z)); zmax = float(np.nanmax(Z))
if not np.isfinite(zmin) or not np.isfinite(zmax):
    st.error("Function produced non-finite values on the grid.")
    st.stop()
if np.isclose(zmin, zmax):
    zmin -= 1.0; zmax += 1.0

lhs_grid = lhs_func(X, Y)
feasible_mask = (lhs_grid <= rhs_val + 1e-9) if operator == "<=" else (np.abs(lhs_grid - rhs_val) < 1e-6)

# ========= Plot (Matplotlib, compact) =========
fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=140)

# Feasible region (subtle)
if show_feasible:
    feas = feasible_mask.astype(float)
    feas[~feasible_mask] = np.nan
    ax.contourf(X, Y, feas, levels=[0.5, 1.5], colors=["#1e90ff33"], alpha=0.35)

# f(x,y) contours (lines only)
ax.contour(X, Y, Z, levels=n_levels, colors="black", linestyles="-", linewidths=1.6)

# Level curve at optimum
ax.contour(X, Y, Z, levels=[max_val], colors="crimson", linestyles="-", linewidths=2.2)

# Optimum marker & gradient arrow
ax.plot([max_point[0]], [max_point[1]], marker="o", color="crimson", ms=5, mec="white", mew=0.8)
arrow_scale = 0.6
ax.annotate("∇f",
            xy=(max_point[0], max_point[1]),
            xytext=(max_point[0] + arrow_scale*grad[0], max_point[1] + arrow_scale*grad[1]),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", lw=1.6, color="crimson"),
            color="crimson", fontsize=base_fs)

# Constraint boundary
try:
    constraint_eq = Eq(lhs_sym, rhs_val)
    bstyle = "-"
    y_solutions = solve(constraint_eq, y_sym)
    if y_solutions:
        y_func = lambdify(x_sym, y_solutions[0], "numpy")
        x_line = np.linspace(xmin, xmax, 800)
        y_line = y_func(x_line)
        mask = np.isfinite(y_line) & (y_line >= ymin) & (y_line <= ymax)
        ax.plot(x_line[mask], y_line[mask], bstyle, color="royalblue", lw=2.0)
    else:
        x_solutions = solve(constraint_eq, x_sym)
        if x_solutions:
            x_func = lambdify(y_sym, x_solutions[0], "numpy")
            y_line = np.linspace(ymin, ymax, 800)
            x_line = x_func(y_line)
            mask = np.isfinite(x_line) & (x_line >= xmin) & (x_line <= xmax)
            ax.plot(x_line[mask], y_line[mask], bstyle, color="royalblue", lw=2.0)
except Exception as e:
    st.warning(f"Constraint boundary could not be drawn: {e}")

# Axes formatting (compact)
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
ax.set_xlabel("x", fontsize=base_fs, labelpad=2)
ax.set_ylabel("y", fontsize=base_fs, labelpad=2)
ax.tick_params(axis="both", labelsize=base_fs-2, length=3, width=0.8)
ax.grid(True, color="0.9", linewidth=0.8)
if equal_axes: ax.set_aspect("equal", adjustable="box")
fig.tight_layout(pad=0.2)

# ===== Render to exact pixel width =====
buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
buf.seek(0)
st.image(buf, width=render_width_px, caption="Constrained optimization (compact)", output_format="PNG")

# ========= Explanation =========
st.markdown(
    f"**Function**: `{f_expr}` &nbsp; | &nbsp; **Constraint**: `{constraint}`  \n"
    f"**Optimum**: ({max_point[0]:.6f}, {max_point[1]:.6f}) &nbsp; | &nbsp; "
    f"**f(optimum)**: {max_val:.6f} &nbsp; | &nbsp; "
    f"**∇f**: (df/dx={grad[0]:.3g}, df/dy={grad[1]:.3g})"
)
