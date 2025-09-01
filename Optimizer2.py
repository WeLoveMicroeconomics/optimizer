import streamlit as st
import numpy as np
import plotly.graph_objs as go
from sympy import symbols, sympify, lambdify, Eq, solve
from scipy.optimize import minimize

st.set_page_config(layout="wide")
st.title("Constrained Optimization Visualizer (Compact Plot)")

# -------- Inputs --------
col_in, col_style = st.columns([2, 1])

with col_in:
    f_expr = st.text_input("f(x, y):", "min(x, y)")
    constraint = st.text_input("Constraint (e.g. x + y <= 10):", "x + y <= 10")
    c1, c2 = st.columns(2)
    with c1:
        xmin = st.number_input("x min:", value=0.0)
        ymin = st.number_input("y min:", value=0.0)
    with c2:
        xmax = st.number_input("x max:", value=10.0)
        ymax = st.number_input("y max:", value=10.0)

with col_style:
    st.markdown("### Plot Settings")
    backend = st.selectbox("Backend", ["Matplotlib (fallback)", "Plotly (interactive)"], index=0)
    colors = st.text_input("Contour color(s)", value="black")
    linestyles = st.text_input("Line style(s)", value="-")
    n_levels = st.slider("Number of contour levels", 5, 20, 9, 1)
    show_feasible = st.toggle("Show feasible fill", value=True)
    equal_axes = st.toggle("Equal axis scale", value=True)
    label_font_size = st.slider("Label font size", 8, 22, 12)
    show_diag = st.toggle("Show diagnostics", value=False)

update = st.button("Update Plot")

def mpl_ls_to_plotly_dash(s):
    return {"-": "solid", "--": "dash", "-.": "dashdot", ":": "dot"}.get(s.strip(), "solid")

if update:
    if xmin >= xmax or ymin >= ymax:
        st.error("Ensure that min values are less than max values.")
        st.stop()

    try:
        # --- parse f ---
        x_sym, y_sym = symbols("x y")
        f_sym = sympify(f_expr, evaluate=False)
        if f_expr.strip().lower().startswith("min"):
            def f_func(x_val, y_val): return np.minimum(x_val, y_val)
        else:
            f_func = lambdify((x_sym, y_sym), f_sym, "numpy")

        # --- constraint ---
        if "<=" in constraint:
            lhs_str, rhs_str = constraint.split("<="); operator = "<="
        elif "==" in constraint:
            lhs_str, rhs_str = constraint.split("=="); operator = "=="
        else:
            st.error("Only '<=' and '==' constraints are supported.")
            st.stop()

        lhs_sym = sympify(lhs_str.strip(), evaluate=False)
        rhs_val = float(rhs_str.strip())
        lhs_func = lambdify((x_sym, y_sym), lhs_sym, "numpy")

        cons = {'type': 'ineq', 'fun': lambda v: rhs_val - lhs_func(v[0], v[1])} if operator == "<=" \
               else {'type': 'eq', 'fun': lambda v: lhs_func(v[0], v[1]) - rhs_val}
        bounds = [(xmin, xmax), (ymin, ymax)]

        def objective(v):
            try:
                val = f_func(v[0], v[1])
                if isinstance(val, np.ndarray): val = val.item()
                if np.isnan(val): return 1e10
                return -val  # maximize f
            except Exception:
                return 1e10

        x0 = [(xmin + xmax) / 2, (ymin + ymax) / 2]
        res = minimize(objective, x0, bounds=bounds, constraints=cons)
        if not res.success:
            st.error(f"Optimization failed: {res.message}")
            st.stop()

        max_point = res.x
        max_val = -res.fun

        # gradient
        eps = 1e-6
        def num_grad(func, v):
            grad = np.zeros_like(v)
            for i in range(len(v)):
                v1 = v.copy(); v2 = v.copy()
                v1[i] += eps; v2[i] -= eps
                grad[i] = (func(v1) - func(v2)) / (2*eps)
            return grad
        grad = -num_grad(objective, max_point)

        # grid + fields
        x_vals = np.linspace(xmin, xmax, 400)
        y_vals = np.linspace(ymin, ymax, 400)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = f_func(X, Y)
        if isinstance(Z, (int, float)): Z = Z * np.ones_like(X, dtype=float)

        zmin = float(np.nanmin(Z)); zmax = float(np.nanmax(Z))
        if not np.isfinite(zmin) or not np.isfinite(zmax):
            st.error("Function produced non-finite values on the grid.")
            st.stop()
        if np.isclose(zmin, zmax): zmin -= 1.0; zmax += 1.0

        lhs_grid = lhs_func(X, Y)
        feasible_mask = (lhs_grid <= rhs_val + 1e-9) if operator == "<=" else (np.abs(lhs_grid - rhs_val) < 1e-6)

        # =========================
        # Matplotlib backend
        # =========================
        if backend.startswith("Matplotlib"):
            import matplotlib.pyplot as plt

            # compact figure
            fig, ax = plt.subplots(figsize=(3.6, 2.7), dpi=120, constrained_layout=False)

            if show_feasible:
                feas = feasible_mask.astype(float)
                feas[~feasible_mask] = np.nan
                ax.contourf(X, Y, feas, levels=[0.5, 1.5], colors=["#1e90ff33"], alpha=0.3)

            color = (colors.split(",")[0] or "black").strip()
            style = (linestyles.split(",")[0] or "-").strip()
            ax.contour(X, Y, Z, levels=n_levels, colors=color, linestyles=style, linewidths=1.8)
            ax.contour(X, Y, Z, levels=[max_val], colors="crimson", linestyles="-", linewidths=2.4)

            ax.plot([max_point[0]], [max_point[1]], marker="o", color="crimson", ms=6, mec="white", mew=0.8)
            arrow_scale = 0.6
            ax.annotate("∇f",
                        xy=(max_point[0], max_point[1]),
                        xytext=(max_point[0] + arrow_scale*grad[0], max_point[1] + arrow_scale*grad[1]),
                        arrowprops=dict(arrowstyle="->", lw=1.8, color="crimson"),
                        color="crimson", fontsize=label_font_size)

            try:
                constraint_eq = Eq(lhs_sym, rhs_val)
                y_solutions = solve(constraint_eq, y_sym)
                boundary_style = "-" if operator == "==" else "--"
                if y_solutions:
                    y_func = lambdify(x_sym, y_solutions[0], "numpy")
                    x_line = np.linspace(xmin, xmax, 800)
                    y_line = y_func(x_line)
                    mask = np.isfinite(y_line) & (y_line >= ymin) & (y_line <= ymax)
                    ax.plot(x_line[mask], y_line[mask], boundary_style, color="royalblue", lw=2.2)
                else:
                    x_solutions = solve(constraint_eq, x_sym)
                    if x_solutions:
                        x_func = lambdify(y_sym, x_solutions[0], "numpy")
                        y_line = np.linspace(ymin, ymax, 800)
                        x_line = x_func(y_line)
                        mask = np.isfinite(x_line) & (x_line >= xmin) & (x_line <= xmax)
                        ax.plot(x_line[mask], y_line[mask], boundary_style, color="royalblue", lw=2.2)
            except Exception as e:
                st.warning(f"Failed to plot constraint boundary: {e}")

            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            ax.set_xlabel("x", fontsize=label_font_size-2, labelpad=2)
            ax.set_ylabel("y", fontsize=label_font_size-2, labelpad=2)
            ax.set_title("Constrained Optimization – Clean Contours", fontsize=label_font_size, pad=6)
            ax.tick_params(axis="both", labelsize=label_font_size-3, length=3, width=0.8)
            if equal_axes: ax.set_aspect("equal", adjustable="box")

            fig.tight_layout(pad=0.3)
            st.pyplot(fig, clear_figure=True, bbox_inches="tight", pad_inches=0.05)

        # =========================
        # Plotly backend
        # =========================
        else:
            line_color = (colors.split(",")[0] or "black").strip()
            line_dash = mpl_ls_to_plotly_dash((linestyles.split(",")[0] or "-").strip())
            fig = go.Figure()
            fig.add_trace(go.Contour(
                z=Z, x=x_vals, y=y_vals,
                showscale=False, ncontours=int(n_levels),
                contours=dict(coloring='lines', showlabels=False),
                line=dict(color=line_color, width=2, dash=line_dash),
                name="f contours", hoverinfo="skip"
            ))
            st.plotly_chart(fig, use_container_width=False)

        if show_diag:
            st.write(f"[diag] optimum: ({max_point[0]:.4f}, {max_point[1]:.4f}) f={max_val:.4f}")

        st.markdown(
            f"**Function**: `{f_expr}` &nbsp; | &nbsp; **Constraint**: `{constraint}`  \n"
            f"**Optimum**: ({max_point[0]:.6f}, {max_point[1]:.6f}) &nbsp; | &nbsp; "
            f"**f(optimum)**: {max_val:.6f} &nbsp; | &nbsp; "
            f"**∇f**: (df/dx={grad[0]:.3g}, df/dy={grad[1]:.3g})"
        )

    except Exception as e:
        st.error(f"Error: {e}")
