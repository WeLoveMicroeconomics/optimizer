import streamlit as st
import numpy as np
import plotly.graph_objs as go
from sympy import symbols, sympify, lambdify, Eq, solve
from scipy.optimize import minimize

# =========================
# App + Layout
# =========================
st.set_page_config(layout="wide")
st.title("Constrained Optimization Visualizer (Clean Contours)")

# --- Inputs (left) & Style (right) ---
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
    st.markdown("### Plot Style")
    colors = st.text_input("Colors (comma-separated)", value="black,royalblue,crimson,teal")
    linestyles = st.text_input("Line styles (comma-separated)", value="-,--,:")
    n_levels = st.slider("Number of f-contour levels", min_value=5, max_value=20, value=9, step=1)
    show_feasible = st.toggle("Show feasible fill", value=True)
    show_grid = st.toggle("Show grid", value=True)
    equal_axes = st.toggle("Equal axis scale", value=True)
    label_font_size = st.slider("Label font size", 10, 22, 14)
    show_diag = st.toggle("Show diagnostics", value=False)

update = st.button("Update Plot")

warning_placeholder = st.empty()
plot_placeholder = st.empty()
diag_placeholder = st.empty()
explanation_placeholder = st.empty()

# =========================
# Helper: line style to dash
# =========================
def mpl_ls_to_plotly_dash(s):
    s = s.strip()
    return {"-": "solid", "--": "dash", "-.": "dashdot", ":": "dot"}.get(s, "solid")

# =========================
# Main
# =========================
if update:
    if xmin >= xmax or ymin >= ymax:
        warning_placeholder.error("Ensure that min values are less than max values.")
    else:
        try:
            # --- Symbols & function ---
            x_sym, y_sym = symbols("x y")
            f_sym = sympify(f_expr, evaluate=False)

            if f_expr.strip().lower().startswith("min"):
                def f_func(x_val, y_val):
                    return np.minimum(x_val, y_val)
            else:
                f_func = lambdify((x_sym, y_sym), f_sym, "numpy")

            # --- Constraint parse ---
            if "<=" in constraint:
                lhs_str, rhs_str = constraint.split("<=")
                operator = "<="
            elif "==" in constraint:
                lhs_str, rhs_str = constraint.split("==")
                operator = "=="
            else:
                warning_placeholder.error("Only '<=' and '==' constraints are supported.")
                st.stop()

            lhs_sym = sympify(lhs_str.strip(), evaluate=False)
            rhs_val = float(rhs_str.strip())
            lhs_func = lambdify((x_sym, y_sym), lhs_sym, "numpy")

            # --- SciPy constraint ---
            if operator == "<=":
                cons = {'type': 'ineq', 'fun': lambda v: rhs_val - lhs_func(v[0], v[1])}
            else:
                cons = {'type': 'eq', 'fun': lambda v: lhs_func(v[0], v[1]) - rhs_val}

            bounds = [(xmin, xmax), (ymin, ymax)]

            def objective(v):
                try:
                    val = f_func(v[0], v[1])
                    if isinstance(val, np.ndarray):
                        val = val.item()
                    if np.isnan(val):
                        return 1e10
                    # SciPy minimizes; negate to maximize
                    return -val
                except Exception:
                    return 1e10

            x0 = [(xmin + xmax) / 2, (ymin + ymax) / 2]
            res = minimize(objective, x0, bounds=bounds, constraints=cons)

            if not res.success:
                warning_placeholder.error(f"Optimization failed: {res.message}")
                st.stop()

            max_point = res.x
            max_val = -res.fun

            # --- Numerical gradient (fallback) ---
            eps = 1e-6
            def num_grad(func, v):
                grad = np.zeros_like(v)
                for i in range(len(v)):
                    v_eps1 = v.copy(); v_eps2 = v.copy()
                    v_eps1[i] += eps; v_eps2[i] -= eps
                    f1 = func(v_eps1)
                    f2 = func(v_eps2)
                    grad[i] = (f1 - f2) / (2 * eps)
                return grad

            grad = -num_grad(objective, max_point)  # ∇f

            # --- Grid + fields ---
            x_vals = np.linspace(xmin, xmax, 400)
            y_vals = np.linspace(ymin, ymax, 400)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = f_func(X, Y)
            if isinstance(Z, (int, float)):
                Z = Z * np.ones_like(X, dtype=float)

            lhs_grid = lhs_func(X, Y)
            if operator == "<=":
                feasible_mask = lhs_grid <= rhs_val + 1e-9
            else:
                feasible_mask = np.abs(lhs_grid - rhs_val) < 1e-6

            # --- Contour levels (robust form) ---
            zmin = float(np.nanmin(Z))
            zmax = float(np.nanmax(Z))
            if not np.isfinite(zmin) or not np.isfinite(zmax):
                warning_placeholder.error("Function produced non-finite values on the grid.")
                st.stop()
            if np.isclose(zmin, zmax):
                zmin -= 1.0
                zmax += 1.0

            # Pick first color & linestyle from the inputs
            line_color = (colors.split(",")[0] or "black").strip()
            first_ls = (linestyles.split(",")[0] or "-").strip()
            line_dash = mpl_ls_to_plotly_dash(first_ls)

            # =========================
            # Build Figure
            # =========================
            fig = go.Figure()

            # Feasible region (optional, subtle)
            if show_feasible:
                feasible_float = feasible_mask.astype(float)
                feasible_float[~feasible_mask] = np.nan
                fig.add_trace(go.Contour(
                    z=feasible_float,
                    x=x_vals,
                    y=y_vals,
                    showscale=False,
                    contours=dict(coloring='heatmap', showlines=False),
                    name='Feasible region',
                    hoverinfo="skip",
                    opacity=0.18,
                    colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(30,144,255,1)']]
                ))

            # f(x,y) contours: single trace with ncontours (line-only)
            fig.add_trace(go.Contour(
                z=Z,
                x=x_vals,
                y=y_vals,
                showscale=False,
                contours=dict(coloring='lines', showlabels=False, ncontours=int(n_levels)),
                line=dict(color=line_color, width=2, dash=line_dash),
                name="f contours",
                hoverinfo="skip"
            ))

            # Level curve at maximum (highlight)
            fig.add_trace(go.Contour(
                z=Z, x=x_vals, y=y_vals,
                contours=dict(start=max_val, end=max_val, size=1e-9,
                              coloring='lines', showlabels=False),
                showscale=False,
                line=dict(color='crimson', width=3),
                name='Level @ optimum',
                hoverinfo="skip"
            ))

            # Maximum point
            fig.add_trace(go.Scatter(
                x=[max_point[0]], y=[max_point[1]],
                mode='markers+text',
                marker=dict(color='crimson', size=10, line=dict(color='white', width=1)),
                text=['optimum'],
                textposition='top center',
                name='Optimum',
                hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>f(x,y)=%{customdata:.4f}<extra></extra>",
                customdata=np.array([[max_val]])
            ))

            # Constraint boundary: explicit line (solid for ==, dashed for <=)
            try:
                constraint_eq = Eq(lhs_sym, rhs_val)
                boundary_color = "royalblue"
                boundary_dash = "solid" if operator == "==" else "dash"

                # Try solve for y(x)
                y_solutions = solve(constraint_eq, y_sym)
                if y_solutions:
                    y_func = lambdify(x_sym, y_solutions[0], 'numpy')
                    x_line = np.linspace(xmin, xmax, 800)
                    y_line = y_func(x_line)
                    mask = np.isfinite(y_line) & (y_line >= ymin) & (y_line <= ymax)
                    x_line = x_line[mask]; y_line = y_line[mask]

                    if x_line.size > 1:
                        fig.add_trace(go.Scatter(
                            x=x_line, y=y_line, mode='lines',
                            line=dict(color=boundary_color, width=3, dash=boundary_dash),
                            name='Constraint boundary',
                            hovertemplate=f"{lhs_str.strip()} = {rhs_val:g}<extra></extra>"
                        ))
                else:
                    # Try solve for x(y)
                    x_solutions = solve(constraint_eq, x_sym)
                    if x_solutions:
                        x_func = lambdify(y_sym, x_solutions[0], 'numpy')
                        y_line = np.linspace(ymin, ymax, 800)
                        x_line = x_func(y_line)
                        mask = np.isfinite(x_line) & (x_line >= xmin) & (x_line <= xmax)
                        x_line = x_line[mask]; y_line = y_line[mask]

                        if y_line.size > 1:
                            fig.add_trace(go.Scatter(
                                x=x_line, y=y_line, mode='lines',
                                line=dict(color=boundary_color, width=3, dash=boundary_dash),
                                name='Constraint boundary',
                                hovertemplate=f"{lhs_str.strip()} = {rhs_val:g}<extra></extra>"
                            ))
                    else:
                        warning_placeholder.warning("Constraint boundary cannot be plotted.")
            except Exception as e:
                warning_placeholder.warning(f"Failed to plot constraint boundary: {e}")

            # --- Layout (clean, like pilot) ---
            axis_common = dict(
                tickfont=dict(size=label_font_size, color='black'),
                showline=True, linewidth=1, linecolor="rgba(0,0,0,0.25)",
                mirror=False, ticks="outside", ticklen=6, tickwidth=1,
                zeroline=False
            )

            fig.update_layout(
                template="plotly_white",
                height=700,  # make sure it's visible
                title=dict(text="Constrained Optimization – Clean Contours", x=0.0,
                           font=dict(size=label_font_size+2)),
                xaxis=dict(
                    title=dict(text="x", font=dict(size=label_font_size, color='black')),
                    range=[xmin, xmax],
                    showgrid=show_grid,
                    gridcolor="rgba(0,0,0,0.08)" if show_grid else None,
                    **axis_common
                ),
                yaxis=dict(
                    title=dict(text="y", font=dict(size=label_font_size, color='black')),
                    range=[ymin, ymax],
                    showgrid=show_grid,
                    gridcolor="rgba(0,0,0,0.08)" if show_grid else None,
                    **axis_common
                ),
                font=dict(family="Arial", size=label_font_size, color='black'),
                legend=dict(orientation='h', yanchor="bottom", y=1.02, xanchor="left", x=0.0,
                            bgcolor="rgba(255,255,255,0.7)", bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
                margin=dict(l=40, r=20, t=60, b=40),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )

            if equal_axes:
                fig.update_yaxes(scaleanchor="x", scaleratio=1)

            # MAIN PLOT FIRST
            plot_placeholder.plotly_chart(fig, use_container_width=True)

            # Optional diagnostics
            if show_diag:
                with diag_placeholder.container():
                    st.write(f"[diag] traces: {len(fig.data)}")
                    _zfinite = int(np.isfinite(Z).sum())
                    st.write(f"[diag] Z finite count: {_zfinite} / {Z.size}")
                    st.plotly_chart(go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1])]), use_container_width=True)

            explanation_placeholder.markdown(f"""
**Function**: `{f_expr}`  
**Constraint**: `{constraint}`  
**Optimum**: ({max_point[0]:.6f}, {max_point[1]:.6f})  
**f(optimum)**: {max_val:.6f}  
**∇f at optimum**: (df/dx = {grad[0]:.3g}, df/dy = {grad[1]:.3g})
            """)

        except Exception as e:
            warning_placeholder.error(f"Error: {e}")
