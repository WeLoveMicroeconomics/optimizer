import streamlit as st
import numpy as np
import plotly.graph_objs as go
from sympy import symbols, sympify, lambdify, Eq, solve
from scipy.optimize import minimize

st.set_page_config(layout="wide")
st.title("Constrained Optimization Visualizer (Debug Rendering)")

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
    st.markdown("### Plot Style / Debug")
    render_mode = st.selectbox(
        "Render mode",
        ["Heatmap + lines (robust)", "Contours only", "DEBUG: Heatmap only"],  # <- debug mode
        index=0
    )
    colors = st.text_input("Contour color(s)", value="black")
    linestyles = st.text_input("Line style(s)", value="-")
    n_levels = st.slider("Number of contour levels", 5, 20, 9, 1)
    show_feasible = st.toggle("Show feasible fill", value=True)
    show_grid = st.toggle("Show grid", value=True)
    equal_axes = st.toggle("Equal axis scale", value=True)
    label_font_size = st.slider("Label font size", 10, 22, 14)
    show_diag = st.toggle("Show diagnostics", value=False)

update = st.button("Update Plot")

# -------- Helpers --------
def mpl_ls_to_plotly_dash(s):
    return {"-": "solid", "--": "dash", "-.": "dashdot", ":": "dot"}.get(s.strip(), "solid")

# -------- Main --------
if update:
    if xmin >= xmax or ymin >= ymax:
        st.error("Ensure that min values are less than max values.")
    else:
        try:
            # parse f
            x_sym, y_sym = symbols("x y")
            f_sym = sympify(f_expr, evaluate=False)
            if f_expr.strip().lower().startswith("min"):
                def f_func(x_val, y_val): return np.minimum(x_val, y_val)
            else:
                f_func = lambdify((x_sym, y_sym), f_sym, "numpy")

            # constraint
            if "<=" in constraint:
                lhs_str, rhs_str = constraint.split("<="); operator = "<="
            elif "==" in constraint:
                lhs_str, rhs_str = constraint.split("=="); operator = "=="
            else:
                st.error("Only '<=' and '==' constraints are supported."); st.stop()

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
                st.error(f"Optimization failed: {res.message}"); st.stop()

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

            # grid
            x_vals = np.linspace(xmin, xmax, 400)
            y_vals = np.linspace(ymin, ymax, 400)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = f_func(X, Y)
            if isinstance(Z, (int, float)): Z = Z * np.ones_like(X, dtype=float)

            # feasible mask
            lhs_grid = lhs_func(X, Y)
            feasible_mask = (lhs_grid <= rhs_val + 1e-9) if operator == "<=" else (np.abs(lhs_grid - rhs_val) < 1e-6)

            # z sanity
            zmin = float(np.nanmin(Z)); zmax = float(np.nanmax(Z))
            if not np.isfinite(zmin) or not np.isfinite(zmax):
                st.error("Function produced non-finite values on the grid."); st.stop()
            if np.isclose(zmin, zmax): zmin -= 1.0; zmax += 1.0

            # style
            line_color = (colors.split(",")[0] or "black").strip()
            line_dash = mpl_ls_to_plotly_dash((linestyles.split(",")[0] or "-").strip())

            # -------- build fig (NO placeholders) --------
            fig = go.Figure()

            # Feasible region
            if show_feasible:
                feasible_float = feasible_mask.astype(float)
                feasible_float[~feasible_mask] = np.nan
                fig.add_trace(go.Contour(
                    z=feasible_float, x=x_vals, y=y_vals,
                    showscale=False,
                    contours=dict(coloring='heatmap', showlines=False),
                    name='Feasible region', hoverinfo="skip",
                    opacity=0.18,
                    colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(30,144,255,1)']]
                ))

            if render_mode == "DEBUG: Heatmap only":
                fig.add_trace(go.Heatmap(
                    z=Z, x=x_vals, y=y_vals, colorscale="Greys",
                    opacity=0.6, showscale=False, hoverinfo="skip"
                ))
            elif render_mode == "Heatmap + lines (robust)":
                fig.add_trace(go.Heatmap(
                    z=Z, x=x_vals, y=y_vals, colorscale="Greys",
                    opacity=0.25, showscale=False, hoverinfo="skip"
                ))
                fig.add_trace(go.Contour(
                    z=Z, x=x_vals, y=y_vals,
                    showscale=False,
                    ncontours=int(n_levels),
                    contours=dict(coloring='lines', showlabels=False),
                    line=dict(color=line_color, width=2, dash=line_dash),
                    name="f contours", hoverinfo="skip"
                ))
            else:  # "Contours only"
                fig.add_trace(go.Contour(
                    z=Z, x=x_vals, y=y_vals,
                    showscale=False,
                    ncontours=int(n_levels),
                    contours=dict(coloring='lines', showlabels=False),
                    line=dict(color=line_color, width=2, dash=line_dash),
                    name="f contours", hoverinfo="skip"
                ))

            # level @ optimum
            fig.add_trace(go.Contour(
                z=Z, x=x_vals, y=y_vals,
                contours=dict(start=max_val, end=max_val, size=1e-9,
                              coloring='lines', showlabels=False),
                showscale=False, line=dict(color='crimson', width=3),
                name='Level @ optimum', hoverinfo="skip"
            ))

            # optimum point
            fig.add_trace(go.Scatter(
                x=[max_point[0]], y=[max_point[1]],
                mode='markers+text',
                marker=dict(color='crimson', size=10, line=dict(color='white', width=1)),
                text=['optimum'], textposition='top center',
                name='Optimum',
                hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>f(x,y)=%{customdata:.4f}<extra></extra>",
                customdata=np.array([[max_val]])
            ))

            # constraint boundary
            try:
                constraint_eq = Eq(lhs_sym, rhs_val)
                boundary_dash = "solid" if operator == "==" else "dash"
                boundary_color = "royalblue"
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
            except Exception as e:
                st.warning(f"Failed to plot constraint boundary: {e}")

            # layout
            axis_common = dict(
                tickfont=dict(size=label_font_size, color='black'),
                showline=True, linewidth=1, linecolor="rgba(0,0,0,0.25)",
                mirror=False, ticks="outside", ticklen=6, tickwidth=1, zeroline=False
            )
            fig.update_layout(
                template="plotly_white",
                height=720,
                title=dict(text="Constrained Optimization – Debug View", x=0.0,
                           font=dict(size=label_font_size+2)),
                xaxis=dict(title=dict(text="x", font=dict(size=label_font_size, color='black')),
                           range=[xmin, xmax], showgrid=show_grid,
                           gridcolor="rgba(0,0,0,0.08)" if show_grid else None, **axis_common),
                yaxis=dict(title=dict(text="y", font=dict(size=label_font_size, color='black')),
                           range=[ymin, ymax], showgrid=show_grid,
                           gridcolor="rgba(0,0,0,0.08)" if show_grid else None, **axis_common),
                font=dict(family="Arial", size=label_font_size, color='black'),
                legend=dict(orientation='h', yanchor="bottom", y=1.02, xanchor="left", x=0.0,
                            bgcolor="rgba(255,255,255,0.7)", bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
                margin=dict(l=40, r=20, t=60, b=40),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            if equal_axes:
                fig.update_yaxes(scaleanchor="x", scaleratio=1)

            # ---- render (no placeholders) ----
            st.plotly_chart(fig, use_container_width=True)

            if show_diag:
                st.write(f"[diag] traces: {len(fig.data)}")
                st.write(f"[diag] Z finite count: {int(np.isfinite(Z).sum())} / {Z.size}")
                st.plotly_chart(go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1])]), use_container_width=True)

            st.markdown(f"**Function**: `{f_expr}` &nbsp; | &nbsp; **Constraint**: `{constraint}`  \n"
                        f"**Optimum**: ({max_point[0]:.6f}, {max_point[1]:.6f}) &nbsp; | &nbsp; "
                        f"**f(optimum)**: {max_val:.6f} &nbsp; | &nbsp; "
                        f"**∇f**: (df/dx={grad[0]:.3g}, df/dy={grad[1]:.3g})")

        except Exception as e:
            st.error(f"Error: {e}")
