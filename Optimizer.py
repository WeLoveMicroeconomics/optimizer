import streamlit as st
import numpy as np
import plotly.graph_objs as go
from sympy import symbols, sympify, lambdify, diff
from scipy.optimize import minimize

st.set_page_config(layout="wide")
st.title("Constrained Optimization Visualizer (Nonlinear Supported)")

# Inputs
f_expr = st.text_input("f(x, y):", "min(x, y)")
constraint = st.text_input("Constraint (e.g. x + y <= 10):", "x + y <= 10")

xmin, xmax = st.number_input("x min:", value=0.0), st.number_input("x max:", value=10.0)
ymin, ymax = st.number_input("y min:", value=0.0), st.number_input("y max:", value=10.0)

update = st.button("Update Plot")

warning_placeholder = st.empty()
plot_placeholder = st.empty()
explanation_placeholder = st.empty()

if update:
    if xmin >= xmax or ymin >= ymax:
        warning_placeholder.error("Ensure that min values are less than max values.")
    else:
        try:
            x_sym, y_sym = symbols("x y")
            f_sym = sympify(f_expr, evaluate=False)

            # Lambdify f: we want numpy vectorized func
            # Special case for min(x,y):
            if f_expr.strip().lower().startswith("min"):
                # manually handle min for numpy arrays
                def f_func(x_val, y_val):
                    return np.minimum(x_val, y_val)
            else:
                f_func = lambdify((x_sym, y_sym), f_sym, "numpy")

            # Parse constraint - support <= or ==
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

            # Define constraint functions for scipy minimize
            if operator == "<=":
                cons = {'type': 'ineq', 'fun': lambda v: rhs_val - lhs_func(v[0], v[1])}
            else:
                cons = {'type': 'eq', 'fun': lambda v: lhs_func(v[0], v[1]) - rhs_val}

            bounds = [(xmin, xmax), (ymin, ymax)]

            # Objective to minimize (negative for maximization)
            def objective(v):
                try:
                    val = f_func(v[0], v[1])
                    # if val is array-like, take scalar - fallback
                    if isinstance(val, np.ndarray):
                        val = val.item()
                    if np.isnan(val):
                        return 1e10
                    return -val
                except Exception:
                    return 1e10

            # Initial guess: center of bounds
            x0 = [(xmin + xmax) / 2, (ymin + ymax) / 2]

            res = minimize(objective, x0, bounds=bounds, constraints=cons)

            if not res.success:
                warning_placeholder.error(f"Optimization failed: {res.message}")
                st.stop()

            max_point = res.x
            max_val = -res.fun

            # Numerical gradient at max_point
            eps = 1e-6
            def num_grad(func, v):
                grad = np.zeros_like(v)
                for i in range(len(v)):
                    v_eps1 = v.copy()
                    v_eps2 = v.copy()
                    v_eps1[i] += eps
                    v_eps2[i] -= eps
                    f1 = func(v_eps1)
                    f2 = func(v_eps2)
                    grad[i] = (f1 - f2) / (2 * eps)
                return grad

            grad = num_grad(objective, max_point)
            grad = -grad  # because objective is negative of f

            # Prepare grid for contour plot
            x_vals = np.linspace(xmin, xmax, 300)
            y_vals = np.linspace(ymin, ymax, 300)
            X, Y = np.meshgrid(x_vals, y_vals)

            # Compute function values on grid
            Z = f_func(X, Y)
            if isinstance(Z, (int, float)):
                Z = Z * np.ones_like(X)

            # Compute feasible mask
            lhs_grid = lhs_func(X, Y)
            if operator == "<=":
                feasible_mask = lhs_grid <= rhs_val + 1e-8
            else:
                feasible_mask = np.abs(lhs_grid - rhs_val) < 1e-4

            Z_masked = np.where(feasible_mask, Z, np.nan)

            # Plotting
            fig = go.Figure()

            fig.add_trace(go.Contour(
                z=Z,
                x=x_vals,
                y=y_vals,
                colorscale='Greys',
                contours=dict(coloring='none', showlabels=True),
                line=dict(width=1),
                name='f(x,y)'
            ))

            fig.add_trace(go.Contour(
                z=feasible_mask.astype(float),
                x=x_vals,
                y=y_vals,
                colorscale='Viridis',                # More visible colorscale
                contours=dict(coloring='lines', showlabels=True),
                line=dict(width=2),
                name='f(x,y)'
            ))

            fig.add_trace(go.Scatter(
                x=[max_point[0]],
                y=[max_point[1]],
                mode='markers+text',
                marker=dict(color='red', size=10),
                text=['Max'],
                textposition='top center',
                name='Maximum Point'
            ))

            arrow_x = max_point[0] + grad[0]
            arrow_y = max_point[1] + grad[1]

            fig.add_shape(
                type='line',
                x0=max_point[0], y0=max_point[1],
                x1=arrow_x, y1=arrow_y,
                line=dict(color='red', width=2, dash='dot')
            )

            fig.update_layout(
                title='Constrained Optimization Plot',
                xaxis=dict(title='x', range=[xmin, xmax]),
                yaxis=dict(title='y', range=[ymin, ymax]),
                font=dict(family='Arial', size=14),
                legend=dict(orientation='h')
            )

            plot_placeholder.plotly_chart(fig, use_container_width=True)

            explanation_placeholder.markdown(f"""
                ### Explanation
                - **Function:** `{f_expr}`
                - **Constraint:** `{constraint}`
                - **Max at:** ({max_point[0]:.6f}, {max_point[1]:.6f})
                - **f(max):** {max_val:.6f}
                - **Gradient at max:** (df/dx = {grad[0]:.2f}, df/dy = {grad[1]:.2f})
            """)

        except Exception as e:
            warning_placeholder.error(f"Error: {e}")
