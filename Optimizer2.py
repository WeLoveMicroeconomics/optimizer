import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sympy import symbols, sympify, lambdify, Min, Max, diff
from scipy.optimize import linprog

st.title("Constrained Optimization Visualizer")

# User Inputs
f_expr_str = st.text_input("Objective function f(x, y):", "x + 2*y")
constraint_expr_str = st.text_input("Constraint (e.g., x + y <= 10):", "x + y <= 10")

col1, col2 = st.columns(2)
with col1:
    xmin = st.number_input("x min", value=0.0)
    xmax = st.number_input("x max", value=10.0)
with col2:
    ymin = st.number_input("y min", value=0.0)
    ymax = st.number_input("y max", value=10.0)

if st.button("Update Plot"):
    x, y = symbols('x y')

    # Support min, max functions in sympify
    local_dict = {"min": Min, "max": Max}
    try:
        f_expr = sympify(f_expr_str, locals=local_dict)
        constraint_expr = sympify(constraint_expr_str.replace("<=", "-" + str(0)), locals=local_dict)
    except Exception as e:
        st.error(f"Invalid expression: {e}")
        st.stop()

    f_func = lambdify((x, y), f_expr, "numpy")
    constraint_func = lambdify((x, y), constraint_expr, "numpy")

    x_vals = np.linspace(xmin, xmax, 300)
    y_vals = np.linspace(ymin, ymax, 300)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z = f_func(X, Y)
    try:
        feasible_mask = constraint_func(X, Y) <= 0
    except:
        st.error("Constraint evaluation failed. Ensure correct syntax.")
        st.stop()

    # Gradient at max point (partial derivatives)
    try:
        fx = diff(f_expr, x)
        fy = diff(f_expr, y)
    except:
        fx = fy = 0

    fx_func = lambdify((x, y), fx, "numpy")
    fy_func = lambdify((x, y), fy, "numpy")

    # Linear optimization for linear objectives
    try:
        c = [-fx_func(0, 0), -fy_func(0, 0)]  # Coefficients negated for maximization
        A = []
        b = []

        if "<=" in constraint_expr_str:
            left, right = constraint_expr_str.split("<=")
            left_expr = sympify(left, locals=local_dict)
            coef_x = diff(left_expr, x).subs({x: 0, y: 0})
            coef_y = diff(left_expr, y).subs({x: 0, y: 0})
            rhs = float(right)
            A.append([float(coef_x), float(coef_y)])
            b.append(rhs)

        bounds = [(xmin, xmax), (ymin, ymax)]

        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)

        if res.success:
            max_x, max_y = res.x
            max_val = f_func(max_x, max_y)
            grad_x = fx_func(max_x, max_y)
            grad_y = fy_func(max_x, max_y)
        else:
            st.warning("Optimization failed. Showing feasible region only.")
            max_x = max_y = max_val = grad_x = grad_y = None
    except:
        st.warning("Non-linear objective or constraint. Skipping optimization.")
        max_x = max_y = max_val = grad_x = grad_y = None

    # Plotting
    fig = go.Figure()

    fig.add_trace(go.Contour(
        z=Z,
        x=x_vals,
        y=y_vals,
        colorscale='Viridis',
        contours=dict(coloring='lines', showlabels=True),
        line=dict(color='black', width=2),
        showscale=False,
        name="Level Curves"
    ))

    fig.add_trace(go.Contour(
        z=feasible_mask.astype(float),
        x=x_vals,
        y=y_vals,
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(30,144,255,0.5)']],
        showscale=False,
        opacity=0.5,
        name="Feasible Region"
    ))

    if max_x is not None:
        fig.add_trace(go.Scatter(
            x=[max_x],
            y=[max_y],
            mode='markers+text',
            marker=dict(color='red', size=10),
            text=["Max"],
            textposition="top center",
            name="Maximum"
        ))

        fig.add_trace(go.Scatter(
            x=[max_x, max_x + grad_x],
            y=[max_y, max_y + grad_y],
            mode='lines',
            line=dict(color='red', dash='dot'),
            name="Gradient"
        ))

    fig.update_layout(
        title="Constrained Optimization Plot",
        xaxis_title="x",
        yaxis_title="y",
        xaxis_range=[xmin, xmax],
        yaxis_range=[ymin, ymax],
        template="plotly_white",
        legend=dict(orientation="h")
    )

    st.plotly_chart(fig)

    if max_x is not None:
        st.markdown(f"**Maximum at:** ({max_x:.4f}, {max_y:.4f})  ")
        st.markdown(f"**Objective value:** {max_val:.4f}")
        st.markdown(f"**Gradient:** (df/dx = {grad_x:.2f}, df/dy = {grad_y:.2f})")
