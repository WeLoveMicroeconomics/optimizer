import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sympy import symbols, lambdify, sympify, Min, Max, diff
from scipy.optimize import linprog

st.set_page_config(page_title="Constrained Optimization Visualizer", layout="wide")

st.title("Constrained Optimization Visualizer with LP Solver")

# User inputs
f_expr = st.text_input("Objective function f(x, y):", "x + 2*y")
constraint_expr = st.text_input("Constraint (e.g., x + y <= 10):", "x + y <= 10")
xmin, xmax = st.slider("x range", -20.0, 20.0, (0.0, 10.0))
ymin, ymax = st.slider("y range", -20.0, 20.0, (0.0, 10.0))

# Create grid
x_vals = np.linspace(xmin, xmax, 300)
y_vals = np.linspace(ymin, ymax, 300)
X, Y = np.meshgrid(x_vals, y_vals)

# Symbolic variables
x, y = symbols('x y')
local_dict = {'min': Min, 'max': Max}

try:
    f_sym = sympify(f_expr, locals=local_dict)
    f_func = lambdify((x, y), f_sym, modules=['numpy'])
    Z = f_func(X, Y)
except Exception as e:
    st.error(f"Invalid function: {e}")
    st.stop()

# Constraint handling
try:
    if '<=' in constraint_expr:
        lhs, rhs = constraint_expr.split('<=')
        bound = float(rhs.strip())
        lhs_func = lambdify((x, y), sympify(lhs.strip(), locals=local_dict), modules=['numpy'])
        mask = lhs_func(X, Y) <= bound
    elif '==' in constraint_expr:
        lhs, rhs = constraint_expr.split('==')
        bound = float(rhs.strip())
        lhs_func = lambdify((x, y), sympify(lhs.strip(), locals=local_dict), modules=['numpy'])
        mask = np.isclose(lhs_func(X, Y), bound)
    else:
        st.error("Only '<=' and '==' constraints are supported.")
        st.stop()
except Exception as e:
    st.error(f"Invalid constraint: {e}")
    st.stop()

# Linear programming solver setup
try:
    grad_x = float(diff(f_sym, x).subs({x: 0, y: 0}))
    grad_y = float(diff(f_sym, y).subs({x: 0, y: 0}))
except:
    grad_x, grad_y = 0, 0

c = [-grad_x, -grad_y]  # Negate for maximization
A, b = [], []

if '<=' in constraint_expr:
    lhs, rhs = constraint_expr.split('<=')
    lhs_x = float(diff(sympify(lhs, locals=local_dict), x).subs({x: 0, y: 0}))
    lhs_y = float(diff(sympify(lhs, locals=local_dict), y).subs({x: 0, y: 0}))
    A.append([lhs_x, lhs_y])
    b.append(float(rhs.strip()))

bounds = [(xmin, xmax), (ymin, ymax)]

result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

if result.success:
    opt_x, opt_y = result.x
    opt_val = f_func(opt_x, opt_y)
else:
    opt_x = opt_y = opt_val = np.nan

# Plotting
fig = go.Figure()

fig.add_trace(go.Contour(
    z=Z,
    x=x_vals,
    y=y_vals,
    colorscale='Viridis',
    contours=dict(coloring='lines', showlabels=True),
    line=dict(color='black', width=2),
    name='f(x,y)'
))

fig.add_trace(go.Contour(
    z=mask.astype(float),
    x=x_vals,
    y=y_vals,
    showscale=False,
    opacity=0.7,
    colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(30,144,255,0.5)']],
    contours=dict(coloring='heatmap', showlines=False),
    name='Feasible Region'
))

if result.success:
    fig.add_trace(go.Scatter(
        x=[opt_x],
        y=[opt_y],
        mode='markers+text',
        marker=dict(color='red', size=10),
        text=['Max'],
        textposition='top center',
        name='Maximum Point'
    ))

    fig.add_shape(type='line',
        x0=opt_x, y0=opt_y, x1=opt_x + grad_x, y1=opt_y + grad_y,
        line=dict(color='red', width=2, dash='dot')
    )

fig.update_layout(
    title='Constrained Optimization Plot',
    xaxis=dict(title='x', range=[xmin, xmax], color='black'),
    yaxis=dict(title='y', range=[ymin, ymax], color='black'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family='Arial', size=14, color='black'),
    legend=dict(orientation='h')
)

st.plotly_chart(fig, use_container_width=True)

if result.success:
    st.success(f"Maximum at (x, y) = ({opt_x:.4f}, {opt_y:.4f}) with f(x, y) = {opt_val:.4f}")
else:
    st.warning("Linear programming did not find a feasible solution.")
