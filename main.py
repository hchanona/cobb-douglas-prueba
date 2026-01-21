import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# -----------------------------
# Cobb-Douglas
# u(x1,x2) = x1^alpha * x2^(1-alpha)
# Demandas Marshallianas:
# x1* = alpha * m / p1
# x2* = (1-alpha) * m / p2
# -----------------------------
def demands_cd(m: float, p1: float, p2: float, alpha: float):
    x1 = alpha * m / p1
    x2 = (1.0 - alpha) * m / p2
    return x1, x2


def utility_cd(x1, x2, alpha: float):
    x1 = np.maximum(x1, 1e-12)
    x2 = np.maximum(x2, 1e-12)
    return (x1 ** alpha) * (x2 ** (1.0 - alpha))


def mrs_cd(x1: float, x2: float, alpha: float):
    # MRS = (alpha/(1-alpha)) * (x2/x1)
    return (alpha / (1.0 - alpha)) * (x2 / x1)


def make_choice_data(m, p1, p2, alpha, n_grid=250):
    x1_star, x2_star = demands_cd(m, p1, p2, alpha)
    u_star = utility_cd(x1_star, x2_star, alpha)

    # límites del plano (x1,x2)
    x1_max = max(1.0, (m / p1) * 1.20)
    x2_max = max(1.0, (m / p2) * 1.20)

    # recta presupuestaria
    x1_line = np.linspace(0.0, x1_max, n_grid)
    x2_line = (m - p1 * x1_line) / p2
    x2_line = np.clip(x2_line, 0.0, None)

    budget = pd.DataFrame({
        "x1": x1_line,
        "x2": x2_line,
        "obj": "Presupuesto"
    })

    # curvas de indiferencia (analíticas, no contour)
    # Para un nivel U: x2 = (U / x1^alpha)^(1/(1-alpha))
    levels = np.array([0.65, 0.80, 1.00, 1.20, 1.40]) * u_star
    x1_iso = np.linspace(1e-6, x1_max, n_grid)

    indiff_list = []
    for i, U in enumerate(levels, start=1):
        x2_iso = (U / (x1_iso ** alpha)) ** (1.0 / (1.0 - alpha))
        x2_iso = np.clip(x2_iso, 0.0, x2_max)
        indiff_list.append(pd.DataFrame({
            "x1": x1_iso,
            "x2": x2_iso,
            "obj": f"Indiferencia {i}"
        }))
    indiff = pd.concat(indiff_list, ignore_index=True)

    optimum = pd.DataFrame({
        "x1": [x1_star],
        "x2": [x2_star],
        "obj": ["Óptimo"]
    })

    return budget, indiff, optimum, x1_star, x2_star, u_star, x1_max, x2_max


def make_demand_data(m, p1, p2, alpha, pmin=0.5, pmax=20.0, n=120):
    pgrid = np.linspace(pmin, pmax, n)
    x1_vals = alpha * m / pgrid
    x2_vals = (1.0 - alpha) * m / pgrid

    d1 = pd.DataFrame({"p": pgrid, "x": x1_vals, "curva": "x1(p1) con m,p2 fijos"})
    d2 = pd.DataFrame({"p": pgrid, "x": x2_vals, "curva": "x2(p2) con m,p1 fijos"})
    return d1, d2


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Cobb-Douglas: Decisión y Demanda", layout="wide")
st.title("Cobb-Douglas: decisión del consumidor y demanda marshalliana")

st.markdown(r"""
Preferencias:
\[
u(x_1,x_2)=x_1^{\alpha}x_2^{1-\alpha}
\]
Problema:
\[
\max_{x_1,x_2\ge 0}\; u(x_1,x_2)\quad \text{s.a.}\quad p_1x_1+p_2x_2\le m.
\]
En Cobb-Douglas, la decisión óptima implica **regla de gasto**:
\[
p_1x_1^\*=\alpha m,\qquad p_2x_2^\*=(1-\alpha)m.
\]
""")

with st.sidebar:
    st.header("Parámetros")
    m = st.slider("Ingreso (m)", 1.0, 500.0, 100.0, 1.0)
    p1 = st.slider("Precio p1", 0.2, 50.0, 5.0, 0.1)
    p2 = st.slider("Precio p2", 0.2, 50.0, 5.0, 0.1)
    alpha = st.slider("α (peso en x1)", 0.05, 0.95, 0.50, 0.05)

    st.divider()
    st.subheader("Demandas (rango de precios)")
    show_demands = st.checkbox("Mostrar curvas de demanda", True)
    pmin = st.slider("Precio mínimo", 0.2, 10.0, 0.5, 0.1)
    pmax = st.slider("Precio máximo", 10.0, 80.0, 20.0, 1.0)
    ngrid = st.slider("Resolución", 40, 300, 140, 10)


# Cálculos
budget, indiff, optimum, x1_star, x2_star, u_star, x1_max, x2_max = make_choice_data(m, p1, p2, alpha)

e1 = p1 * x1_star
e2 = p2 * x2_star

mrs_star = mrs_cd(x1_star, x2_star, alpha)
price_ratio = p1 / p2


# Bloque didáctico: “la decisión”
st.subheader("La decisión del consumidor (clarita y verificable)")

st.markdown(rf"""
**Regla de gasto:** el consumidor asigna el ingreso en proporciones fijas:  
\[
\underbrace{{p_1x_1^\*}}_{{\text{{gasto en 1}}}}=\alpha m
\qquad\text{{y}}\qquad
\underbrace{{p_2x_2^\*}}_{{\text{{gasto en 2}}}}=(1-\alpha)m.
\]

**De ahí se obtiene la canasta óptima:**
\[
x_1^\*=\frac{{\alpha m}}{{p_1}},\qquad x_2^\*=\frac{{(1-\alpha)m}}{{p_2}}.
\]
""")

c1, c2, c3, c4 = st.columns(4)
c1.metric("x1*", f"{x1_star:.4f}")
c2.metric("x2*", f"{x2_star:.4f}")
c3.metric("p1·x1*", f"{e1:.2f}")
c4.metric("p2·x2*", f"{e2:.2f}")

st.markdown(rf"""
**Chequeo de optimalidad (tangencia):**  
\[
\text{{MRS}}(x^\*)=\frac{{\alpha}}{{1-\alpha}}\cdot\frac{{x_2^\*}}{{x_1^\*}}\approx \frac{{p_1}}{{p_2}}.
\]
Con tus valores:
- \(\text{{MRS}}(x^\*) = {mrs_star:.4f}\)
- \(\frac{{p_1}}{{p_2}} = {price_ratio:.4f}\)
""")


# Gráfica de elección (Altair, interactiva)
st.subheader("Plano (x1, x2): presupuesto, indiferencia y óptimo")

base = alt.Chart(pd.concat([budget, indiff], ignore_index=True)).encode(
    x=alt.X("x1:Q", title="x1", scale=alt.Scale(domain=[0, x1_max])),
    y=alt.Y("x2:Q", title="x2", scale=alt.Scale(domain=[0, x2_max])),
)

lines = base.mark_line().encode(
    detail="obj:N",
    tooltip=["obj:N", alt.Tooltip("x1:Q", format=".3f"), alt.Tooltip("x2:Q", format=".3f")]
)

opt_pt = alt.Chart(optimum).mark_point(size=120).encode(
    x="x1:Q", y="x2:Q",
    tooltip=[alt.Tooltip("x1:Q", format=".4f"), alt.Tooltip("x2:Q", format=".4f")]
)

opt_label = alt.Chart(optimum).mark_text(align="left", dx=10, dy=-10).encode(
    x="x1:Q", y="x2:Q",
    text=alt.value(f"Óptimo ({x1_star:.2f}, {x2_star:.2f})")
)

st.altair_chart((lines + opt_pt + opt_label).interactive(), use_container_width=True)


# Demandas (ceteris paribus)
if show_demands:
    st.subheader("Demandas Marshallianas (ceteris paribus)")

    d1, d2 = make_demand_data(m, p1, p2, alpha, pmin=pmin, pmax=pmax, n=ngrid)

    colL, colR = st.columns(2)

    with colL:
        chart1 = alt.Chart(d1).mark_line().encode(
            x=alt.X("p:Q", title="p1 (varía)"),
            y=alt.Y("x:Q", title="x1(p1)"),
            tooltip=[alt.Tooltip("p:Q", format=".2f"), alt.Tooltip("x:Q", format=".3f")]
        )
        point1 = alt.Chart(pd.DataFrame({"p":[p1], "x":[alpha*m/p1]})).mark_point(size=90).encode(x="p:Q", y="x:Q")
        st.altair_chart((chart1 + point1).interactive(), use_container_width=True)

    with colR:
        chart2 = alt.Chart(d2).mark_line().encode(
            x=alt.X("p:Q", title="p2 (varía)"),
            y=alt.Y("x:Q", title="x2(p2)"),
            tooltip=[alt.Tooltip("p:Q", format=".2f"), alt.Tooltip("x:Q", format=".3f")]
        )
        point2 = alt.Chart(pd.DataFrame({"p":[p2], "x":[(1-alpha)*m/p2]})).mark_point(size=90).encode(x="p:Q", y="x:Q")
        st.altair_chart((chart2 + point2).interactive(), use_container_width=True)

st.caption("Si sube m, suben x1* y x2* proporcionalmente. Si sube p1 (con lo demás fijo), cae x1* como αm/p1.")
