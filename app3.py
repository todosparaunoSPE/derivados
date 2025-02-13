# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:42:00 2025

@author: jperezr
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import norm

# Estilo de fondo
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background:
radial-gradient(black 15%, transparent 16%) 0 0,
radial-gradient(black 15%, transparent 16%) 8px 8px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 0 1px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 8px 9px;
background-color:#282828;
background-size:16px 16px;
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# Configuración inicial de la app
st.set_page_config(page_title="Análisis de Derivados - PENSIONISSSTE", layout="wide")
st.title("📈 Análisis de Derivados para PENSIONISSSTE")
st.write("**Desarrollado por Javier Horacio Pérez Ricárdez**")

# Botón para descargar el archivo PDF
st.sidebar.download_button(
    label="Descargar Manual en PDF",
    data=open("manual.pdf", "rb").read(),
    file_name="manual.pdf",
    mime="application/pdf"
)

# Sección de Ayuda
st.sidebar.title("ℹ️ Ayuda")
st.sidebar.info(
    "Esta aplicación permite analizar derivados financieros como Futuros, Forwards, Opciones y Swaps, aplicados a PENSIONISSSTE. "
    "Incluye visualización de datos, cálculo de precios, simulaciones, análisis de cobertura, backtesting y control de riesgos."
)

# 1. VISUALIZACIÓN DE DATOS
st.header("1️⃣ Visualización de Datos de Derivados")
precio_subyacente = np.linspace(80, 120, 100)
opcion_call = np.maximum(precio_subyacente - 100, 0)

df_viz = pd.DataFrame({"Precio Subyacente": precio_subyacente, "Valor Opción Call": opcion_call})
st.write("### Datos de Derivados - Opción Call")
st.dataframe(df_viz)
fig = px.line(df_viz, x="Precio Subyacente", y="Valor Opción Call", title="Valor de una Opción Call vs. Precio Subyacente")
st.plotly_chart(fig)

# 2. CÁLCULO DE PRECIOS Y VALUACIÓN (Black-Scholes para Opciones)
st.header("2️⃣ Cálculo de Precios y Valuación de Derivados")
st.subheader("Modelo Black-Scholes para Opciones")

S = st.number_input("Precio del Activo Subyacente", value=100.0)
K = st.number_input("Precio de Ejercicio", value=100.0)
r = st.number_input("Tasa Libre de Riesgo (%)", value=5.0) / 100
t = st.number_input("Tiempo al Vencimiento (años)", value=1.0)
sigma = st.number_input("Volatilidad (%)", value=20.0) / 100

d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
d2 = d1 - sigma * np.sqrt(t)
call_price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
st.write(f"Precio de la opción Call: **${call_price:.2f}**")

# 3. SIMULACIÓN Y ANÁLISIS DE COBERTURA
st.header("3️⃣ Simulación y Análisis de Cobertura")
n_simulaciones = st.slider("Número de Simulaciones Monte Carlo", 100, 10000, 1000)
sim_returns = np.random.normal(r, sigma, (n_simulaciones, 252))
sim_prices = S * np.exp(np.cumsum(sim_returns, axis=1))

df_sim = pd.DataFrame(sim_prices.T, columns=[f"Simulación {i+1}" for i in range(n_simulaciones)])
st.write("### Datos de Simulación Monte Carlo")
st.dataframe(df_sim.head(10))
fig_sim = px.line(df_sim, title="Simulación Monte Carlo de Precios del Subyacente")
st.plotly_chart(fig_sim)

# 4. BACKTESTING DE ESTRATEGIAS
st.header("4️⃣ Backtesting de Estrategias")
dias = 252
rendimientos = np.random.normal(0, 0.02, dias)
precios = 100 * np.exp(np.cumsum(rendimientos))
df_backtest = pd.DataFrame({"Día": np.arange(dias), "Precio": precios})
st.write("### Datos de Backtesting de Estrategias")
st.dataframe(df_backtest.head(10))
fig_backtest = px.line(df_backtest, x="Día", y="Precio", title="Backtesting de Estrategias de Derivados")
st.plotly_chart(fig_backtest)

# 5. ALERTAS Y LÍMITES DE APALANCAMIENTO
st.header("5️⃣ Alertas y Límites de Apalancamiento")
apalancamiento_max = st.slider("Límite de Apalancamiento (%)", 100, 500, 200)
actual_apalancamiento = np.random.randint(50, 400)
st.write(f"El apalancamiento actual es: **{actual_apalancamiento}%**")
if actual_apalancamiento > apalancamiento_max:
    st.error("⚠️ ¡El apalancamiento excede el límite permitido!")
else:
    st.success("✅ El apalancamiento está dentro del límite permitido.")

# 6. CÁLCULO DE ROLLOVER PARA FUTUROS
st.header("6️⃣ Cálculo de Rollover para Futuros")
contratos = st.slider("Número de contratos en posición", 1, 100, 10)
dias_vencimiento = st.slider("Días para el vencimiento actual", 1, 30, 5)
dias_renovación = st.slider("Días antes del vencimiento para renovar", 1, 10, 3)

if dias_vencimiento <= dias_renovación:
    st.warning("⚠️ ¡Es momento de realizar el rollover de los futuros!")
else:
    st.info("📊 Todavía hay tiempo antes del vencimiento de los futuros.")

# Copyright
st.sidebar.write("© 2025 Javier Horacio Pérez Ricárdez - Todos los derechos reservados.")
