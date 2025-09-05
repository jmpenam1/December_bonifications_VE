# app.py — Calculadora de Bonificaciones de Fin de Año (VE Group)
# ---------------------------------------------------------------
# Nueva lógica según especificaciones:
# - CSV mínimo: Name, Salary, HireDate, Performance (0-1).
# - Ventas alcanzadas: input manual (monto en USD) para todo el año.
# - Meta anual (Target) y umbral de activación (%), p.ej. 15M USD y 80%.
# - Porcentaje de bonificación (sobre salario) empieza en start_pct al umbral
#   y escala lineal hasta pct_100 en 100% de la meta.
# - Sobrecumplimiento: por cada +1% > 100% de la meta, aumenta bonus +inc_perc_pp
#   (puntos porcentuales sobre el salario). Todo parametrizable en la UI.
# - Antigüedad: factor = 1.0 si HireDate <= 2025-01-01; si no, proporcional de
#   (días trabajados hasta 2025-12-31) / (días del año 2025).
# - Bonificación final = Salary * TenureFactor * Performance * BonusPctBySales.
# - UI: Cargar CSV o ingreso manual de un empleado, ventas alcanzadas, y gráficos.

import io
import json
import datetime as dt
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------
# Configuración de la página
# ---------------------------
st.set_page_config(
    page_title="Calculadora Bonos VE Group",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Calculadora de Bonificaciones de Fin de Año — VE Group")
st.caption("Versión basada en reglas: umbral 80% → 50% del salario; 100% → 100%; >100% +3pp por cada +1% (todo editable)")

# ---------------------------
# Utilidades y plantilla
# ---------------------------

@st.cache_data
def plantilla_empleados() -> pd.DataFrame:
    data = {
        "Name": ["Ana Pérez", "Luis Gómez", "María Ruiz"],
        "Salary": [3500_000, 3200_000, 4000_000],  # en COP por defecto (ajusta si usas USD)
        "HireDate": ["2024-02-10", "2025-05-15", "2023-11-03"],
        "Performance": [0.85, 0.90, 0.80],  # 0-1
    }
    return pd.DataFrame(data)


def _parse_date(d):
    try:
        return pd.to_datetime(d).date()
    except Exception:
        return None


def tenure_factor(hire_date: dt.date) -> float:
    # Regla: si ingresó el 1-ene-2025 o antes, factor 1.0; si después, proporcional hasta 31-dic-2025
    cutoff = dt.date(2025, 1, 1)
    end = dt.date(2025, 12, 31)
    year_days = (end - dt.date(2025, 1, 1)).days + 1  # 365 en 2025
    if not isinstance(hire_date, dt.date):
        return 1.0
    if hire_date <= cutoff:
        return 1.0
    if hire_date > end:
        return 0.0
    worked_days = (end - hire_date).days + 1
    return max(0.0, min(1.0, worked_days / year_days))


def bonus_pct_by_sales(ratio: float, threshold: float, start_pct: float, pct_100: float, inc_pp_over_100: float, cap_pct: float | None) -> float:
    """
    ratio: ventas/meta (e.g., 0.82 = 82%).
    start_pct: porcentaje sobre salario en el umbral (e.g., 0.50 para 50%).
    pct_100: porcentaje sobre salario en 100% (e.g., 1.00 para 100%).
    inc_pp_over_100: incremento en puntos porcentuales por cada +1%>100% (e.g., 0.03 = +3pp).
    cap_pct: tope opcional del porcentaje total sobre salario (None = sin tope).
    """
    if ratio < threshold:
        pct = 0.0
    elif ratio < 1.0:
        # Interpolación lineal entre start_pct (en threshold) y pct_100 (en 1.0)
        span = 1.0 - threshold
        frac = 0.0 if span <= 0 else (ratio - threshold) / span
        pct = start_pct + (pct_100 - start_pct) * frac
    else:
        over = max(0.0, ratio - 1.0)  # p.ej., 1.05 => 0.05 (5%)
        pct = pct_100 + over * 100.0 * inc_pp_over_100
    if cap_pct is not None:
        pct = min(pct, cap_pct)
    return max(0.0, pct)


# ---------------------------
# Sidebar — Parámetros
# ---------------------------
st.sidebar.header("⚙️ Parámetros de la política")

# Ventas y meta
col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    target_amount = st.number_input("Meta anual (USD)", min_value=0.0, value=15_000_000.0, step=100_000.0, format="%f")
with col_t2:
    achieved_amount = st.number_input("Ventas alcanzadas (USD)", min_value=0.0, value=12_000_000.0, step=100_000.0, format="%f")

threshold = st.slider("Umbral de activación (% de meta)", min_value=0, max_value=100, value=80, step=1) / 100.0
start_pct = st.number_input("% Bono al umbral (sobre salario)", min_value=0.0, max_value=5.0, value=0.50, step=0.05)
pct_100 = st.number_input("% Bono al 100% (sobre salario)", min_value=0.0, max_value=5.0, value=1.00, step=0.05)
inc_pp_over_100 = st.number_input("+pp por cada +1% > 100%", min_value=0.0, max_value=0.2, value=0.03, step=0.005)
cap_pct_enabled = st.checkbox("Aplicar tope máximo de % sobre salario", value=True)
cap_pct = st.number_input("Tope máximo (% sobre salario)", min_value=0.0, max_value=10.0, value=3.0, step=0.1) if cap_pct_enabled else None

sales_ratio = (achieved_amount / target_amount) if target_amount > 0 else 0.0
sales_pct_view = f"{sales_ratio*100:.2f}%"

st.sidebar.markdown(f"**Cumplimiento actual:** {sales_pct_view}")

policy = {
    "target_amount": target_amount,
    "achieved_amount": achieved_amount,
    "threshold": threshold,
    "start_pct": start_pct,
    "pct_100": pct_100,
    "inc_pp_over_100": inc_pp_over_100,
    "cap_pct": cap_pct,
}

st.sidebar.download_button(
    "⬇️ Descargar snapshot de política (JSON)",
    data=json.dumps(policy, indent=2),
    file_name="policy_bonus.json",
    mime="application/json",
)

# ---------------------------
# Datos de empleados
# ---------------------------
st.subheader("1) Datos de empleados")
left, right = st.columns([2, 1])
with left:
    uploaded = st.file_uploader("Sube un CSV con columnas: Name, Salary, HireDate, Performance", type=["csv"]).
with right:
    tpl = plantilla_empleados()
    st.write("Plantilla de ejemplo:")
    st.dataframe(tpl, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Descargar plantilla CSV",
        data=tpl.to_csv(index=False).encode("utf-8"),
        file_name="empleados_minimo.csv",
        mime="text/csv",
    )

manual_tab, _ = st.tabs(["Ingreso manual", " "])
with manual_tab:
    st.markdown("**Ingreso rápido (un empleado)**")
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        m_name = st.text_input("Nombre", value="Empleado Demo")
        m_salary = st.number_input("Salario", min_value=0.0, value=3_500_000.0, step=100_000.0)
    with mcol2:
        m_hire = st.date_input("Fecha de ingreso", value=dt.date(2024, 1, 1))
        m_perf = st.number_input("Performance (0-1)", min_value=0.0, max_value=1.0, value=0.85, step=0.01)

    df_manual = pd.DataFrame([
        {"Name": m_name, "Salary": m_salary, "HireDate": m_hire, "Performance": m_perf}
    ])

# Selección de fuente de datos
if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = df_manual.copy()

# Normalización
if "HireDate" in df_raw.columns:
    df_raw["HireDate"] = pd.to_datetime(df_raw["HireDate"], errors="coerce").dt.date

expected_cols = ["Name", "Salary", "HireDate", "Performance"]
missing = [c for c in expected_cols if c not in df_raw.columns]
if missing:
    st.warning(f"Faltan columnas mínimas: {missing}")

# ---------------------------
# Cálculo
# ---------------------------

def compute(df: pd.DataFrame, pol: Dict[str, Any]) -> pd.DataFrame:
    ratio = (pol["achieved_amount"] / pol["target_amount"]) if pol["target_amount"] > 0 else 0.0
    out_rows = []
    for _, r in df.iterrows():
        salary = float(r.get("Salary", 0) or 0)
        perf = float(r.get("Performance", 0) or 0)
        hdate = r.get("HireDate")
        tf = tenure_factor(hdate)
        bonus_pct = bonus_pct_by_sales(
            ratio=ratio,
            threshold=pol["threshold"],
            start_pct=pol["start_pct"],
            pct_100=pol["pct_100"],
            inc_pp_over_100=pol["inc_pp_over_100"],
            cap_pct=pol["cap_pct"],
        )
        bonus_amount = salary * tf * perf * bonus_pct
        out_rows.append({
            "TenureFactor": round(tf, 4),
            "%BonusBySales": round(bonus_pct, 4),
            "BonusAmount": round(bonus_amount, 2),
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(out_rows)], axis=1)

st.subheader("2) Resultados")
if st.button("🚀 Calcular bonificaciones"):
    df_calc = compute(df_raw, policy)

    # KPIs
    total_bonus = float(df_calc["BonusAmount"].sum())
    avg_bonus = float(df_calc["BonusAmount"].mean()) if len(df_calc) else 0.0
    headcount = int(df_calc.shape[0])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Empleados", headcount)
    k2.metric("Ventas alcanzadas", f"${policy['achieved_amount']:,.0f}")
    k3.metric("% Cumplimiento", f"{(policy['achieved_amount']/policy['target_amount']*100 if policy['target_amount'] else 0):.2f}%")
    k4.metric("Bono total", f"${total_bonus:,.0f}")

    st.dataframe(df_calc, use_container_width=True)

    # ---------------------------
    # Visualizaciones
    # ---------------------------
    st.subheader("3) Visualizaciones")
    v1, v2 = st.columns(2)

    with v1:
        st.markdown("**Curva de % de Bono vs % de Meta**")
        xs = np.linspace(0, 1.3, 131)  # hasta 130%
        ys = [bonus_pct_by_sales(x, policy["threshold"], policy["start_pct"], policy["pct_100"], policy["inc_pp_over_100"], policy["cap_pct"]) for x in xs]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(xs*100, np.array(ys)*100)
        ax.axvline((sales_ratio*100), linestyle='--')
        ax.set_xlabel("% de meta alcanzada")
        ax.set_ylabel("% de bono sobre salario")
        ax.set_title("Regla de bonificación por cumplimiento")
        st.pyplot(fig)

    with v2:
        st.markdown("**Distribución de Bonos**")
        fig2, ax2 = plt.subplots(figsize=(6,4))
        df_calc["BonusAmount"].plot(kind="hist", bins=10, ax=ax2)
        ax2.set_xlabel("Bono")
        ax2.set_title("Distribución del Bono")
        st.pyplot(fig2)

    v3, v4 = st.columns(2)
    with v3:
        st.markdown("**Top 10 Bonos**")
        cols_show = [c for c in ["Name", "Salary", "TenureFactor", "Performance", "%BonusBySales", "BonusAmount"] if c in df_calc.columns]
        st.dataframe(df_calc.sort_values("BonusAmount", ascending=False).head(10)[cols_show], use_container_width=True, hide_index=True)

    with v4:
        st.markdown("**Indicador de cumplimiento**")
        st.progress(min(1.0, max(0.0, sales_ratio)))
        st.write(f"Cumplimiento actual: **{sales_ratio*100:.2f}%**")

    # ---------------------------
    # Exportación
    # ---------------------------
    st.subheader("4) Exportar")
    csv_bytes = df_calc.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ CSV de resultados", data=csv_bytes, file_name="bonos_resultados.csv", mime="text/csv")
else:
    st.info("Configura parámetros, carga tus datos o usa ingreso manual y pulsa **Calcular bonificaciones**.")

# ---------------------------
# Ayuda
# ---------------------------
with st.expander("ℹ️ Notas y supuestos"):
    st.markdown(
        """
        **CSV mínimo**: `Name`, `Salary`, `HireDate`, `Performance`.

        **Regla de bono**:
        - Si `ventas/meta < umbral` → % bono = 0%.
        - Entre `umbral` y `100%` → interpolación lineal de **start_pct** a **pct_100**.
        - `>100%` → por cada +1% se suma **inc_pp_over_100** (puntos porcentuales) al % bono.
        - Opcional **tope máximo** de % bono total.

        **Antigüedad** (año 2025): si `HireDate ≤ 2025-01-01` → 1.0; si después, proporcional hasta 31-12-2025.

        **Bono final** = `Salary * TenureFactor * Performance * %Bono`.
        """
    )

# ---------------------------
# Requisitos (para requirements.txt)
# ---------------------------
# streamlit>=1.36.0
# pandas>=2.2.2
# numpy>=1.26.4
# matplotlib>=3.8.4
