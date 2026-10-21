"""Streamlit monitoring app for the PaySim fraud model.

Runs from a clean clone: it only needs the versioned scored datasets and the
Evidently snapshots produced by notebook 04. If the local champion model exists,
the real-time simulation scores transactions live with it.

Usage:
    uv run streamlit run app/app.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `uv sync`

from fraud_detection import config  # noqa: E402
from fraud_detection.evaluate import compute_metrics  # noqa: E402
from fraud_detection.features import MODEL_FEATURES, add_features  # noqa: E402
from fraud_detection.monitoring import add_labels, load_snapshot  # noqa: E402

BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#898781", "#52514e"
REPO_URL = "https://github.com/richardmk/Project-MLEngineer3"
DAGSHUB_URL = "https://dagshub.com/richardmk/ProjectMLEngineer3/experiments"
ALL = "Toda la producción"

st.set_page_config(page_title="Monitoreo de fraude · PaySim", page_icon="🛡️", layout="wide")


# --- Data -------------------------------------------------------------------
@st.cache_data
def load_info() -> dict:
    """Model card exported by notebook 04."""
    return json.loads(config.MODEL_INFO_JSON.read_text())


@st.cache_data
def load_summary() -> pd.DataFrame:
    """Per-day monitoring indicators exported by notebook 04."""
    return pd.read_csv(config.MONITORING_SUMMARY_CSV)


@st.cache_data
def load_production(threshold: float) -> pd.DataFrame:
    """Scored production transactions with features, labels and period."""
    raw = pd.read_csv(config.PRODUCTION_SCORED_CSV)
    return add_labels(add_features(raw), threshold)


@st.cache_data(max_entries=4)
def report_html(name: str) -> str:
    """Render a stored Evidently snapshot to HTML (cached per period)."""
    return load_snapshot(config.EVIDENTLY_DIR / f"{name}.json.gz").get_html_str(as_iframe=False)


@st.cache_resource
def load_model():
    """Champion model if the local copy exists (git-ignored), else None."""
    try:
        import joblib

        return joblib.load(config.CHAMPION_MODEL_PATH)
    except (FileNotFoundError, OSError):
        return None


def base_layout(fig: go.Figure, title: str, height: int = 320) -> go.Figure:
    """Shared low-chrome Plotly layout."""
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 15}},
        template="plotly_white",
        height=height,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h", "y": -0.2},
        hovermode="x unified",
        font={"color": INK},
    )
    return fig


def line_vs_reference(
    summary: pd.DataFrame,
    col: str,
    ref: float | None,
    title: str,
    selected: int | None,
    log: bool = False,
) -> go.Figure:
    """Daily indicator with an optional reference line and the selected day highlighted."""
    fig = go.Figure()
    fig.add_scatter(
        x=summary["period"],
        y=summary[col],
        mode="lines+markers",
        name="Producción",
        line={"color": ORANGE, "width": 2},
    )
    if ref is not None:
        fig.add_hline(
            y=ref,
            line={"color": BLUE, "dash": "dash", "width": 1.5},
            annotation_text="referencia / offline",
            annotation_font_color=BLUE,
        )
    if selected is not None:
        fig.add_vrect(
            x0=selected - 0.5, x1=selected + 0.5, fillcolor=GRAY, opacity=0.15, line_width=0
        )
    fig.update_xaxes(title="día simulado", dtick=1)
    if log:
        fig.update_yaxes(type="log")
    return base_layout(fig, title)


# --- Sidebar ----------------------------------------------------------------
info = load_info()
summary = load_summary()
threshold = info["threshold"]

with st.sidebar:
    st.title("🛡️ Monitoreo de fraude")
    st.caption("Modelo de detección de fraude sobre transacciones móviles (PaySim)")
    st.markdown(
        f"[Código fuente]({REPO_URL}) · [Experimentos MLflow]({DAGSHUB_URL}) · "
        "[Evidently docs](https://docs.evidentlyai.com)"
    )
    st.divider()
    labels = {ALL: None} | {
        f"Día {r.period} · steps {r.step_min}–{r.step_max}": int(r.period)
        for r in summary.itertuples()
    }
    choice = st.selectbox("📅 Periodo", list(labels), index=1)
    period = labels[choice]
    st.divider()
    st.markdown(
        f"**Modelo:** `{info['registered_model']}` v{info['version']} (alias `{info['alias']}`)  \n"
        f"**Algoritmo:** {info['model_name']}  \n**Threshold:** {threshold:.4f}  \n"
        f"**Referencia:** steps ≤ 400 · **Producción:** steps ≥ 401"
    )

# --- Header & KPIs ----------------------------------------------------------
st.header(f"Proyecto: detección de fraude PaySim · {choice}")
production = load_production(threshold)
current = production if period is None else production[production["period"] == period]
m = compute_metrics(current["isFraud"].to_numpy(), current["prediction"].to_numpy(), threshold)
offline = info["offline_test"]
row = None if period is None else summary.set_index("period").loc[period]

k = st.columns(6)
k[0].metric("Transacciones", f"{len(current):,}")
k[1].metric(
    "Tasa de fraude",
    f"{current['isFraud'].mean():.2%}",
    f"{current['isFraud'].mean() / info['reference_fraud_rate']:.1f}× referencia",
    delta_color="off",
)
k[2].metric("Alertas del modelo", f"{int(current['prediction_label'].sum()):,}")
k[3].metric(
    "Recall online",
    f"{m['recall']:.2%}",
    f"{(m['recall'] - offline['recall']) * 100:+.2f} pp vs offline",
)
k[4].metric(
    "Precision online",
    f"{m['precision']:.2%}",
    f"{(m['precision'] - offline['precision']) * 100:+.2f} pp vs offline",
)
if row is not None:
    k[5].metric(
        "Features con drift",
        f"{row['features_drift_share']:.0%}",
        "dataset drift" if row["features_drift_share"] > 0.5 else "sin dataset drift",
        delta_color="inverse" if row["features_drift_share"] > 0.5 else "off",
    )
else:
    k[5].metric("Fraudes no detectados", f"{m['fn']:,} de {m['tp'] + m['fn']:,}")

tab_summary, tab_report, tab_drift, tab_live = st.tabs(
    ["📊 Resumen", "🧪 Reporte Evidently", "🔁 Drift por feature", "⚡ Tiempo real"]
)

# --- Summary tab ------------------------------------------------------------
with tab_summary:
    st.caption(
        "Métricas online por día (labels de producción) frente a las métricas offline del test. "
        "La franja gris marca el periodo seleccionado."
    )
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(
        line_vs_reference(summary, "recall", offline["recall"], "Recall online", period),
        width="stretch",
    )
    c2.plotly_chart(
        line_vs_reference(summary, "precision", offline["precision"], "Precision online", period),
        width="stretch",
    )
    c3.plotly_chart(
        line_vs_reference(summary, "f2", offline["f2"], "F2 online", period), width="stretch"
    )
    c4, c5, c6 = st.columns(3)
    c4.plotly_chart(
        line_vs_reference(
            summary,
            "fraud_rate",
            info["reference_fraud_rate"],
            "Tasa de fraude (prior shift, log)",
            period,
            log=True,
        ),
        width="stretch",
    )
    c5.plotly_chart(
        line_vs_reference(
            summary,
            "share_frauds_drain",
            None,
            "% de fraudes que vacían la cuenta (concepto)",
            period,
        ),
        width="stretch",
    )
    c6.plotly_chart(
        line_vs_reference(
            summary, "features_drift_share", 0.5, "% de features con drift (umbral 50 %)", period
        ),
        width="stretch",
    )
    with st.expander("Tabla de indicadores por día"):
        st.dataframe(
            summary[
                [
                    "period",
                    "step_min",
                    "step_max",
                    "transacciones",
                    "fraudes",
                    "fraud_rate",
                    "features_drift_share",
                    "drift_score__prediction",
                    "drift_score__isFraud",
                    "share_frauds_drain",
                    "pr_auc",
                    "precision",
                    "recall",
                    "f2",
                    "fn",
                ]
            ],
            hide_index=True,
            width="stretch",
        )

# --- Evidently tab ----------------------------------------------------------
with tab_report:
    name = "production_all" if period is None else f"day_{period:02d}"
    st.caption(
        "Reporte de Evidently: drift de features, drift del score (prediction), drift del target y "
        "desempeño de clasificación (este último se omite en días con una sola clase)."
    )
    with st.spinner("Renderizando reporte..."):
        # Trusted content: HTML generated by this project from its own Evidently snapshots.
        st.iframe(report_html(name), height=1500)

# --- Drift tab --------------------------------------------------------------
with tab_drift:
    drift_cols = [f"drift_score__{f}" for f in MODEL_FEATURES]
    heat = summary.set_index("period")[drift_cols].T
    heat.index = MODEL_FEATURES
    heat = heat.loc[heat.mean(axis=1).sort_values().index]
    fig = go.Figure(
        go.Heatmap(
            z=heat.clip(upper=1.0).values,
            x=[f"D{p}" for p in heat.columns],
            y=heat.index,
            customdata=heat.values,
            colorscale="Blues",
            zmin=0,
            zmax=1,
            hovertemplate="%{y} · %{x}<br>drift score %{customdata:.3f}<extra></extra>",
            colorbar={"title": "score"},
        )
    )
    st.plotly_chart(
        base_layout(fig, "Drift score por feature y día (umbral 0.1, escala cortada en 1.0)", 520),
        width="stretch",
    )
    if row is not None:
        scores = row[drift_cols].astype(float)
        scores.index = MODEL_FEATURES
        scores = scores.sort_values()
        fig = go.Figure(
            go.Bar(
                x=scores.values,
                y=scores.index,
                orientation="h",
                marker_color=[ORANGE if v >= 0.1 else BLUE for v in scores.values],
            )
        )
        fig.add_vline(x=0.1, line={"color": GRAY, "dash": "dot"}, annotation_text="umbral 0.1")
        fig.update_layout(hovermode="y")
        st.plotly_chart(
            base_layout(fig, f"Drift score por feature · día {period} (naranja = drift)", 480),
            width="stretch",
        )

# --- Real-time tab (Reto ML 2) ----------------------------------------------
with tab_live:
    model = load_model()
    st.caption(
        "Simula la llegada de las transacciones de producción hora a hora (1 step = 1 hora) "
        "y recalcula las métricas "
        "acumuladas en cada paso. "
        + (
            "Las transacciones se **puntúan en vivo** con el modelo campeón local."
            if model is not None
            else "Sin `models/champion.joblib` local se usan los scores precalculados "
            "de `data/production_scored.csv`."
        )
    )
    steps = sorted(production["step"].unique())
    s1, s2, s3 = st.columns([2, 1, 1])
    start, end = s1.select_slider(
        "Rango de steps", options=steps, value=(steps[0], steps[min(71, len(steps) - 1)])
    )
    delay = s2.slider("Segundos por step", 0.0, 1.0, 0.15, 0.05)
    run = s3.button("▶ Iniciar stream", type="primary", width="stretch")

    kpi_box, chart_box, alerts_box = st.empty(), st.empty(), st.empty()
    if run:
        stream = production[(production["step"] >= start) & (production["step"] <= end)]
        tp = fp = fn = tn = 0
        history = []
        for step, batch in stream.groupby("step", sort=True):
            if model is not None:
                batch = batch.assign(prediction=model.predict_proba(batch[MODEL_FEATURES])[:, 1])
            flagged = batch["prediction"].to_numpy() >= threshold
            fraud = batch["isFraud"].to_numpy() == 1
            tp += int((flagged & fraud).sum())
            fp += int((flagged & ~fraud).sum())
            fn += int((~flagged & fraud).sum())
            tn += int((~flagged & ~fraud).sum())
            recall = tp / (tp + fn) if tp + fn else np.nan
            precision = tp / (tp + fp) if tp + fp else np.nan
            history.append(
                {
                    "step": step,
                    "transacciones": len(batch),
                    "alertas": int(flagged.sum()),
                    "fraudes": int(fraud.sum()),
                    "recall": recall,
                    "precision": precision,
                }
            )
            hist = pd.DataFrame(history)

            with kpi_box.container():
                c = st.columns(5)
                c[0].metric(
                    "Step actual",
                    f"{step}",
                    f"día {(step - 1) // 24 + 1}, hora {(step - 1) % 24}",
                    delta_color="off",
                )
                c[1].metric("Transacciones procesadas", f"{tp + fp + fn + tn:,}")
                c[2].metric("Alertas acumuladas", f"{tp + fp:,}", f"{fp} falsas", delta_color="off")
                c[3].metric("Recall acumulado", "–" if np.isnan(recall) else f"{recall:.2%}")
                c[4].metric(
                    "Precision acumulada", "–" if np.isnan(precision) else f"{precision:.2%}"
                )
            fig = go.Figure()
            fig.add_bar(
                x=hist["step"],
                y=hist["fraudes"],
                name="Fraudes reales",
                marker_color=ORANGE,
                opacity=0.5,
            )
            fig.add_scatter(
                x=hist["step"],
                y=hist["alertas"],
                name="Alertas del modelo",
                mode="lines+markers",
                line={"color": BLUE, "width": 2},
            )
            fig.update_xaxes(title="step (hora)")
            chart_box.plotly_chart(
                base_layout(fig, "Fraudes reales vs alertas por hora", 300),
                width="stretch",
                key=f"live_{step}",
            )
            latest = batch.loc[
                flagged, ["step", "type", "amount", "oldbalanceOrg", "prediction", "isFraud"]
            ]
            alerts_box.dataframe(
                latest.sort_values("prediction", ascending=False).head(8),
                hide_index=True,
                width="stretch",
            )
            time.sleep(delay)
        st.success(
            f"Stream terminado: {len(history)} steps · recall {recall:.2%} · "
            f"precision {precision:.2%} · "
            f"{fn} fraudes no detectados · {fp} falsas alarmas"
        )
