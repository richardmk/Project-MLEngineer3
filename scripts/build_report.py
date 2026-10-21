"""Build the final self-contained HTML report and the README figures from executed notebooks.

Figures are taken from the notebooks' saved outputs (no recomputation); tables come from
the CSVs in reports/. Run after executing notebooks 01-04:

    uv run python scripts/build_report.py
"""

import base64
import html
import json
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# (notebook, text that identifies the code cell, image index in that cell, figure id, caption)
FIGURE_SPECS = [
    (
        "01_preprocesamiento",
        "El fraude solo ocurre en TRANSFER y CASH_OUT",
        0,
        "fraud_by_type",
        "El fraude solo ocurre en TRANSFER y CASH_OUT.",
    ),
    (
        "01_preprocesamiento",
        "Volumen de transacciones por step",
        0,
        "temporal",
        "Volumen, fraudes y tasa de fraude por step: el fraude es ~constante y el volumen legítimo cae tras el step 400.",
    ),
    (
        "02_entrenamiento",
        "mejor PR-AUC acumulado vs tiempo",
        0,
        "tpe_vs_random",
        "Optuna TPE + poda vs Random Search: mismo valor final, distinto costo de cómputo.",
    ),
    (
        "02_entrenamiento",
        "Curva PR out-of-fold",
        0,
        "pr_oof",
        "Curvas PR out-of-fold de los 4 algoritmos frente a la regla is_full_drain.",
    ),
    (
        "02_entrenamiento",
        "Costo de negocio vs threshold",
        0,
        "cost_curve",
        "Costo de negocio vs threshold: plano en un rango amplio; el threshold F2 cae en el centro del óptimo.",
    ),
    (
        "02_entrenamiento",
        "fig_pr = plot_pr_curve",
        1,
        "confusion_test",
        "Matriz de confusión del campeón en test: 955 de 956 fraudes, 0 falsos positivos.",
    ),
    (
        "03_interpretabilidad",
        "shap.plots.beeswarm",
        0,
        "shap_summary",
        "SHAP summary: is_full_drain domina; un origen que cuadra y un destino vacío empujan hacia fraude.",
    ),
    (
        "03_interpretabilidad",
        "PDP (naranja) e ICE",
        0,
        "pdp_ice",
        "PDP e ICE de las 4 features principales.",
    ),
    (
        "03_interpretabilidad",
        "shap.plots.waterfall",
        1,
        "waterfall_rule_miss",
        "SHAP waterfall de un fraude que la regla no detecta (monto topeado en 10 M).",
    ),
    (
        "04_monitoreo",
        "Drift score por feature y día",
        0,
        "drift_heatmap",
        "Data drift por feature y día de producción (Evidently).",
    ),
    (
        "04_monitoreo",
        "panels = [",
        0,
        "concept_drift",
        "Prior shift, prediction drift y concepto: la relación vaciado → fraude se mantiene.",
    ),
    (
        "04_monitoreo",
        'metrics = [("pr_auc"',
        0,
        "online_metrics",
        "Métricas online por día vs offline (test).",
    ),
]
README_FIGURES = {"shap_summary", "drift_heatmap", "online_metrics"}


def cell_images(notebook: str, marker: str) -> list[str]:
    """Base64 PNGs produced by the first code cell whose source contains `marker`."""
    nb = nbformat.read(NB / f"{notebook}.ipynb", as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code" and marker in cell.source:
            return [
                o["data"]["image/png"] for o in cell.outputs if "image/png" in o.get("data", {})
            ]
    raise ValueError(f"No cell with {marker!r} in {notebook}")


def table(df: pd.DataFrame, floatfmt: str = "{:,.4f}") -> str:
    """Render a DataFrame as a plain HTML table."""

    def fmt(v: object) -> str:
        if isinstance(v, float):
            return floatfmt.format(v)
        if isinstance(v, int):
            return f"{v:,}"
        return html.escape(str(v))

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{fmt(v)}</td>" for v in r) + "</tr>"
        for r in df.itertuples(index=False)
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def figure(fig_id: str, images: dict[str, str], captions: dict[str, str]) -> str:
    """HTML figure with an embedded PNG."""
    return (
        f'<figure><img alt="{html.escape(captions[fig_id])}" src="data:image/png;base64,{images[fig_id]}">'
        f"<figcaption>{html.escape(captions[fig_id])}</figcaption></figure>"
    )


CSS = """
:root { --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9;
        --accent:#2a78d6; --warn:#eb6834; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#0d0d0d; --surface:#1a1a19;
        --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --line:#2c2c2a; --accent:#3987e5; --warn:#d95926; } }
:root[data-theme="dark"] { --bg:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
        --line:#2c2c2a; --accent:#3987e5; --warn:#d95926; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 1.9rem; line-height:1.2; margin: 0 0 4px; }
h2 { font-size: 1.3rem; margin: 40px 0 12px; padding-top: 12px; border-top: 1px solid var(--line); }
p, li { color: var(--ink2); }
.subtitle { color: var(--muted); margin: 0 0 24px; }
.kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin: 20px 0; }
.kpi { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.kpi .v { font-size:1.6rem; font-weight:700; color:var(--ink); font-variant-numeric: tabular-nums; }
.kpi .l { font-size:.85rem; color:var(--muted); }
.note { background:var(--surface); border-left:4px solid var(--warn); padding:12px 16px; border-radius:6px; }
figure { margin: 18px 0; background:#fcfcfb; border:1px solid var(--line); border-radius:10px; padding:10px; }
figure img { width:100%; height:auto; display:block; }
figcaption { color:#52514e; font-size:.9rem; padding: 6px 4px 0; }
.table-wrap { overflow-x:auto; margin: 12px 0; }
table { border-collapse: collapse; width:100%; font-size:.9rem; font-variant-numeric: tabular-nums; }
th, td { text-align:right; padding:6px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
th { color:var(--muted); font-weight:600; }
a { color: var(--accent); }
"""


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    images, captions = {}, {}
    for notebook, marker, idx, fig_id, caption in FIGURE_SPECS:
        images[fig_id] = cell_images(notebook, marker)[idx]
        captions[fig_id] = caption
        if fig_id in README_FIGURES:
            (FIGURES / f"{fig_id}.png").write_bytes(base64.b64decode(images[fig_id]))

    info = json.loads((REPORTS / "evidently" / "model_info.json").read_text())
    study = pd.read_csv(REPORTS / "optimization" / "study_summary.csv").rename(
        columns={"Unnamed: 0": "estudio"}
    )
    sanity = pd.read_csv(REPORTS / "optimization" / "sanity_checks.csv")[
        ["experimento", "n_features", "cv_pr_auc"]
    ]
    test = pd.read_csv(REPORTS / "optimization" / "test_results.csv").rename(
        columns={"Unnamed: 0": "modelo"}
    )
    monitoring = pd.read_csv(REPORTS / "evidently" / "monitoring_summary.csv")
    tp, fn, fp = (int(monitoring[c].sum()) for c in ["tp", "fn", "fp"])
    online_recall = tp / (tp + fn)
    daily = monitoring[
        [
            "period",
            "transacciones",
            "fraudes",
            "fraud_rate",
            "features_drift_share",
            "share_frauds_drain",
            "precision",
            "recall",
            "f2",
            "fn",
        ]
    ].rename(columns={"period": "día"})
    t = test.iloc[0]

    body = f"""
<main>
<h1>Detección de fraude en transacciones móviles (PaySim)</h1>
<p class="subtitle">Proyecto final · Curso III, Especialización en Machine Learning Engineering · release v1.0.0</p>

<div class="kpis">
  <div class="kpi"><div class="v">{t["pr_auc"]:.5f}</div><div class="l">PR-AUC en test (offline)</div></div>
  <div class="kpi"><div class="v">{int(t["tp"])}/{int(t["tp"] + t["fn"])}</div><div class="l">fraudes detectados en test</div></div>
  <div class="kpi"><div class="v">{int(t["fp"])}</div><div class="l">falsos positivos en test</div></div>
  <div class="kpi"><div class="v">{online_recall:.2%}</div><div class="l">recall online (producción)</div></div>
  <div class="kpi"><div class="v">{fp}</div><div class="l">falsas alarmas en producción</div></div>
</div>

<h2>1. Problema</h2>
<p>Clasificación binaria supervisada de <code>isFraud</code> sobre la simulación PaySim (1 step = 1 hora).
Steps ≤ 400: referencia/entrenamiento (train 1–312, test 313–400). Steps ≥ 401: producción, solo para monitoreo.
Tasa de fraude 0.129 %, presente solo en TRANSFER y CASH_OUT, que son el alcance del modelo.</p>
{figure("fraud_by_type", images, captions)}
{figure("temporal", images, captions)}

<h2>2. Entrenamiento y optimización</h2>
<p>CV <em>expanding window</em> por días completos, submuestreo de legítimos solo en train, Optuna TPE con
MedianPruner frente a Random Search, y un estudio AutoML con el algoritmo como hiperparámetro. Todo registrado en
<a href="https://dagshub.com/richardmk/ProjectMLEngineer3/experiments">MLflow (DagsHub)</a>.</p>
{table(study)}
{figure("tpe_vs_random", images, captions)}
<p class="note"><strong>¿Por qué el PR-AUC es casi 1?</strong> No es leakage: con labels barajados el modelo cae al
azar. El agente fraudulento de PaySim vacía la cuenta, y esa sola regla ya logra un PR-AUC de 0.975.</p>
{table(sanity)}
{figure("pr_oof", images, captions)}
{figure("cost_curve", images, captions)}

<h2>3. Resultados offline (test)</h2>
<p>Campeón: <strong>{info["model_name"]}</strong>, registrado como <code>{info["registered_model"]}</code>
v{info["version"]} (alias <code>{info["alias"]}</code>), threshold {info["threshold"]:.4f}.</p>
{table(test[["modelo", "pr_auc", "roc_auc", "precision", "recall", "f1", "f2", "tp", "fp", "fn"]])}
{figure("confusion_test", images, captions)}

<h2>4. Interpretabilidad</h2>
<p>SHAP, permutation importance y PDP/ICE coinciden: <code>is_full_drain</code> es la señal dominante, seguida de
la consistencia del saldo del origen y del saldo del destino (cuentas mula vacías). El valor del modelo sobre la regla
está en casos como los montos topeados en 10 M.</p>
{figure("shap_summary", images, captions)}
{figure("pdp_ice", images, captions)}
{figure("waterfall_rule_miss", images, captions)}

<h2>5. Monitoreo en producción</h2>
<p>Quince periodos diarios con reportes Evidently. Hay data drift y prediction drift fuertes, causados por un prior
shift de ~10×, pero <strong>no hay concept drift ni model drift</strong>: precision 1.0 todos los días y recall online
de {online_recall:.2%} ({tp:,} de {tp + fn:,} fraudes).</p>
{figure("drift_heatmap", images, captions)}
{figure("concept_drift", images, captions)}
{figure("online_metrics", images, captions)}
{table(daily)}

<h2>6. Conclusiones</h2>
<ul>
<li>El fraude de PaySim es casi determinista (vaciar la cuenta); se verificó que el desempeño casi perfecto no es leakage.</li>
<li>El modelo supera a la regla en los casos frontera sin agregar falsas alarmas.</li>
<li>La optimización está saturada: la ventaja de TPE con poda es de eficiencia (−21 % de cómputo en XGBoost).</li>
<li>El modelo es robusto al prior shift y al drift de features observados; el concepto no cambió.</li>
<li>Indicador clave a vigilar: el % de fraudes que vacían la cuenta (alerta temprana de cambio de táctica).</li>
</ul>
<p class="subtitle">Código: <a href="https://github.com/richardmk/Project-MLEngineer3">github.com/richardmk/Project-MLEngineer3</a></p>
</main>"""

    page = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fraude PaySim · Report</title><style>{CSS}</style></head><body>{body}</body></html>"""
    out = REPORTS / "report.html"
    out.write_text(page, encoding="utf-8")
    print(
        f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB) and {len(README_FIGURES)} figures in {FIGURES}"
    )


if __name__ == "__main__":
    main()
