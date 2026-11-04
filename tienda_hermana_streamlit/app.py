from __future__ import annotations

import hashlib
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from book_source import BookSourceError, download_onedrive_book
from core import (
    FEATURE_FAMILIES,
    WorkbookError,
    fit_te_family_weights,
    legacy_partial_score,
    read_jun_excel,
    rank_similar_stores,
    weighted_knn_forecast,
)

TYPE_OPTIONS = [
    "Esquina Positiva",
    "Esquina Negativa",
    "Semiesquina",
    "Medianera Doble Altura",
    "Medianera Simple",
    "Especial",
]
ACCESS_OPTIONS = ["Excelente", "Muy Buena", "Buena", "Regular", "Mala", "No Cumple"]
TRANSPARENCY_OPTIONS: list[object] = [0.9, 0.8, 0.7, 0.6, 0.5, "No Cumple"]
NO_SELECTION = "— Seleccionar —"

st.set_page_config(page_title="Tienda Hermana | Análisis de punto", page_icon="🏪", layout="wide")
st.markdown(
    """
    <style>
    :root {
      --oxxo-red: #d71920;
      --oxxo-red-dark: #b70f16;
      --oxxo-yellow: #ffd400;
      --oxxo-blue: #064c8c;
      --ink: #182434;
      --muted: #5d6877;
      --paper: #fffdfa;
    }
    .stApp {background: var(--paper); color: var(--ink);}
    .block-container {padding-top: 1.15rem; max-width: 1480px;}
    .brand-banner {
      background: linear-gradient(110deg, var(--oxxo-red), var(--oxxo-red-dark));
      border-radius: 16px 16px 6px 6px; padding: 1.15rem 1.5rem .9rem;
      color: #fff; border-bottom: 9px solid var(--oxxo-yellow); margin-bottom: 1.2rem;
      box-shadow: 0 5px 18px rgba(40,25,25,.12);
    }
    .brand-kicker {font-size: .78rem; text-transform: uppercase; letter-spacing: .16em; opacity: .9; font-weight: 700;}
    .brand-title {font-size: clamp(1.8rem, 4vw, 2.8rem); line-height: 1.05; font-weight: 850; margin-top: .2rem;}
    .brand-subtitle {font-size: 1rem; margin-top: .45rem; opacity: .95;}
    h1, h2, h3 {color: var(--oxxo-blue) !important;}
    [data-testid="stMetric"] {background: #fff; border: 1px solid #e7e9ed; border-top: 4px solid var(--oxxo-yellow); padding: 12px 16px; border-radius: 12px; box-shadow: 0 2px 10px rgba(30,40,55,.05);}
    [data-testid="stMetricValue"] {font-size: 2rem; font-weight: 800; color: var(--oxxo-blue);}
    [data-testid="stMetricLabel"] {color: var(--muted); font-weight: 650;}
    div[data-testid="stVerticalBlockBorderWrapper"] {border-color: #e5e7eb; border-radius: 14px; background: #fff;}
    div.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
      background: var(--oxxo-red); color: #fff; border: 0; border-radius: 9px; font-weight: 750; min-height: 2.8rem;
    }
    div.stButton > button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
      background: var(--oxxo-red-dark); color: #fff; border: 0;
    }
    [data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus {border-color: var(--oxxo-blue); box-shadow: 0 0 0 1px var(--oxxo-blue);}
    .step-chip {display:inline-block; background:var(--oxxo-yellow); color:#16243a; border-radius:999px; padding:.22rem .62rem; font-size:.78rem; font-weight:800; margin-bottom:.25rem;}
    .match-score {color:var(--oxxo-blue); font-size:.9rem; font-weight:700; background:#eef5fb; border-radius:8px; display:inline-block; padding:.35rem .55rem;}
    .big-sale-label {font-size:1rem; font-weight:750; color:var(--muted); letter-spacing:.05em;}
    .big-sale-value {font-size:clamp(2.8rem, 6vw, 5rem); font-weight:850; line-height:1.04; color:var(--oxxo-red); margin:.2rem 0 .45rem;}
    div[data-testid="stAlert"] {border-radius:10px;}
    </style>
    <div class="brand-banner">
      <div class="brand-kicker">Análisis comercial de ubicaciones</div>
      <div class="brand-title">TIENDA HERMANA</div>
      <div class="brand-subtitle">Compara el nuevo punto con tiendas abiertas y construye su propuesta</div>
    </div>
    """,
    unsafe_allow_html=True,
)


def display_number(value: object, decimals: int = 0) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return str(value)


def display_transparency(value: object) -> str:
    if value == "No Cumple":
        return "No Cumple (0 puntos)"
    try:
        points = {0.9: 5, 0.8: 4, 0.7: 3, 0.6: 2, 0.5: 1}
        numeric = float(value)
        score = next((p for v, p in points.items() if np.isclose(numeric, v)), "—")
        return f"{numeric:.0%} ({score} puntos)"
    except (TypeError, ValueError):
        return "—"


def optional_book_select(label: str, series: pd.Series, key: str) -> object | None:
    values: dict[str, object] = {}
    for value in series.dropna().tolist():
        label_value = str(value).strip()
        if label_value:
            values.setdefault(label_value, value)
    labels = [NO_SELECTION] + sorted(values, key=lambda item: item.casefold())
    chosen = st.selectbox(label, labels, key=key)
    return None if chosen == NO_SELECTION else values[chosen]


def optional_numeric(label: str, key: str, step: float = 1.0) -> float | None:
    value = st.number_input(
        label,
        min_value=0.0,
        value=None,
        step=step,
        format="%.2f",
        placeholder="Déjalo vacío si no conoces el dato",
        key=key,
    )
    return None if value is None else float(value)


def clean_target_value(value: object) -> object | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def candidate_label(cr: str, ranking: pd.DataFrame) -> str:
    match = ranking.loc[ranking["CR"].astype(str).eq(str(cr))]
    if match.empty:
        return str(cr)
    row = match.iloc[0]
    return f"{row['NAME']} · CR {row['CR']} · {row.get('MUN', '')} · {row['SIMILITUD']:.1f}%"


def score_table(project_score: dict[str, object], mirror_score: dict[str, object]) -> pd.DataFrame:
    labels = ["Tipo", "Fachada", "Transparencia", "Área piso de venta", "Accesibilidad"]
    def cell(value: object) -> str:
        return "—" if value is None or pd.isna(value) else str(value)

    rows = [
        {
            "Variable del formato": label,
            "Punto nuevo · puntos": cell(project_score["scores"].get(label)),
            "Tienda hermana · puntos": cell(mirror_score["scores"].get(label)),
        }
        for label in labels
    ]
    rows.append({
        "Variable del formato": "Subtotal parcial (máx. 25)",
        "Punto nuevo · puntos": f"{project_score['puntaje_parcial']} / 25 ({project_score['variables_con_puntaje']}/5 con lookup)",
        "Tienda hermana · puntos": f"{mirror_score['puntaje_parcial']} / 25 ({mirror_score['variables_con_puntaje']}/5 con lookup)",
    })
    return pd.DataFrame(rows)


def get_secret_book_url() -> str:
    try:
        return str(st.secrets.get("BOOK_ONEDRIVE_URL", "")).strip()
    except Exception:
        return ""


def clear_book_state() -> None:
    for key in [
        "_book_bytes", "_book_source_key", "_book_key", "book_dataframe", "model_weights",
        "analysis", "estimate_result", "analysis_counter",
    ]:
        st.session_state.pop(key, None)


# ------------------------------- Book source -------------------------------
secret_url = get_secret_book_url()
manual_upload = None
refresh_onedrive = False

with st.sidebar:
    st.markdown("## Fuente de datos")
    if secret_url:
        st.success("Book conectado a OneDrive")
        st.caption("Origen configurado en Streamlit Secrets. El enlace no se muestra en pantalla.")
        refresh_onedrive = st.button("Actualizar Book desde OneDrive", width="stretch")
        with st.expander("Usar un Excel manual por esta sesión"):
            manual_upload = st.file_uploader("Book alternativo (.xlsx)", type=["xlsx"], key="manual_book_override")
    else:
        st.warning("OneDrive aún no está configurado")
        st.caption("Agrega BOOK_ONEDRIVE_URL en Secrets de Streamlit o carga el archivo manualmente.")
        manual_upload = st.file_uploader("Cargar Book (.xlsx)", type=["xlsx"], key="manual_book")
    st.divider()
    st.caption("Se procesa la hoja JUN y únicamente tiendas con ESTADO = ABIERTA.")

if refresh_onedrive:
    clear_book_state()

try:
    if manual_upload is not None:
        file_bytes = manual_upload.getvalue()
        source_key = "upload:" + hashlib.sha256(file_bytes).hexdigest()
        source_name = manual_upload.name
    elif secret_url:
        source_key = "onedrive:" + hashlib.sha256(secret_url.encode("utf-8")).hexdigest()
        if st.session_state.get("_book_source_key") == source_key and "_book_bytes" in st.session_state:
            file_bytes = st.session_state["_book_bytes"]
        else:
            with st.spinner("Conectando con OneDrive y leyendo el Book..."):
                file_bytes = download_onedrive_book(secret_url)
            st.session_state["_book_bytes"] = file_bytes
            source_name = "Book de OneDrive"
    else:
        st.info("Carga un Book o configura OneDrive para iniciar la búsqueda.")
        st.stop()

    if st.session_state.get("_book_source_key") != source_key:
        clear_book_state()
        st.session_state["_book_source_key"] = source_key
        st.session_state["_book_bytes"] = file_bytes
    book_key = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state.get("_book_key") != book_key:
        for key in ["book_dataframe", "model_weights", "analysis", "estimate_result", "analysis_counter"]:
            st.session_state.pop(key, None)
        st.session_state["_book_key"] = book_key
    if "book_dataframe" not in st.session_state:
        with st.spinner("Validando la hoja JUN y filtrando tiendas abiertas..."):
            st.session_state["book_dataframe"] = read_jun_excel(file_bytes)
    stores = st.session_state["book_dataframe"]
except (BookSourceError, WorkbookError) as exc:
    st.error(str(exc))
    st.info("Para OneDrive, usa un enlace compartido con permisos de lectura que permita descargar el .xlsx. Si la organización exige inicio de sesión, el enlace solo no basta: carga el Excel manualmente o se requiere configurar Microsoft Graph con autenticación aprobada.")
    st.stop()
except Exception as exc:
    st.error(f"No pude cargar el Book ({type(exc).__name__}). Revisa el archivo, los permisos de OneDrive o las columnas de JUN.")
    st.stop()

sales_options = [column for column in ["VU6M", "VENTAS OUM"] if column in stores.columns]
if not sales_options:
    st.error("La hoja JUN no contiene VU6M ni VENTAS OUM para estimar ventas.")
    st.stop()

if "model_weights" not in st.session_state:
    with st.spinner("Preparando el modelo de similitud con las referencias históricas..."):
        st.session_state["model_weights"] = fit_te_family_weights(stores)
model_weights = st.session_state["model_weights"]

with st.container(border=True):
    source_label = "OneDrive" if secret_url and manual_upload is None else "archivo de esta sesión"
    st.markdown(f"**Book/JUN listo:** {len(stores):,} tiendas abiertas · Fuente: {source_label}".replace(",", "."))

st.markdown('<span class="step-chip">PASO 1</span>', unsafe_allow_html=True)
st.header("Ficha del punto nuevo")
st.caption("Llena los datos que tienes del punto. Los desplegables se alimentan de los valores que aparecen en las tiendas abiertas de Book.")

with st.form("new_point_form", clear_on_submit=False):
    project_name = st.text_input("Nombre del punto nuevo", key="project_name_input", placeholder="Ej.: Nuevo punto Centro")

    st.markdown("**Ubicación**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_department = optional_book_select("Departamento", stores["DEPARTAMENTO"], "target_department")
    with c2:
        target_municipality = optional_book_select("Municipio", stores["MUN"], "target_municipality")
    with c3:
        target_zone = optional_book_select("Zona", stores["ZONA"], "target_zone")
    with c4:
        target_upz = optional_book_select("UPZ / comuna", stores["UPZ/COMUNA"], "target_upz")

    st.markdown("**Perfil del entorno**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_stratum = optional_book_select("Estrato", stores["ESTRATO"], "target_stratum")
    with c2:
        target_generator = optional_book_select("Generador", stores["GENERADOR"], "target_generator")
    with c3:
        target_segment = optional_book_select("Segmento (SEG26)", stores["SEG26"], "target_segment")
    with c4:
        target_hunger = optional_book_select("Potencial hambre (PTH)", stores["PTH"], "target_pth")

    st.markdown("**Características del local**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_local_type = optional_book_select("Tipo de local", stores["TIPO DE LOCAL"], "target_local_type")
    with c2:
        target_parking = optional_book_select("Parqueadero", stores["PARQUEADERO"], "target_parking")
    with c3:
        target_open_24 = optional_book_select("Operación 24 horas", stores["24H"], "target_open_24")
    with c4:
        target_area = optional_numeric("Área total (m²)", "target_area")
    c1, c2, c3 = st.columns(3)
    with c1:
        target_sales_floor = optional_numeric("Piso de venta (m²)", "target_sales_floor")
    with c2:
        target_facade = optional_numeric("Fachada P", "target_facade")

    st.markdown("**Viviendas, empleos y tráfico externo**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_v100 = optional_numeric("Viviendas a 100 m (V100)", "target_v100")
    with c2:
        target_vt = optional_numeric("Viviendas totales (VT)", "target_vt")
    with c3:
        target_e100 = optional_numeric("Empleos a 100 m (E100)", "target_e100")
    with c4:
        target_et = optional_numeric("Empleos totales (ET)", "target_et")
    c1, c2, c3 = st.columns(3)
    with c1:
        target_tr15 = optional_numeric("Personas cada 15 min (TR15MIN)", "target_tr15")
    with c2:
        target_tv15 = optional_numeric("Vehículos cada 15 min (TV15MIN)", "target_tv15")
    with c3:
        target_tm15 = optional_numeric("Motos cada 15 min (TM15MIN)", "target_tm15")

    st.markdown("**Parámetros de venta**")
    c1, c2, c3 = st.columns(3)
    with c1:
        outcome = st.selectbox(
            "Venta a estimar",
            sales_options,
            index=sales_options.index("VU6M") if "VU6M" in sales_options else 0,
            format_func=lambda value: "Ventas últimos 6 meses (VU6M)" if value == "VU6M" else "Ventas último mes (VENTAS OUM)",
        )
    with c2:
        k_neighbors = st.number_input("Comparables para la venta (k)", min_value=1, max_value=25, value=10, step=1)
    with c3:
        min_coverage = st.number_input("Cobertura mínima", min_value=0.1, max_value=1.0, value=0.4, step=0.05, format="%.2f")
    unit_label = st.text_input("Unidad de Book", value="Unidad reportada en Book")

    search_clicked = st.form_submit_button("Buscar tienda hermana", type="primary", width="stretch")

if search_clicked:
    st.session_state.pop("analysis", None)
    st.session_state.pop("estimate_result", None)
    target = {
        "DEPARTAMENTO": target_department,
        "MUN": target_municipality,
        "ZONA": target_zone,
        "UPZ/COMUNA": target_upz,
        "ESTRATO": target_stratum,
        "GENERADOR": target_generator,
        "SEG26": target_segment,
        "PTH": target_hunger,
        "TIPO DE LOCAL": target_local_type,
        "PARQUEADERO": target_parking,
        "24H": target_open_24,
        "AREA": target_area,
        "PISO DE VENTA": target_sales_floor,
        "FACHADA P": target_facade,
        "V100": target_v100,
        "VT": target_vt,
        "E100": target_e100,
        "ET": target_et,
        "TR15MIN": target_tr15,
        "TV15MIN": target_tv15,
        "TM15MIN": target_tm15,
    }
    target = {key: clean_target_value(value) for key, value in target.items()}
    active = [column for columns in FEATURE_FAMILIES.values() for column in columns if target.get(column) is not None]
    active_families = sum(any(target.get(column) is not None for column in columns) for columns in FEATURE_FAMILIES.values())
    if not project_name.strip():
        st.error("Escribe el nombre del punto nuevo para mostrarlo al lado de su tienda hermana.")
    elif len(active) < 2 or active_families < 2:
        st.error("Completa al menos dos variables de Book en dos familias distintas antes de buscar.")
    else:
        try:
            with st.spinner("Buscando la tienda abierta más parecida..."):
                ranking = rank_similar_stores(
                    target, stores, family_weights=model_weights, min_coverage=float(min_coverage)
                )
                coverage_fallback = False
                if ranking.empty:
                    ranking = rank_similar_stores(
                        target, stores, family_weights=model_weights, min_coverage=0.0
                    )
                    coverage_fallback = not ranking.empty
            if ranking.empty:
                st.error("No encontré candidatas con datos comparables. Verifica las variables diligenciadas y que Book tenga valores en esos campos.")
            else:
                counter = int(st.session_state.get("analysis_counter", 0)) + 1
                st.session_state["analysis_counter"] = counter
                st.session_state["analysis"] = {
                    "id": counter,
                    "target": target,
                    "project_name": project_name.strip(),
                    "ranking": ranking,
                    "outcome": outcome,
                    "k": int(k_neighbors),
                    "unit": unit_label.strip() or "Unidad reportada en Book",
                    "coverage_fallback": coverage_fallback,
                    "requested_coverage": float(min_coverage),
                }
                st.success("¡Listo! Abajo aparecen el punto nuevo y la tienda hermana lado a lado.")
        except Exception as exc:
            st.error(f"No se pudo calcular la tienda hermana ({type(exc).__name__}). Intenta de nuevo; si continúa, revisa la conexión con Book.")

analysis = st.session_state.get("analysis")
if analysis:
    ranking = analysis["ranking"]
    analysis_id = analysis["id"]
    top_rows = ranking.head(20).copy()
    candidate_crs = top_rows["CR"].astype(str).tolist()
    selected_key = f"selected_mirror_cr_{analysis_id}"

    # First-ranked CR is the default. Alternatives remain available without
    # interrupting the requested side-by-side form.
    with st.expander("Ver otras tiendas hermanas similares (opcional)"):
        st.caption("La recomendación principal es la primera del ranking. CR/ID diferencia tiendas con nombres repetidos.")
        st.dataframe(
            top_rows[[column for column in ["CR", "NAME", "MUN", "SIMILITUD", "COBERTURA"] if column in top_rows]],
            width="stretch",
            hide_index=True,
        )
        selected_cr = st.selectbox(
            "Cambiar tienda hermana",
            candidate_crs,
            index=0,
            key=selected_key,
            format_func=lambda cr: candidate_label(cr, ranking),
        )

    if selected_key not in st.session_state:
        selected_cr = candidate_crs[0]
    else:
        selected_cr = str(st.session_state[selected_key])
    selected_rank = ranking.loc[ranking["CR"].astype(str).eq(str(selected_cr))].iloc[0]
    mirror_matches = stores.loc[stores["CR"].astype(str).eq(str(selected_cr))]
    if mirror_matches.empty:
        st.error("No pude volver a localizar la tienda hermana en el Book. Pulsa Buscar de nuevo.")
        st.stop()
    mirror = mirror_matches.iloc[0]

    st.markdown('<span class="step-chip">PASO 2</span>', unsafe_allow_html=True)
    st.header("Punto nuevo y tienda hermana")
    if analysis["coverage_fallback"]:
        st.warning(
            f"No hubo candidatas con la cobertura mínima solicitada ({analysis['requested_coverage']:.0%}). "
            f"Se muestra la mejor coincidencia disponible con {selected_rank['COBERTURA']:.0%} de cobertura."
        )

    st.markdown("Completa las variables **debajo de cada nombre**; las dos fichas quedan en paralelo.")
    with st.form(f"parallel_form_{analysis_id}_{selected_cr}"):
        new_col, mirror_col = st.columns(2, gap="large")
        with new_col:
            with st.container(border=True):
                st.markdown('<span class="step-chip">PUNTO NUEVO</span>', unsafe_allow_html=True)
                st.subheader(analysis["project_name"])
                st.caption("Proyecto que acabas de diligenciar")
            st.selectbox("Tipo", [NO_SELECTION] + TYPE_OPTIONS, key=f"new_type_{analysis_id}")
            new_transparency = st.selectbox(
                "Transparencia",
                [NO_SELECTION] + TRANSPARENCY_OPTIONS,
                key=f"new_transparency_{analysis_id}",
                format_func=lambda value: NO_SELECTION if value == NO_SELECTION else display_transparency(value),
            )
            new_access = st.selectbox("Accesibilidad", [NO_SELECTION] + ACCESS_OPTIONS, key=f"new_access_{analysis_id}")
        with mirror_col:
            with st.container(border=True):
                st.markdown('<span class="step-chip">TIENDA HERMANA</span>', unsafe_allow_html=True)
                st.subheader(str(mirror["NAME"]))
                st.caption(f"CR / ID: {mirror['CR']} · {mirror.get('MUN', '')}")
                st.markdown(f'<span class="match-score">Similitud {selected_rank["SIMILITUD"]:.1f}% · Cobertura {selected_rank["COBERTURA"]:.0%}</span>', unsafe_allow_html=True)
            mirror_type = st.selectbox("Tipo ", [NO_SELECTION] + TYPE_OPTIONS, key=f"mirror_type_{analysis_id}_{selected_cr}")
            mirror_transparency = st.selectbox(
                "Transparencia ",
                [NO_SELECTION] + TRANSPARENCY_OPTIONS,
                key=f"mirror_transparency_{analysis_id}_{selected_cr}",
                format_func=lambda value: NO_SELECTION if value == NO_SELECTION else display_transparency(value),
            )
            mirror_access = st.selectbox("Accesibilidad ", [NO_SELECTION] + ACCESS_OPTIONS, key=f"mirror_access_{analysis_id}_{selected_cr}")
        estimate_clicked = st.form_submit_button("Estimar venta", type="primary", width="stretch")

    if estimate_clicked:
        new_type = st.session_state.get(f"new_type_{analysis_id}", NO_SELECTION)
        mirror_type = st.session_state.get(f"mirror_type_{analysis_id}_{selected_cr}", NO_SELECTION)
        required_values = [new_type, new_transparency, new_access, mirror_type, mirror_transparency, mirror_access]
        if any(value == NO_SELECTION for value in required_values):
            st.error("Completa Tipo, Transparencia y Accesibilidad en ambas columnas antes de estimar.")
        else:
            project_legacy = {"tipo": new_type, "transparencia": new_transparency, "accesibilidad": new_access}
            mirror_legacy = {"tipo": mirror_type, "transparencia": mirror_transparency, "accesibilidad": mirror_access}
            with st.spinner("Calculando la estimación de venta..."):
                forecast, neighbors = weighted_knn_forecast(
                    ranking, stores, analysis["outcome"], k=analysis["k"], include_cr=selected_cr
                )
            if not forecast:
                st.error(f"No hay ventas {analysis['outcome']} disponibles entre los comparables. Prueba otra variable de venta o revisa Book.")
            else:
                st.session_state["estimate_result"] = {
                    "analysis_id": analysis_id,
                    "selected_cr": str(selected_cr),
                    "project_points": legacy_partial_score(analysis["target"], project_legacy),
                    "mirror_points": legacy_partial_score(mirror, mirror_legacy),
                    "forecast": forecast,
                    "neighbors": neighbors,
                }

    result = st.session_state.get("estimate_result")
    if result and result.get("analysis_id") == analysis_id and result.get("selected_cr") == str(selected_cr):
        st.divider()
        st.markdown('<span class="step-chip">PASO 3</span>', unsafe_allow_html=True)
        st.header("Estimación y resumen")
        estimate = result["forecast"]["estimacion_ponderada"]
        st.markdown('<div class="big-sale-label">VENTA ESTIMADA DEL PUNTO NUEVO</div>', unsafe_allow_html=True)
        safe_unit = html.escape(str(analysis["unit"]))
        st.markdown(f'<div class="big-sale-value">{display_number(estimate)} <span style="font-size:1.15rem;font-weight:650">{safe_unit}</span></div>', unsafe_allow_html=True)
        st.caption(
            f"{analysis['outcome']} · promedio ponderado de {result['forecast']['n_vecinos']} tiendas comparables, "
            f"incluyendo {mirror['NAME']} (CR {mirror['CR']})."
        )

        st.subheader("Puntajes de la ficha histórica")
        st.dataframe(score_table(result["project_points"], result["mirror_points"]), width="stretch", hide_index=True)
        st.caption(
            "Los puntos replican los lookups exactos del formato anterior. Fachada y Área de piso de venta solo puntúan si el valor de Book coincide exactamente; "
            "el subtotal es parcial y no representa los 12 rubros completos."
        )

        st.subheader("Tiendas que forman la estimación")
        st.write(
            f"Se promedian las {result['forecast']['n_vecinos']} ventas disponibles con peso inverso a distancia: "
            "1/(distancia + 0,025)². La tabla muestra cuánto aporta cada tienda."
        )
        neighbors = result["neighbors"].copy()
        neighbors["Peso en estimación (%)"] = neighbors["PESO_RELATIVO"] * 100
        neighbors["Aporte a la venta"] = neighbors["APORTE_ESTIMACION"]
        columns = ["CR", "NAME", "MUN", "SIMILITUD", "COBERTURA", "VENTA_OBJETIVO", "Peso en estimación (%)", "Aporte a la venta"]
        st.dataframe(neighbors[[column for column in columns if column in neighbors]], width="stretch", hide_index=True)
        st.caption(
            f"Mediana de comparables: {display_number(result['forecast']['mediana_vecinos'])}. "
            f"P10–P90 observado: {display_number(result['forecast']['p10_vecinos'])}–{display_number(result['forecast']['p90_vecinos'])}; "
            "es dispersión del grupo, no intervalo de confianza."
        )
        st.info(
            "Los puntos de Transparencia y Accesibilidad aparecen en el resumen, pero no ajustan la venta: Book no contiene esos atributos de forma consistente para entrenar su efecto."
        )
    elif result and result.get("analysis_id") == analysis_id:
        st.info("Cambiaste la tienda hermana. Revisa sus datos en la columna derecha y pulsa **Estimar venta** otra vez.")
