from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from core import (
    DEFAULT_FAMILY_WEIGHTS,
    FEATURE_FAMILIES,
    WorkbookError,
    backtest_sales_loocv,
    backtest_te,
    legacy_partial_score,
    read_jun_excel,
    rank_similar_stores,
    weighted_knn_forecast,
)


st.set_page_config(page_title="Tienda Hermana | Modelo de comparables", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.3rem; max-width: 1500px;}
    [data-testid="stMetric"] {background: #f4f7f7; border: 1px solid #e2eaea; padding: 12px 16px; border-radius: 10px;}
    div[data-testid="stAlert"] {border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def display_number(value: object, decimals: int = 1) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return str(value)


def optional_category(label: str, values: list[object], key: str) -> object | None:
    cleaned = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    options = ["No informado"] + sorted(cleaned, key=lambda x: x.casefold())
    chosen = st.selectbox(label, options, key=key)
    return None if chosen == "No informado" else chosen


def optional_numeric(label: str, key: str, maximum: float | None = None) -> float | None:
    return st.number_input(
        label,
        min_value=0.0,
        max_value=maximum,
        value=None,
        step=1.0,
        format="%.2f",
        placeholder="Dejar vacío si no se conoce",
        key=key,
    )


def clean_target_value(value: object) -> object | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def render_candidate_table(ranking: pd.DataFrame, stores: pd.DataFrame, outcome: str, limit: int = 15) -> pd.DataFrame:
    if ranking.empty:
        return pd.DataFrame()
    details = [
        "CR", "AREA", "PISO DE VENTA", "FACHADA P", "ESTRATO", "TIPO DE LOCAL", "PARQUEADERO", "24H",
        "V100", "VT", "E100", "ET", "TR15MIN", "TV15MIN", "TM15MIN", "GENERADOR", "SEG26", "TIE26", "PTH",
        "VENTAS OUM", "VU6M", "TE",
    ]
    available = [column for column in details if column in stores.columns]
    joined = ranking.merge(stores[available], on="CR", how="left", suffixes=("", "_BOOK"))
    show = [
        "CR", "NAME", "MUN", "SIMILITUD", "COBERTURA", "VARIABLES_COMPARADAS", "SEG26", "TIE26",
        "TE", "TIPO DE LOCAL", "AREA", "PISO DE VENTA", "FACHADA P", "ESTRATO", "V100", "VT", "E100", "ET",
        "TR15MIN", "TV15MIN", "TM15MIN", outcome,
    ]
    return joined[[column for column in show if column in joined.columns]].head(limit).copy()


def make_export_rows(
    project_name: str,
    target: dict[str, object],
    project_legacy: dict[str, object],
    mirror: pd.Series,
    mirror_legacy: dict[str, object],
    selected_rank: pd.Series,
    forecast: dict[str, object],
    outcome: str,
    unit_label: str,
) -> pd.DataFrame:
    mapping = {
        "Tienda / punto": "NAME",
        "CR": "CR",
        "Departamento": "DEPARTAMENTO",
        "Municipio": "MUN",
        "Zona": "ZONA",
        "UPZ / comuna": "UPZ/COMUNA",
        "Estrato": "ESTRATO",
        "Tipo de local (Book)": "TIPO DE LOCAL",
        "Parqueadero": "PARQUEADERO",
        "Área total (m²)": "AREA",
        "Piso de venta (m²)": "PISO DE VENTA",
        "Fachada P (Book)": "FACHADA P",
        "Viviendas a 100 m (V100)": "V100",
        "Viviendas totales (VT)": "VT",
        "Empleos a 100 m (E100)": "E100",
        "Empleos totales (ET)": "ET",
        "Personas cada 15 min (TR15MIN)": "TR15MIN",
        "Vehículos cada 15 min (TV15MIN)": "TV15MIN",
        "Motos cada 15 min (TM15MIN)": "TM15MIN",
        "24 horas": "24H",
        "Generador (Book)": "GENERADOR",
        "Segmento (SEG26)": "SEG26",
        "Generación (TIE26)": "TIE26",
        "Potencial hambre (PTH)": "PTH",
    }
    rows = []
    for side, record, legacy, code, name, score, coverage in [
        ("Proyecto nuevo", target, project_legacy, "", project_name, np.nan, np.nan),
        (
            "Tienda hermana seleccionada", mirror, mirror_legacy, mirror.get("CR", ""), mirror.get("NAME", ""),
            selected_rank.get("SIMILITUD", np.nan), selected_rank.get("COBERTURA", np.nan),
        ),
    ]:
        legacy_score = legacy_partial_score(record, legacy)
        item = {"Lado": side, "CR": code, "Tienda / punto": name}
        for label, source_col in mapping.items():
            if label == "Tienda / punto":
                continue
            item[label] = clean_target_value(record.get(source_col))
        item["Tipo (formato histórico)"] = legacy.get("tipo", "")
        item["Transparencia (captura manual)"] = legacy.get("transparencia")
        item["Accesibilidad (captura manual)"] = legacy.get("accesibilidad", "")
        item["Puntos históricos · Tipo"] = legacy_score["scores"]["Tipo"]
        item["Puntos históricos · Fachada"] = legacy_score["scores"]["Fachada"]
        item["Puntos históricos · Transparencia"] = legacy_score["scores"]["Transparencia"]
        item["Puntos históricos · Área piso venta"] = legacy_score["scores"]["Área piso de venta"]
        item["Puntos históricos · Accesibilidad"] = legacy_score["scores"]["Accesibilidad"]
        item["Subtotal histórico (parcial)"] = legacy_score["puntaje_parcial"]
        item["Variables con puntaje (de 5)"] = legacy_score["variables_con_puntaje"]
        item["Puntaje de similitud (%)"] = score
        item["Cobertura del puntaje (%)"] = coverage * 100 if pd.notna(coverage) else np.nan
        item["Variable estimada"] = outcome
        item["Unidad declarada"] = unit_label
        item["Venta estimada ponderada"] = forecast.get("estimacion_ponderada")
        item["Mediana ventas vecinos"] = forecast.get("mediana_vecinos")
        item["P10 ventas vecinos"] = forecast.get("p10_vecinos")
        item["P90 ventas vecinos"] = forecast.get("p90_vecinos")
        item["Cantidad vecinos"] = forecast.get("n_vecinos")
        rows.append(item)
    return pd.DataFrame(rows)


st.title("Tienda Hermana")
st.caption("Ranking estadístico de comparables y estimación de venta por vecinos similares — hoja Book/JUN")
st.write(
    "Carga el Book, ingresa las características conocidas del punto propuesto y revisa las tiendas abiertas más comparables. "
    "El archivo empresarial se procesa en esta sesión; no viene incluido en el proyecto."
)

with st.sidebar:
    st.header("1. Base de datos")
    uploaded = st.file_uploader("Sube el archivo Book (.xlsx)", type=["xlsx"], help="El libro debe incluir la hoja JUN.")
    st.divider()

if not uploaded:
    st.info("Para iniciar, sube el archivo Book que contiene la hoja JUN.")
    st.markdown(
        "**El modelo utilizará solo `ESTADO = ABIERTA`.** No usará `%Batting` ni `ME`; ventas, tráfico de tienda, contribución, margen y renta no se emplean para elegir el espejo."
    )
    st.markdown("La tienda espejo histórica `TE` se reserva para contrastar las selecciones pasadas, no entra como variable del ranking.")
    st.stop()

file_bytes = uploaded.getvalue()
book_key = hashlib.sha256(file_bytes).hexdigest()
if st.session_state.get("_book_key") != book_key:
    for key in ["analysis", "te_validation", "sales_validation", "mirror_legacy_by_cr", "book_dataframe"]:
        st.session_state.pop(key, None)
    st.session_state["_book_key"] = book_key

try:
    if "book_dataframe" not in st.session_state:
        st.session_state["book_dataframe"] = read_jun_excel(file_bytes)
    stores = st.session_state["book_dataframe"]
except WorkbookError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"No pude procesar el Excel: {exc}")
    st.stop()

sales_columns = [column for column in ["VENTAS OUM", "VU6M"] if column in stores.columns]
if not sales_columns:
    st.error("JUN no tiene las columnas de ventas requeridas.")
    st.stop()

with st.sidebar:
    st.header("2. Ajustes del modelo")
    st.caption("Pesos iniciales iguales por familia; son ajustables, no se presentan como pesos aprendidos.")
    weights = {}
    for family, default in DEFAULT_FAMILY_WEIGHTS.items():
        weights[family] = st.slider(f"Peso · {family}", 0.0, 3.0, float(default), 0.1, key=f"weight_{family}")
    min_coverage = st.slider("Cobertura mínima de datos", 0.10, 1.00, 0.40, 0.05)
    k_neighbors = st.slider("Número de tiendas para estimar ventas (k)", 3, 25, 10, 1)
    outcome = st.selectbox("Venta a estimar", sales_columns, index=sales_columns.index("VU6M") if "VU6M" in sales_columns else 0)
    unit_label = st.text_input("Unidad de venta (sin conversión)", value="Unidad reportada en Book")

open_count = len(stores)
valid_oum = int(pd.to_numeric(stores["VENTAS OUM"], errors="coerce").notna().sum()) if "VENTAS OUM" in stores else 0
valid_vu6m = int(pd.to_numeric(stores["VU6M"], errors="coerce").notna().sum()) if "VU6M" in stores else 0
st.success(f"Book listo: {open_count} tiendas abiertas y {valid_oum} ventas OUM / {valid_vu6m} VU6M disponibles.")

with st.expander("Alcance, exclusiones y limitaciones de los datos", expanded=False):
    st.markdown(
        "- Universo y comparables: únicamente tiendas `ABIERTA`; `CR` se muestra como identificador para diferenciar nombres repetidos.\n"
        "- Ranking: distancia tipo Gower agrupada por familia. Numéricas: diferencia normalizada con rango robusto; categorías: coincidencia/no coincidencia. Los faltantes se omiten y reducen la cobertura.\n"
        "- Para el ranking se usan ubicación, perfil/segmentación declarada, infraestructura, hogares/empleos y tráfico externo de 15 minutos. No se usan las ventas ni variables de desempeño para buscar el espejo.\n"
        "- El estimador de venta es una media ponderada por similitud de los `k` vecinos; también muestra mediana y percentiles 10–90 observados en esos vecinos. Es un benchmark de análogos, no garantía ni intervalo de confianza.\n"
        "- `TE` se usa solo para backtest de concordancia con las asignaciones históricas enlazables. Coincidir con TE no prueba causalidad ni que la selección histórica fuera óptima.\n"
        "- JUN no tiene transparencia ni accesibilidad: esos dos campos se capturan manualmente en la ficha paralela. El sistema no infiere su efecto estadístico.\n"
        "- No se interpreta la unidad monetaria de ventas; el resultado conserva el valor de Book y la etiqueta que ingreses."
    )

st.subheader("Punto nuevo")
with st.form("new_site_form"):
    project_name = st.text_input("Nombre propuesto", placeholder="Nombre del proyecto/punto")
    loc_tab, profile_tab, infra_tab, market_tab, flow_tab, legacy_tab = st.tabs(
        ["Ubicación", "Perfil", "Infraestructura", "Hogares y empleo", "Tráfico externo", "Ficha histórica"]
    )
    target_values: dict[str, object] = {}

    with loc_tab:
        c1, c2 = st.columns(2)
        with c1:
            target_values["DEPARTAMENTO"] = c1.text_input("Departamento", key="target_department").strip() or None
            target_values["MUN"] = c1.text_input("Municipio", key="target_municipality").strip() or None
        with c2:
            target_values["ZONA"] = c2.text_input("Zona", key="target_zone").strip() or None
            target_values["UPZ/COMUNA"] = c2.text_input("UPZ / comuna", key="target_upz").strip() or None
        st.caption("Se puede ingresar una ciudad aunque aún no exista en Book; será tratada como categoría diferente.")

    with profile_tab:
        c1, c2, c3 = st.columns(3)
        with c1:
            target_values["ESTRATO"] = optional_numeric("Estrato", "target_stratum", maximum=6.0)
            target_values["GENERADOR"] = optional_category("Generador", stores["GENERADOR"].dropna().tolist(), "target_generator")
        with c2:
            target_values["SEG26"] = optional_category("Segmento (SEG26)", stores["SEG26"].dropna().tolist(), "target_segment")
            target_values["TIE26"] = optional_category("Generación (TIE26)", stores["TIE26"].dropna().tolist(), "target_generation")
        with c3:
            target_values["PTH"] = optional_category("Potencial hambre (PTH)", stores["PTH"].dropna().tolist(), "target_pth")
        st.caption("Las categorías se comparan como etiquetas; el sistema no asigna significado adicional a cada código.")

    with infra_tab:
        c1, c2, c3 = st.columns(3)
        with c1:
            target_values["TIPO DE LOCAL"] = optional_category("Tipo de local (Book)", stores["TIPO DE LOCAL"].dropna().tolist(), "target_local_type")
            target_values["PARQUEADERO"] = optional_category("¿Tiene parqueadero?", stores["PARQUEADERO"].dropna().tolist(), "target_parking")
        with c2:
            target_values["AREA"] = optional_numeric("Área total (m²)", "target_area")
            target_values["PISO DE VENTA"] = optional_numeric("Piso de venta (m²)", "target_sales_floor")
        with c3:
            target_values["FACHADA P"] = optional_numeric("Fachada P (unidad de Book)", "target_facade")
            target_values["24H"] = optional_category("Operación 24 horas", stores["24H"].dropna().tolist(), "target_24h")
        st.caption("La fachada se presenta tal como está en JUN; confirma su unidad antes de convertirla o compararla con el formato histórico.")

    with market_tab:
        c1, c2 = st.columns(2)
        with c1:
            target_values["V100"] = optional_numeric("Viviendas a 100 m (V100)", "target_v100")
            target_values["VT"] = optional_numeric("Viviendas totales (VT)", "target_vt")
        with c2:
            target_values["E100"] = optional_numeric("Empleos a 100 m (E100)", "target_e100")
            target_values["ET"] = optional_numeric("Empleos totales (ET)", "target_et")

    with flow_tab:
        c1, c2, c3 = st.columns(3)
        with c1:
            target_values["TR15MIN"] = optional_numeric("Personas cada 15 min (TR15MIN)", "target_tr15")
        with c2:
            target_values["TV15MIN"] = optional_numeric("Vehículos cada 15 min (TV15MIN)", "target_tv15")
        with c3:
            target_values["TM15MIN"] = optional_numeric("Motos cada 15 min (TM15MIN)", "target_tm15")

    with legacy_tab:
        st.caption("Estos atributos se guardan para completar la ficha en paralelo. Transparencia y accesibilidad no existen en JUN y deben capturarse en campo.")
        c1, c2, c3 = st.columns(3)
        with c1:
            project_legacy_type = c1.selectbox(
                "Tipo (formato histórico)",
                ["No definido", "Esquina Positiva", "Esquina Negativa", "Semiesquina", "Medianera Doble Altura", "Medianera Simple", "Especial"],
                key="project_legacy_type",
            )
        with c2:
            project_transparency = c2.number_input(
                "Transparencia (formato histórico)", min_value=0.0, max_value=1.0, value=None, step=0.05,
                format="%.2f", placeholder="0–1", key="project_transparency",
            )
        with c3:
            project_accessibility = c3.selectbox(
                "Accesibilidad (formato histórico)",
                ["No definida", "Excelente", "Muy Buena", "Buena", "Regular", "Mala", "No Cumple"],
                key="project_accessibility",
            )

    submitted = st.form_submit_button("Buscar tienda hermana y estimar venta", type="primary", use_container_width=True)

if submitted:
    clean_target = {key: clean_target_value(value) for key, value in target_values.items()}
    active_features = [
        column for family_features in FEATURE_FAMILIES.values()
        for column in family_features if column in clean_target and clean_target[column] is not None
    ]
    active_families = {
        family for family, columns in FEATURE_FAMILIES.items()
        if any(column in active_features for column in columns) and weights.get(family, 0.0) > 0
    }
    if len(active_features) < 2 or len(active_families) < 2:
        st.error("Ingresa al menos 2 variables conocidas en 2 familias (por ejemplo, municipio + área + viviendas/empleos).")
    else:
        with st.spinner("Calculando comparables y venta de referencia..."):
            ranking = rank_similar_stores(clean_target, stores, weights, min_coverage=min_coverage)
            forecast, neighbors = weighted_knn_forecast(ranking, stores, outcome, k=k_neighbors)
        if ranking.empty:
            st.warning("No se encontraron tiendas con la cobertura mínima indicada. Baja la cobertura mínima o completa más variables del punto nuevo.")
        else:
            st.session_state["analysis"] = {
                "ranking": ranking,
                "forecast": forecast,
                "neighbors": neighbors,
                "target": clean_target,
                "project_name": project_name.strip() or "Punto nuevo sin nombre",
                "project_legacy": {
                    "tipo": "" if project_legacy_type == "No definido" else project_legacy_type,
                    "transparencia": project_transparency,
                    "accesibilidad": "" if project_accessibility == "No definida" else project_accessibility,
                },
                "weights": dict(weights),
                "coverage": min_coverage,
                "outcome": outcome,
                "k": k_neighbors,
                "unit": unit_label,
            }
            st.session_state.setdefault("mirror_legacy_by_cr", {})
            st.success("Análisis calculado. Puedes cambiar la tienda espejo elegida y completar su ficha manual.")

analysis = st.session_state.get("analysis")
if analysis:
    ranking = analysis["ranking"]
    forecast = analysis["forecast"]
    selected_outcome = analysis["outcome"]
    st.divider()
    results_tab, parallel_tab, validation_tab = st.tabs(["Resultados", "Ficha paralela", "Validación estadística"])

    with results_tab:
        first = ranking.iloc[0]
        st.subheader("Resultado de similitud")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tienda hermana sugerida", str(first["NAME"]))
        m2.metric("CR del espejo", str(first["CR"]))
        m3.metric("Similitud", f"{first['SIMILITUD']:.1f}%")
        m4.metric("Cobertura", f"{first['COBERTURA']:.0%}")
        if forecast:
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Venta estimada ponderada", display_number(forecast["estimacion_ponderada"]), help=f"Variable {selected_outcome}; unidad original declarada como: {analysis['unit']}.")
            s2.metric("Mediana de vecinos", display_number(forecast["mediana_vecinos"]))
            s3.metric("Rango P10–P90 observado", f"{display_number(forecast['p10_vecinos'])} – {display_number(forecast['p90_vecinos'])}")
            s4.metric("Vecinos con venta", int(forecast["n_vecinos"]))
            st.caption(
                f"El pronóstico es una media ponderada por similitud de los k={analysis['k']} comparables más cercanos con {selected_outcome} disponible. "
                "P10–P90 describe la dispersión de esos vecinos, no es un intervalo de confianza."
            )
        else:
            st.warning("Hay ranking de similitud, pero no se pudo calcular venta: los comparables no tienen la variable de venta seleccionada.")

        st.markdown("#### Tiendas comparables")
        candidates_view = render_candidate_table(ranking, stores, selected_outcome, limit=25)
        st.dataframe(candidates_view, use_container_width=True, hide_index=True)
        st.caption("Incluye CR para distinguir tiendas con nombres iguales. `TE` muestra el espejo histórico registrado en Book.")

        with st.expander("Contribución de familias al puntaje de la primera tienda", expanded=False):
            top = ranking.iloc[0]
            contrib = []
            family_columns = {
                "Ubicación": "DIST_Ubicación",
                "Perfil de entorno": "DIST_Perfil de entorno",
                "Infraestructura": "DIST_Infraestructura",
                "Vivienda y empleo": "DIST_Vivienda y empleo",
                "Tráfico externo": "DIST_Tráfico externo",
            }
            for family, column in family_columns.items():
                if column in ranking.columns and pd.notna(top[column]):
                    contrib.append({"Familia": family, "Distancia (0=igual, 1=máxima)": top[column]})
            if contrib:
                st.dataframe(pd.DataFrame(contrib), use_container_width=True, hide_index=True)
            else:
                st.caption("Sin contribuciones disponibles.")

    with parallel_tab:
        st.subheader("Variables del proyecto y de la tienda hermana")
        st.write("Completa los atributos del formato histórico. Los campos de transparencia y accesibilidad del espejo requieren validación/captura manual porque no están en JUN.")
        candidate_labels = [
            f"{row['CR']} · {row['NAME']} · {row.get('MUN', '')} · {row['SIMILITUD']:.1f}%"
            for _, row in ranking.head(25).iterrows()
        ]
        selected_label = st.selectbox("Espejo para completar", candidate_labels, key="selected_mirror_label")
        selected_cr = selected_label.split(" · ", 1)[0]
        selected_rank = ranking.loc[ranking["CR"].astype(str).eq(selected_cr)].iloc[0]
        mirror = stores.loc[stores["CR"].astype(str).eq(selected_cr)].iloc[0]
        st.caption(
            f"Book/JUN reporta Tipo de local = **{mirror.get('TIPO DE LOCAL', '—')}**, Fachada P = **{mirror.get('FACHADA P', '—')}**, "
            f"piso de venta = **{mirror.get('PISO DE VENTA', '—')} m²**. `Tipo de local` de Book no equivale necesariamente al `Tipo` categórico del formato anterior."
        )
        state_map = st.session_state.setdefault("mirror_legacy_by_cr", {})
        current = state_map.get(selected_cr, {})
        legacy_type_options = ["No definido", "Esquina Positiva", "Esquina Negativa", "Semiesquina", "Medianera Doble Altura", "Medianera Simple", "Especial"]
        access_options = ["No definida", "Excelente", "Muy Buena", "Buena", "Regular", "Mala", "No Cumple"]
        with st.form(f"mirror_attributes_{selected_cr}"):
            left, right = st.columns(2)
            with left:
                st.markdown("**Punto nuevo**")
                st.write(analysis["project_name"])
                st.write(f"Tipo (formato histórico): {analysis['project_legacy'].get('tipo') or 'No definido'}")
                st.write(f"Transparencia: {display_number(analysis['project_legacy'].get('transparencia'), 2)}")
                st.write(f"Accesibilidad: {analysis['project_legacy'].get('accesibilidad') or 'No definida'}")
            with right:
                st.markdown("**Tienda hermana elegida**")
                st.write(f"{mirror['CR']} · {mirror['NAME']}")
                m_type = st.selectbox(
                    "Tipo (formato histórico)", legacy_type_options,
                    index=legacy_type_options.index(current.get("tipo", "No definido")) if current.get("tipo", "No definido") in legacy_type_options else 0,
                    key=f"mirror_type_{selected_cr}",
                )
                m_transparency = st.number_input(
                    "Transparencia (captura manual)", min_value=0.0, max_value=1.0, value=current.get("transparencia"),
                    step=0.05, format="%.2f", placeholder="0–1", key=f"mirror_transparency_{selected_cr}",
                )
                m_access = st.selectbox(
                    "Accesibilidad (captura manual)", access_options,
                    index=access_options.index(current.get("accesibilidad", "No definida")) if current.get("accesibilidad", "No definida") in access_options else 0,
                    key=f"mirror_access_{selected_cr}",
                )
            save_parallel = st.form_submit_button("Guardar ficha paralela", type="primary")
        if save_parallel:
            state_map[selected_cr] = {
                "tipo": "" if m_type == "No definido" else m_type,
                "transparencia": m_transparency,
                "accesibilidad": "" if m_access == "No definida" else m_access,
            }
            st.session_state["mirror_legacy_by_cr"] = state_map
            st.success("Ficha paralela guardada para esta sesión.")

        current = st.session_state.get("mirror_legacy_by_cr", {}).get(selected_cr, current)
        project_points = legacy_partial_score(analysis["target"], analysis["project_legacy"])
        mirror_points = legacy_partial_score(mirror, current)
        score_rows = []
        for label, key in [
            ("Tipo", "Tipo"),
            ("Fachada", "Fachada"),
            ("Transparencia", "Transparencia"),
            ("Área piso de venta", "Área piso de venta"),
            ("Accesibilidad", "Accesibilidad"),
        ]:
            score_rows.append({
                "Variable histórica": label,
                "Punto nuevo · puntos": project_points["scores"][key],
                "Espejo · puntos": mirror_points["scores"][key],
            })
        score_rows.append({
            "Variable histórica": "Subtotal parcial (máx. 25)",
            "Punto nuevo · puntos": f"{project_points['puntaje_parcial']} / {project_points['maximo_si_completo']} · {project_points['variables_con_puntaje']}/5 con lookup",
            "Espejo · puntos": f"{mirror_points['puntaje_parcial']} / {mirror_points['maximo_si_completo']} · {mirror_points['variables_con_puntaje']}/5 con lookup",
        })
        st.markdown("#### Puntaje histórico parcial — 5 variables")
        st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)
        st.caption("Replica los lookups exactos visibles en Base Datos (filas 2–7) para Tipo, Fachada, Transparencia, Área de piso de venta y Accesibilidad. Solo suma variables con lookup exacto; no es el puntaje total del formato.")
        export = make_export_rows(
            analysis["project_name"], analysis["target"], analysis["project_legacy"], mirror, current,
            selected_rank, analysis["forecast"], selected_outcome, analysis["unit"],
        )
        st.markdown("#### Vista previa")
        st.dataframe(export, use_container_width=True, hide_index=True)
        csv_bytes = export.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Descargar ficha paralela CSV",
            data=csv_bytes,
            file_name="ficha_tienda_hermana.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("El CSV tiene dos filas —proyecto y espejo— para llevar los datos al formato de ficha anterior. No transforma la moneda ni inventa transparencia/accesibilidad.")

    with validation_tab:
        st.subheader("Contraste con la base histórica")
        st.write("TE se usa como etiqueta histórica para verificar si el ranking reproduce el espejo registrado. Las referencias genéricas o no enlazables quedan fuera del indicador.")
        if st.button("Calcular backtest de TE", key="run_te_backtest"):
            bar = st.progress(0.0, text="Evaluando espejos históricos enlazables...")
            try:
                metrics, te_results, label_counts = backtest_te(
                    stores, family_weights=analysis["weights"], min_coverage=analysis["coverage"], progress=bar.progress
                )
                st.session_state["te_validation"] = (metrics, te_results, label_counts)
                bar.empty()
            except Exception as exc:
                bar.empty()
                st.error(f"No se pudo calcular el backtest de TE: {exc}")
        te_validation = st.session_state.get("te_validation")
        if te_validation:
            metrics, te_results, label_counts = te_validation
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Etiquetas enlazables", metrics["n_etiquetas_enlazadas"])
            q2.metric("TE en Top 1", f"{metrics['top1']:.1%}" if pd.notna(metrics["top1"]) else "—")
            q3.metric("TE en Top 5", f"{metrics['top5']:.1%}" if pd.notna(metrics["top5"]) else "—")
            q4.metric("Rango recíproco medio (MRR)", f"{metrics['mrr']:.3f}" if pd.notna(metrics["mrr"]) else "—")
            st.write("**Estado de las referencias TE:** " + " · ".join(f"{key}: {value}" for key, value in label_counts.items()))
            st.caption("Backtest retrospectivo con la información actual; no es causal y puede diferir del proceso histórico original.")
            if pd.notna(metrics["top5"]) and metrics["top5"] < 0.10:
                st.warning("La concordancia con TE en Top 5 es baja. Revisa las variables disponibles, los pesos y la calidad de las etiquetas antes de usar el ranking como selección automática.")
            if not te_results.empty:
                st.dataframe(te_results.sort_values("RANGO_MODELO", na_position="last"), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### Validación de la estimación de ventas")
        st.write("Validación leave-one-out: se oculta una tienda abierta, se estima con las demás y se compara contra su venta real. La referencia base es la mediana de ventas del resto.")
        if st.button("Validar estimador de ventas (leave-one-out)", key="run_sales_backtest"):
            bar = st.progress(0.0, text=f"Validando {selected_outcome} para tiendas abiertas...")
            try:
                metrics, sales_results = backtest_sales_loocv(
                    stores, selected_outcome, family_weights=analysis["weights"], min_coverage=analysis["coverage"],
                    k=analysis["k"], progress=bar.progress,
                )
                st.session_state["sales_validation"] = (metrics, sales_results, selected_outcome)
                bar.empty()
            except Exception as exc:
                bar.empty()
                st.error(f"No se pudo validar ventas: {exc}")
        sales_validation = st.session_state.get("sales_validation")
        if sales_validation and sales_validation[2] == selected_outcome:
            metrics, sales_results, validation_outcome = sales_validation
            if metrics.get("n", 0):
                a, b, c, d = st.columns(4)
                a.metric("Puntos evaluados", int(metrics["n"]))
                b.metric("MAE del modelo", display_number(metrics["mae"]))
                c.metric("MAE referencia global", display_number(metrics["mae_baseline"]))
                d.metric("Mejora vs. referencia", f"{metrics['mejora_mae_pct']:.1f}%" if pd.notna(metrics["mejora_mae_pct"]) else "—")
                st.caption(f"Error porcentual absoluto medio: {metrics['mape_pct']:.1f}% · Mediana del error absoluto porcentual: {metrics['mediana_error_abs_pct']:.1f}%. Valores en {analysis['unit']}.")
                if pd.notna(metrics["mape_pct"]) and metrics["mape_pct"] > 50:
                    st.warning("El error porcentual medio es alto, aunque la mediana pueda ser menor. Usa la estimación como referencia preliminar y revisa los casos extremos.")
                with st.expander("Ver errores punto por punto"):
                    st.dataframe(sales_results.sort_values("REAL", ascending=False), use_container_width=True, hide_index=True)
            else:
                st.warning("No hubo suficientes análogos disponibles para validar esta variable de venta.")
        st.warning("La concordancia con TE valida reproducción de elecciones históricas; la validación leave-one-out estima error en tiendas existentes. Ninguna sustituye una prueba temporal prospectiva con aperturas nuevas.")

else:
    st.info("Completa el punto propuesto y pulsa **Buscar tienda hermana y estimar venta**. Los campos desconocidos pueden quedar vacíos; se reducirán la cobertura y las variables comparables.")
