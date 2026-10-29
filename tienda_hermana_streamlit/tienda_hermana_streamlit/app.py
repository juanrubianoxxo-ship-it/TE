from __future__ import annotations

import hashlib
import html

import numpy as np
import pandas as pd
import streamlit as st

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

st.set_page_config(page_title="Tienda Hermana | Propuesta de venta", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.25rem; max-width: 1500px;}
    [data-testid="stMetric"] {background: #f4f7f7; border: 1px solid #e2eaea; padding: 12px 16px; border-radius: 10px;}
    [data-testid="stMetricValue"] {font-size: 2.15rem; font-weight: 750;}
    .big-sale-label {font-size: 1rem; font-weight: 650; color: #52636b; letter-spacing: .04em;}
    .big-sale-value {font-size: clamp(2.7rem, 6vw, 5rem); font-weight: 800; line-height: 1.05; color: #0b6256; margin: .2rem 0 .5rem 0;}
    div[data-testid="stAlert"] {border-radius: 10px;}
    </style>
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
    """Dropdown whose values come only from distinct, non-missing open-Book values."""
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
        placeholder="Dejar vacío si no se conoce",
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
    return (
        f"{row['CR']} · {row['NAME']} · {row.get('MUN', '')} · "
        f"similitud {row['SIMILITUD']:.1f}%"
    )


def score_table(project_score: dict[str, object], mirror_score: dict[str, object]) -> pd.DataFrame:
    labels = ["Tipo", "Fachada", "Transparencia", "Área piso de venta", "Accesibilidad"]
    rows = [
        {
            "Variable del formato": label,
            "Punto nuevo · puntos": project_score["scores"].get(label),
            "Tienda hermana · puntos": mirror_score["scores"].get(label),
        }
        for label in labels
    ]
    rows.append({
        "Variable del formato": "Subtotal parcial (máx. 25)",
        "Punto nuevo · puntos": (
            f"{project_score['puntaje_parcial']} / 25 "
            f"({project_score['variables_con_puntaje']}/5 variables con lookup)"
        ),
        "Tienda hermana · puntos": (
            f"{mirror_score['puntaje_parcial']} / 25 "
            f"({mirror_score['variables_con_puntaje']}/5 variables con lookup)"
        ),
    })
    return pd.DataFrame(rows)


st.title("Tienda Hermana")
st.caption("Completa el punto nuevo, revisa la tienda espejo y estima la venta con sus comparables.")
st.write("El formulario pregunta primero las características del nuevo punto presentes en Book. Luego permite revisar el espejo y completar en paralelo las tres variables del formato histórico.")

with st.sidebar:
    st.header("Base de tiendas")
    uploaded = st.file_uploader("Cargar Book (.xlsx)", type=["xlsx"], help="Se utiliza la hoja JUN y solo las filas ABIERTA.")
    st.caption("El archivo permanece en la sesión de la app; no se incluye en el proyecto de GitHub.")

if not uploaded:
    st.info("Carga el Book para iniciar.")
    st.stop()

file_bytes = uploaded.getvalue()
book_key = hashlib.sha256(file_bytes).hexdigest()
if st.session_state.get("_book_key") != book_key:
    for key in [
        "book_dataframe", "model_weights", "analysis", "estimate_result",
        "analysis_counter", "mirror_attributes",
    ]:
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

sales_options = [column for column in ["VU6M", "VENTAS OUM"] if column in stores.columns]
if not sales_options:
    st.error("La hoja JUN no contiene VU6M ni VENTAS OUM para estimar.")
    st.stop()

if "model_weights" not in st.session_state:
    with st.spinner("Preparando el modelo de similitud con las referencias históricas..."):
        st.session_state["model_weights"] = fit_te_family_weights(stores)
model_weights = st.session_state["model_weights"]

n_open = len(stores)
with st.container(border=True):
    st.markdown(f"**Book/JUN listo:** {n_open:,} tiendas abiertas. Las opciones de los desplegables se toman de esos registros.".replace(",", "."))

st.header("1 · Datos del punto nuevo")
st.caption("Completa todo lo disponible antes de buscar. Los campos categóricos usan valores existentes en Book; los desconocidos pueden quedar sin informar.")

with st.form("new_point_form", clear_on_submit=False):
    project_name = st.text_input("Nombre del proyecto / nuevo punto", key="project_name_input")

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
        target_facade = optional_numeric("Fachada P (valor de Book)", "target_facade")

    st.markdown("**Viviendas, empleo y tráfico externo**")
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

    st.markdown("**Estimación**")
    c1, c2, c3 = st.columns(3)
    with c1:
        outcome = st.selectbox(
            "Venta objetivo",
            sales_options,
            index=sales_options.index("VU6M") if "VU6M" in sales_options else 0,
            format_func=lambda value: "Ventas operativas últimos 6 meses (VU6M)" if value == "VU6M" else "Ventas operativas último mes (VENTAS OUM)",
        )
    with c2:
        k_neighbors = st.number_input("Número de comparables (k)", min_value=1, max_value=25, value=10, step=1)
    with c3:
        min_coverage = st.number_input("Cobertura mínima del ranking", min_value=0.1, max_value=1.0, value=0.4, step=0.05, format="%.2f")
    unit_label = st.text_input("Unidad tal como está en Book", value="Unidad reportada en Book")

    search_clicked = st.form_submit_button("Buscar tienda hermana", type="primary", use_container_width=True)

if search_clicked:
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
    active = [
        column for columns in FEATURE_FAMILIES.values()
        for column in columns if target.get(column) is not None
    ]
    active_families = sum(
        any(target.get(column) is not None for column in columns)
        for columns in FEATURE_FAMILIES.values()
    )
    if len(active) < 2 or active_families < 2:
        st.error("Completa por lo menos dos variables, pertenecientes a dos familias distintas, antes de buscar el espejo.")
    else:
        with st.spinner("Buscando tiendas abiertas con características similares..."):
            ranking = rank_similar_stores(
                target,
                stores,
                family_weights=model_weights,
                min_coverage=float(min_coverage),
            )
        if ranking.empty:
            st.warning("No se encontró un espejo con la cobertura indicada. Completa más datos del punto o baja la cobertura mínima.")
        else:
            counter = int(st.session_state.get("analysis_counter", 0)) + 1
            st.session_state["analysis_counter"] = counter
            st.session_state["analysis"] = {
                "id": counter,
                "target": target,
                "project_name": project_name.strip() or "Punto nuevo sin nombre",
                "ranking": ranking,
                "outcome": outcome,
                "k": int(k_neighbors),
                "min_coverage": float(min_coverage),
                "unit": unit_label.strip() or "Unidad reportada en Book",
            }
            st.session_state.pop("estimate_result", None)
            st.success("Ranking listo. Revisa el espejo sugerido y completa abajo las tres variables en paralelo.")

analysis = st.session_state.get("analysis")
if analysis:
    ranking = analysis["ranking"]
    analysis_id = analysis["id"]
    st.divider()
    st.header("2 · Tienda hermana sugerida")

    if ranking.empty:
        st.warning("No hay candidatos disponibles con esos datos.")
        st.stop()

    top = ranking.iloc[0]
    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Primera recomendación", str(top["NAME"]))
    top2.metric("CR / ID de la tienda", str(top["CR"]))
    top3.metric("Similitud", f"{top['SIMILITUD']:.1f}%")
    top4.metric("Cobertura", f"{top['COBERTURA']:.0%}")

    top_rows = ranking.head(15).copy()
    st.caption("La primera opción es la tienda más similar. El CR se muestra para diferenciar tiendas con nombres repetidos; puedes elegir otra del ranking si corresponde.")
    visible_columns = [column for column in ["CR", "NAME", "MUN", "SIMILITUD", "COBERTURA", "VARIABLES_COMPARADAS"] if column in top_rows]
    st.dataframe(top_rows[visible_columns], use_container_width=True, hide_index=True)

    candidate_crs = top_rows["CR"].astype(str).tolist()
    selected_key = f"selected_mirror_cr_{analysis_id}"
    selected_cr = st.selectbox(
        "Selecciona el espejo para completar la ficha",
        candidate_crs,
        index=0,
        key=selected_key,
        format_func=lambda cr: candidate_label(cr, ranking),
    )
    selected_rank = ranking.loc[ranking["CR"].astype(str).eq(str(selected_cr))].iloc[0]
    mirror = stores.loc[stores["CR"].astype(str).eq(str(selected_cr))].iloc[0]
    st.info(
        f"Espejo elegido: **{mirror['NAME']}** · CR **{mirror['CR']}** · "
        f"{selected_rank['SIMILITUD']:.1f}% de similitud."
    )

    st.header("3 · Completar ficha en paralelo")
    st.caption("Captura Tipo, Transparencia y Accesibilidad tanto para el proyecto como para el espejo. Los puntajes corresponden a las tablas del formato histórico.")
    with st.form(f"legacy_parallel_form_{analysis_id}_{selected_cr}"):
        new_col, mirror_col = st.columns(2)
        with new_col:
            st.markdown(f"**Punto nuevo · {analysis['project_name']}**")
            new_type = st.selectbox("Tipo (formato histórico)", [NO_SELECTION] + TYPE_OPTIONS, key=f"new_type_{analysis_id}")
            new_transparency = st.selectbox(
                "Transparencia",
                [NO_SELECTION] + TRANSPARENCY_OPTIONS,
                key=f"new_transparency_{analysis_id}",
                format_func=lambda value: (
                    NO_SELECTION if value == NO_SELECTION else
                    display_transparency(value)
                ),
            )
            new_access = st.selectbox("Accesibilidad", [NO_SELECTION] + ACCESS_OPTIONS, key=f"new_access_{analysis_id}")
        with mirror_col:
            st.markdown(f"**Tienda hermana · {mirror['NAME']} · CR {mirror['CR']}**")
            mirror_type = st.selectbox("Tipo (formato histórico) ", [NO_SELECTION] + TYPE_OPTIONS, key=f"mirror_type_{analysis_id}_{selected_cr}")
            mirror_transparency = st.selectbox(
                "Transparencia ",
                [NO_SELECTION] + TRANSPARENCY_OPTIONS,
                key=f"mirror_transparency_{analysis_id}_{selected_cr}",
                format_func=lambda value: (
                    NO_SELECTION if value == NO_SELECTION else
                    display_transparency(value)
                ),
            )
            mirror_access = st.selectbox("Accesibilidad ", [NO_SELECTION] + ACCESS_OPTIONS, key=f"mirror_access_{analysis_id}_{selected_cr}")
        estimate_clicked = st.form_submit_button("Estimar venta", type="primary", use_container_width=True)

    if estimate_clicked:
        required_values = [new_type, new_transparency, new_access, mirror_type, mirror_transparency, mirror_access]
        if any(value == NO_SELECTION for value in required_values):
            st.error("Completa las seis selecciones de la ficha paralela antes de estimar la venta.")
        else:
            project_legacy = {
                "tipo": new_type,
                "transparencia": new_transparency,
                "accesibilidad": new_access,
            }
            mirror_legacy = {
                "tipo": mirror_type,
                "transparencia": mirror_transparency,
                "accesibilidad": mirror_access,
            }
            with st.spinner("Calculando la venta estimada con los comparables y la ficha completada..."):
                forecast, neighbors = weighted_knn_forecast(
                    ranking,
                    stores,
                    analysis["outcome"],
                    k=analysis["k"],
                    include_cr=selected_cr,
                )
            if not forecast:
                st.error(f"No hay ventas {analysis['outcome']} disponibles en los comparables seleccionados.")
            else:
                project_points = legacy_partial_score(analysis["target"], project_legacy)
                mirror_points = legacy_partial_score(mirror, mirror_legacy)
                st.session_state["estimate_result"] = {
                    "analysis_id": analysis_id,
                    "selected_cr": str(selected_cr),
                    "project_legacy": project_legacy,
                    "mirror_legacy": mirror_legacy,
                    "project_points": project_points,
                    "mirror_points": mirror_points,
                    "forecast": forecast,
                    "neighbors": neighbors,
                }

    result = st.session_state.get("estimate_result")
    if result and result.get("analysis_id") == analysis_id and result.get("selected_cr") == str(selected_cr):
        st.divider()
        st.header("4 · Resultado y resumen")
        estimate = result["forecast"]["estimacion_ponderada"]
        st.markdown('<div class="big-sale-label">VENTA ESTIMADA DEL NUEVO PUNTO</div>', unsafe_allow_html=True)
        safe_unit = html.escape(str(analysis["unit"]))
        st.markdown(f'<div class="big-sale-value">{display_number(estimate)} <span style="font-size:1.2rem;font-weight:600">{safe_unit}</span></div>', unsafe_allow_html=True)
        st.caption(
            f"Variable estimada: **{analysis['outcome']}** · Promedio ponderado de {result['forecast']['n_vecinos']} tiendas comparables, "
            f"incluyendo el espejo elegido **{mirror['NAME']} (CR {mirror['CR']})**."
        )

        st.subheader("Puntajes de la ficha histórica")
        st.dataframe(
            score_table(result["project_points"], result["mirror_points"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Se aplican los lookups exactos del formato anterior para Tipo, Transparencia y Accesibilidad; "
            "Fachada y Área de piso de venta se puntúan cuando su valor de Book coincide exactamente con la tabla. "
            "El subtotal es parcial, no representa los 12 rubros completos del formato."
        )

        st.subheader("Cómo se calculó la venta")
        st.write(
            f"Se tomaron las {result['forecast']['n_vecinos']} tiendas más similares con **{analysis['outcome']}** disponible. "
            "La estimación es el promedio de sus ventas, ponderado por similitud; el espejo seleccionado se incluye en el grupo."
        )
        neighbors = result["neighbors"].copy()
        neighbors["Peso en estimación (%)"] = neighbors["PESO_RELATIVO"] * 100
        neighbors["Aporte a la venta"] = neighbors["APORTE_ESTIMACION"]
        show_columns = [
            "CR", "NAME", "MUN", "SIMILITUD", "COBERTURA", "VENTA_OBJETIVO",
            "Peso en estimación (%)", "Aporte a la venta",
        ]
        show_columns = [column for column in show_columns if column in neighbors]
        st.dataframe(neighbors[show_columns], use_container_width=True, hide_index=True)
        st.caption(
            f"Fórmula: suma de (venta de cada comparable × peso relativo). Peso inverso a distancia: 1/(distancia + 0,025)². "
            f"Mediana del grupo: {display_number(result['forecast']['mediana_vecinos'])}; "
            f"P10–P90 observado: {display_number(result['forecast']['p10_vecinos'])}–{display_number(result['forecast']['p90_vecinos'])} "
            "(dispersión de comparables, no intervalo de confianza)."
        )
        st.info(
            "Los puntos históricos se muestran como parte del resumen de la ficha. Transparencia y accesibilidad no están en JUN para las tiendas de entrenamiento, "
            "por lo que sus puntos no se usan como predictores de ventas hasta contar con esos datos históricos de forma consistente."
        )
    elif result and result.get("analysis_id") == analysis_id:
        st.info("Cambió el espejo seleccionado. Completa su ficha paralela y pulsa **Estimar venta** para actualizar el resultado.")
