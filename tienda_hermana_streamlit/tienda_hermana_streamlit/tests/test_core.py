import numpy as np
import pandas as pd

from core import (
    FEATURE_FAMILIES,
    _pairwise_distance_matrices,
    legacy_partial_score,
    normalize_text,
    prepare_jun,
    rank_similar_stores,
    weighted_knn_forecast,
)


def _row(cr, name, status, city, area, homes, sales, te="NAME"):
    row = {column: None for columns in FEATURE_FAMILIES.values() for column in columns}
    row.update({
        "CR": cr,
        "NAME": name,
        "ESTADO": status,
        "DEPARTAMENTO": "D",
        "MUN": city,
        "ZONA": "Z",
        "UPZ/COMUNA": "U",
        "ESTRATO": 3,
        "TIPO DE LOCAL": "Esquinero",
        "PARQUEADERO": "NO",
        "AREA": area,
        "PISO DE VENTA": area * 0.6,
        "FACHADA P": 8,
        "V100": homes,
        "VT": homes * 3,
        "E100": 50,
        "ET": 100,
        "TR15MIN": 120,
        "TV15MIN": 10,
        "TM15MIN": 5,
        "24H": "SI",
        "GENERADOR": "BAJA DENSIDAD",
        "SEG26": "BASE",
        "TIE26": "TMCB",
        "PTH": "Medio",
        "TE": te,
        "VENTAS OUM": sales,
        "VU6M": sales * 1.02,
    })
    return row


def _stores():
    raw = pd.DataFrame([
        _row("CR1", "Tienda Uno", "ABIERTA", "Bogotá", 100, 200, 200000, "Tienda Dos"),
        _row("CR2", "Tienda Dos", "ABIERTA", "Bogotá", 105, 210, 220000, "Tienda Uno"),
        _row("CR3", "Tienda Cerrada", "CERRADA", "Cali", 90, 400, 400000),
    ])
    return prepare_jun(raw)


def test_normalize_text_ignores_accents_and_punctuation():
    assert normalize_text("Árbol-22 / Bogotá") == "ARBOL22BOGOTA"


def test_prepare_jun_keeps_only_open_and_parses_sales():
    stores = _stores()
    assert stores["CR"].tolist() == ["CR1", "CR2"]
    assert pd.api.types.is_numeric_dtype(stores["VU6M"])


def test_ranking_prefers_nearby_categorical_and_numeric_store():
    stores = _stores()
    target = stores.iloc[0].to_dict()
    ranked = rank_similar_stores(target, stores, min_coverage=0.25, exclude_cr="CR1")
    assert ranked.iloc[0]["CR"] == "CR2"
    assert 0 <= ranked.iloc[0]["SIMILITUD"] <= 100
    assert ranked.iloc[0]["COBERTURA"] >= 0.25


def test_weighted_knn_returns_sales_and_neighbors():
    stores = _stores()
    target = stores.iloc[0].to_dict()
    ranked = rank_similar_stores(target, stores, min_coverage=0.25, exclude_cr="CR1")
    summary, neighbors = weighted_knn_forecast(ranked, stores, "VU6M", k=1)
    assert summary["n_vecinos"] == 1
    assert np.isclose(summary["estimacion_ponderada"], 220000 * 1.02)
    assert neighbors.iloc[0]["CR"] == "CR2"


def test_legacy_partial_score_uses_exact_template_lookups_only():
    record = {"FACHADA P": 8, "PISO DE VENTA": 80}
    legacy = {"tipo": "Esquina Positiva", "transparencia": 0.9, "accesibilidad": "Excelente"}
    result = legacy_partial_score(record, legacy)
    assert result["scores"] == {
        "Tipo": 5,
        "Fachada": 3,
        "Transparencia": 5,
        "Área piso de venta": 3,
        "Accesibilidad": 5,
    }
    assert result["puntaje_parcial"] == 21
    assert result["variables_con_puntaje"] == 5

    partial = legacy_partial_score({"FACHADA P": 8.32, "PISO DE VENTA": 62.39}, legacy)
    assert partial["scores"]["Fachada"] is None
    assert partial["scores"]["Área piso de venta"] is None


def test_pairwise_distance_matrix_is_zero_on_diagonal_and_has_full_coverage():
    stores = _stores()
    distances, coverage = _pairwise_distance_matrices(stores)
    assert np.allclose(np.diag(distances), 0.0)
    assert np.allclose(coverage, 1.0)
    assert np.allclose(distances, distances.T)
