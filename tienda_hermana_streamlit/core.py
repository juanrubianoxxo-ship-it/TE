from __future__ import annotations

import re
import unicodedata
from io import BytesIO
from typing import Callable, Mapping

import numpy as np
import pandas as pd


FEATURE_FAMILIES: dict[str, list[str]] = {
    "Ubicación": ["DEPARTAMENTO", "MUN", "ZONA", "UPZ/COMUNA"],
    "Perfil de entorno": ["ESTRATO", "GENERADOR", "SEG26", "PTH"],
    "Infraestructura": [
        "TIPO DE LOCAL", "PARQUEADERO", "24H", "AREA", "PISO DE VENTA", "FACHADA P"
    ],
    "Vivienda y empleo": ["V100", "VT", "E100", "ET"],
    "Tráfico externo": ["TR15MIN", "TV15MIN", "TM15MIN"],
}

NUMERIC_FEATURES = {
    "ESTRATO", "AREA", "PISO DE VENTA", "FACHADA P", "V100", "VT", "E100", "ET",
    "TR15MIN", "TV15MIN", "TM15MIN",
}
LOG_FEATURES = {"V100", "VT", "E100", "ET", "TR15MIN", "TV15MIN", "TM15MIN"}

REQUIRED_COLUMNS = {
    "CR", "NAME", "ESTADO", "DEPARTAMENTO", "MUN", "ZONA", "UPZ/COMUNA", "ESTRATO",
    "TIPO DE LOCAL", "PARQUEADERO", "AREA", "PISO DE VENTA", "FACHADA P", "V100", "VT",
    "E100", "ET", "TR15MIN", "TV15MIN", "TM15MIN", "24H", "GENERADOR", "SEG26",
    "PTH", "TE", "VENTAS OUM", "VU6M",
}

DEFAULT_FAMILY_WEIGHTS = {family: 1.0 for family in FEATURE_FAMILIES}
GENERIC_TE_LABELS = {
    "", "NAME", "PT", "N/A", "NA", "ND", "0", "NO", "PENDIENTE", "SININFO",
    "SINDATO", "NONE", "NAN", "-",
}

LEGACY_TYPE_POINTS = {
    "ESQUINAPOSITIVA": 5,
    "ESQUINANEGATIVA": 4,
    "SEMIESQUINA": 3,
    "MEDIANERADOBLEALTURA": 2,
    "MEDIANERASIMPLE": 1,
    "ESPECIAL": 0,
}
LEGACY_FACADE_POINTS = {10.0: 5, 9.0: 4, 8.0: 3, 7.0: 2, 6.0: 1}
LEGACY_TRANSPARENCY_POINTS = {0.9: 5, 0.8: 4, 0.7: 3, 0.6: 2, 0.5: 1}
LEGACY_FLOOR_AREA_POINTS = {100.0: 5, 90.0: 4, 80.0: 3, 70.0: 2, 60.0: 1}
LEGACY_ACCESS_POINTS = {
    "EXCELENTE": 5,
    "MUYBUENA": 4,
    "BUENA": 3,
    "REGULAR": 2,
    "MALA": 1,
    "NOCUMPLE": 0,
}


class WorkbookError(ValueError):
    """Error legible para un archivo Book incompatible."""


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    text = "".join(
        char for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _blank_to_nan(value: object) -> object:
    if value is None or (isinstance(value, str) and value.strip().upper() in {"#N/A", "N/A", "NA", ""}):
        return np.nan
    return value


def prepare_jun(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate JUN and return only unique, open stores with parsed modeling fields."""
    df = raw.copy()
    df.columns = [str(column).strip() for column in df.columns]
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise WorkbookError("Faltan columnas requeridas en JUN: " + ", ".join(missing))

    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].map(_blank_to_nan)

    status = df["ESTADO"].astype("string").str.strip().str.upper()
    open_df = df.loc[status.eq("ABIERTA")].copy()
    if open_df.empty:
        raise WorkbookError("JUN no contiene filas con ESTADO = ABIERTA.")

    open_df["CR"] = open_df["CR"].astype("string").str.strip()
    open_df["NAME"] = open_df["NAME"].astype("string").str.strip()
    duplicated = open_df["CR"].duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = open_df.loc[duplicated, "CR"].astype(str).unique().tolist()
        raise WorkbookError("CR duplicados entre tiendas ABIERTAS: " + ", ".join(duplicate_ids[:10]))

    numeric_columns = set(NUMERIC_FEATURES) | {
        "VENTAS OUM", "VU6M", "TRAFICO UM", "TRU6", "TICKET UM", "TICKET U6M",
        "CONTRIBUCION UM", "CONTRIBUCION U6M", "MARGEN UM", "MARGEN U6M", "RENTA", "RU6M",
        "MESOP", "Y", "X",
    }
    for column in numeric_columns & set(open_df.columns):
        open_df[column] = pd.to_numeric(open_df[column], errors="coerce")

    # Normalized keys support auditable TE lookup; original values remain untouched.
    open_df["_NAME_KEY"] = open_df["NAME"].map(normalize_text)
    open_df["_TE_KEY"] = open_df["TE"].map(normalize_text)
    open_df.reset_index(drop=True, inplace=True)
    return open_df


def legacy_partial_score(record: Mapping[str, object], legacy: Mapping[str, object]) -> dict[str, object]:
    """Reproduce exact lookups from Base Datos rows 2–7 for five legacy fields.

    This is intentionally partial: other rows of the historical score need
    project-specific inputs and are not estimated here.
    """
    type_key = normalize_text(legacy.get("tipo"))
    access_key = normalize_text(legacy.get("accesibilidad"))

    def exact_score(value: object, lookup: Mapping[float, int]) -> int | None:
        try:
            if value is None or pd.isna(value):
                return None
            number = float(value)
        except (TypeError, ValueError):
            if normalize_text(value) == "NOCUMPLE":
                return 0
            return None
        return next((points for key, points in lookup.items() if np.isclose(number, key, atol=1e-9)), None)

    scores = {
        "Tipo": LEGACY_TYPE_POINTS.get(type_key),
        "Fachada": exact_score(record.get("FACHADA P"), LEGACY_FACADE_POINTS),
        "Transparencia": exact_score(legacy.get("transparencia"), LEGACY_TRANSPARENCY_POINTS),
        "Área piso de venta": exact_score(record.get("PISO DE VENTA"), LEGACY_FLOOR_AREA_POINTS),
        "Accesibilidad": LEGACY_ACCESS_POINTS.get(access_key),
    }
    known = [value for value in scores.values() if value is not None]
    return {
        "scores": scores,
        "puntaje_parcial": int(sum(known)),
        "variables_con_puntaje": int(len(known)),
        "maximo_parcial": int(5 * len(known)),
        "maximo_si_completo": 25,
    }


def read_jun_excel(file_bytes: bytes) -> pd.DataFrame:
    """Read the JUN worksheet from an uploaded workbook."""
    try:
        workbook = pd.ExcelFile(BytesIO(file_bytes))
    except Exception as exc:  # pandas gives several engine-specific exceptions
        raise WorkbookError(f"No pude abrir el archivo Excel: {exc}") from exc
    if "JUN" not in workbook.sheet_names:
        raise WorkbookError(
            "No encontré una hoja llamada JUN. Hojas detectadas: " + ", ".join(workbook.sheet_names)
        )
    try:
        raw = pd.read_excel(workbook, sheet_name="JUN")
    except Exception as exc:
        raise WorkbookError(f"No pude leer la hoja JUN: {exc}") from exc
    return prepare_jun(raw)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _numeric_distance(values: pd.Series, target: float, column: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    target_value = float(target)
    if column in LOG_FEATURES:
        numeric = np.log1p(numeric.clip(lower=0))
        target_value = float(np.log1p(max(target_value, 0)))

    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    low = float(valid.quantile(0.05))
    high = float(valid.quantile(0.95))
    scale = high - low
    if not np.isfinite(scale) or scale <= 1e-12:
        distance = np.where(np.isclose(numeric, target_value, equal_nan=False), 0.0, 1.0)
    else:
        distance = (numeric - target_value).abs() / scale
        distance = distance.clip(lower=0.0, upper=1.0)
    return pd.Series(distance, index=values.index, dtype=float).where(numeric.notna())


def rank_similar_stores(
    target: Mapping[str, object],
    stores: pd.DataFrame,
    family_weights: Mapping[str, float] | None = None,
    min_coverage: float = 0.40,
    exclude_cr: object | None = None,
) -> pd.DataFrame:
    """Rank open stores with a weighted, grouped Gower-like distance.

    Categorical features use exact normalized match (0/1); numeric features use
    clipped differences on a log1p/robust-range scale. Missing target/source
    values are omitted and coverage is reported, not imputed.
    """
    weights = dict(DEFAULT_FAMILY_WEIGHTS)
    if family_weights:
        weights.update({key: max(float(value), 0.0) for key, value in family_weights.items()})

    active: list[tuple[str, str, float, object]] = []
    for family, columns in FEATURE_FAMILIES.items():
        available = [
            column for column in columns
            if column in stores.columns and column in target and not _is_missing(target[column])
        ]
        if not available or weights.get(family, 0.0) <= 0:
            continue
        per_feature_weight = weights[family] / len(available)
        for column in available:
            active.append((family, column, per_feature_weight, target[column]))

    if not active:
        return pd.DataFrame()

    candidates = stores.copy()
    if exclude_cr is not None:
        candidates = candidates.loc[candidates["CR"].astype(str) != str(exclude_cr)]
    if candidates.empty:
        return pd.DataFrame()

    weighted_distance = pd.Series(0.0, index=candidates.index)
    available_weight = pd.Series(0.0, index=candidates.index)
    total_weight = sum(item[2] for item in active)
    family_distance_numerator = {family: pd.Series(0.0, index=candidates.index) for family in FEATURE_FAMILIES}
    family_available_weight = {family: pd.Series(0.0, index=candidates.index) for family in FEATURE_FAMILIES}
    feature_counts = pd.Series(0, index=candidates.index, dtype=int)

    for family, column, feature_weight, target_value in active:
        if column in NUMERIC_FEATURES:
            distance = _numeric_distance(candidates[column], float(target_value), column)
        else:
            target_key = normalize_text(target_value)
            candidate_keys = candidates[column].map(normalize_text)
            distance = pd.Series(
                np.where(candidate_keys.eq(""), np.nan, candidate_keys.ne(target_key).astype(float)),
                index=candidates.index,
            )
        available = distance.notna()
        weighted_distance.loc[available] += distance.loc[available] * feature_weight
        available_weight.loc[available] += feature_weight
        family_distance_numerator[family].loc[available] += distance.loc[available] * feature_weight
        family_available_weight[family].loc[available] += feature_weight
        feature_counts.loc[available] += 1

    result = candidates[[column for column in ["CR", "NAME", "MUN", "DEPARTAMENTO", "SEG26"] if column in candidates]].copy()
    result["COBERTURA"] = (available_weight / total_weight).reindex(candidates.index)
    result["DISTANCIA"] = (weighted_distance / available_weight.replace(0, np.nan)).reindex(candidates.index)
    result["SIMILITUD"] = (100.0 * (1.0 - result["DISTANCIA"])).clip(0, 100)
    result["VARIABLES_COMPARADAS"] = feature_counts.reindex(candidates.index)
    for family in FEATURE_FAMILIES:
        denom = family_available_weight[family].replace(0, np.nan)
        result[f"DIST_{family}"] = (family_distance_numerator[family] / denom).reindex(candidates.index)

    result = result.loc[result["COBERTURA"].ge(float(min_coverage))].copy()
    result.sort_values(["SIMILITUD", "COBERTURA", "CR"], ascending=[False, False, True], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def _pairwise_distance_matrices(
    stores: pd.DataFrame,
    family_weights: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized pairwise equivalent of rank_similar_stores for backtests."""
    weights = dict(DEFAULT_FAMILY_WEIGHTS)
    if family_weights:
        weights.update({key: max(float(value), 0.0) for key, value in family_weights.items()})
    n = len(stores)
    numerator = np.zeros((n, n), dtype=float)
    denominator = np.zeros((n, n), dtype=float)
    total_target_weight = np.zeros(n, dtype=float)

    for family, columns in FEATURE_FAMILIES.items():
        family_columns = [column for column in columns if column in stores.columns]
        family_weight = weights.get(family, 0.0)
        if not family_columns or family_weight <= 0:
            continue

        vectors: dict[str, np.ndarray] = {}
        valid_vectors: dict[str, np.ndarray] = {}
        for column in family_columns:
            if column in NUMERIC_FEATURES:
                values = pd.to_numeric(stores[column], errors="coerce").to_numpy(dtype=float)
                if column in LOG_FEATURES:
                    values = np.log1p(np.clip(values, 0, None))
                vectors[column] = values
            else:
                values = stores[column].map(normalize_text).to_numpy(dtype=object)
                vectors[column] = values
            valid_vectors[column] = np.array([
                (np.isfinite(value) if column in NUMERIC_FEATURES else bool(value))
                for value in vectors[column]
            ], dtype=bool)

        target_feature_count = np.zeros(n, dtype=float)
        for column in family_columns:
            target_feature_count += valid_vectors[column].astype(float)
        active_family = target_feature_count > 0
        total_target_weight[active_family] += family_weight
        per_target_feature_weight = np.divide(
            family_weight,
            target_feature_count,
            out=np.zeros(n, dtype=float),
            where=active_family,
        )

        for column in family_columns:
            values = vectors[column]
            valid = valid_vectors[column]
            pair_valid = valid[:, None] & valid[None, :]
            if column in NUMERIC_FEATURES:
                observed = values[valid]
                if not len(observed):
                    continue
                low, high = np.quantile(observed, [0.05, 0.95])
                scale = float(high - low)
                if not np.isfinite(scale) or scale <= 1e-12:
                    difference = np.not_equal.outer(values, values).astype(float)
                else:
                    difference = np.clip(np.abs(values[:, None] - values[None, :]) / scale, 0.0, 1.0)
            else:
                difference = np.not_equal.outer(values, values).astype(float)
            pair_weight = per_target_feature_weight[:, None] * pair_valid
            numerator += np.where(pair_valid, difference, 0.0) * pair_weight
            denominator += pair_weight

    distances = np.divide(
        numerator, denominator,
        out=np.full((n, n), np.nan, dtype=float),
        where=denominator > 0,
    )
    coverage = np.divide(
        denominator,
        total_target_weight[:, None],
        out=np.zeros((n, n), dtype=float),
        where=total_target_weight[:, None] > 0,
    )
    return distances, coverage


def resolve_te_labels(stores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Classify TE labels as linked-to-open-store, generic, ambiguous or unresolved."""
    name_to_indices: dict[str, list[int]] = {}
    for idx, key in enumerate(stores["_NAME_KEY"]):
        if key:
            name_to_indices.setdefault(key, []).append(idx)

    entries = []
    for idx, row in stores.iterrows():
        key = row.get("_TE_KEY", "")
        label = row.get("TE")
        if not key or key in GENERIC_TE_LABELS:
            status = "genérica/sin etiqueta"
            mirror_idx = None
        elif key not in name_to_indices:
            status = "no enlazada a tienda abierta"
            mirror_idx = None
        elif len(name_to_indices[key]) > 1:
            status = "ambigua: nombre duplicado"
            mirror_idx = None
        else:
            mirror_idx = name_to_indices[key][0]
            if mirror_idx == idx:
                status = "autorreferencia"
                mirror_idx = None
            else:
                status = "enlazada"
        entries.append({
            "target_index": idx,
            "target_cr": row.get("CR"),
            "target_name": row.get("NAME"),
            "te_label": label,
            "status": status,
            "mirror_index": mirror_idx,
            "mirror_cr": stores.iloc[mirror_idx]["CR"] if mirror_idx is not None else None,
            "mirror_name": stores.iloc[mirror_idx]["NAME"] if mirror_idx is not None else None,
        })
    links = pd.DataFrame(entries)
    counts = links["status"].value_counts().to_dict()
    return links, counts


def fit_te_family_weights(
    stores: pd.DataFrame,
    min_coverage: float = 0.40,
    seed: int = 2026,
    search_steps: int = 96,
) -> dict[str, float]:
    """Fit a compact family-weighted metric to the historical TE mirror labels.

    TE is used as a training target only; it is not added as a predictor. The
    returned weights are normalized to mean 1 and used for new-store ranking.
    With too few resolvable TE labels, equal weights are returned.
    """
    families = list(FEATURE_FAMILIES)
    equal_weights = {family: 1.0 for family in families}
    links, _ = resolve_te_labels(stores)
    linked = links.loc[links["status"].eq("enlazada")]
    if len(linked) < 20:
        return equal_weights

    family_matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    active_by_family = []
    for family in families:
        one_family = {name: float(name == family) for name in families}
        distance, coverage = _pairwise_distance_matrices(stores, one_family)
        family_matrices[family] = (distance, coverage)
        active_by_family.append(np.isfinite(distance).any(axis=1))
    active = np.column_stack(active_by_family)
    crs = stores["CR"].astype(str).to_numpy()
    pairs = [
        (int(row.target_index), int(row.mirror_index))
        for row in linked.itertuples(index=False)
    ]
    total_pairs = len(pairs)

    def evaluate(raw_weights: np.ndarray) -> float:
        weights = np.maximum(np.asarray(raw_weights, dtype=float), 1e-5)
        weights /= weights.sum()
        n = len(stores)
        numerator = np.zeros((n, n), dtype=float)
        denominator = np.zeros((n, n), dtype=float)
        for index, family in enumerate(families):
            distance, coverage = family_matrices[family]
            pair_weight = weights[index] * coverage
            valid = np.isfinite(distance)
            numerator += np.where(valid, distance, 0.0) * pair_weight
            denominator += pair_weight
        combined = np.divide(
            numerator,
            denominator,
            out=np.full((n, n), np.nan, dtype=float),
            where=denominator > 0,
        )
        target_weight = (active * weights[None, :]).sum(axis=1)[:, None]
        coverage = np.divide(
            denominator,
            target_weight,
            out=np.zeros((n, n), dtype=float),
            where=target_weight > 0,
        )
        reciprocal_ranks = []
        for target_idx, mirror_idx in pairs:
            eligible = np.isfinite(combined[target_idx]) & (coverage[target_idx] >= min_coverage)
            eligible[target_idx] = False
            if not eligible[mirror_idx]:
                reciprocal_ranks.append(0.0)
                continue
            distance_to_mirror = combined[target_idx, mirror_idx]
            coverage_to_mirror = coverage[target_idx, mirror_idx]
            same_distance = np.isclose(combined[target_idx], distance_to_mirror, atol=1e-12)
            same_coverage = np.isclose(coverage[target_idx], coverage_to_mirror, atol=1e-12)
            ahead = eligible & (
                (combined[target_idx] < distance_to_mirror - 1e-12)
                | (same_distance & (
                    (coverage[target_idx] > coverage_to_mirror + 1e-12)
                    | (same_coverage & (crs < crs[mirror_idx]))
                ))
            )
            reciprocal_ranks.append(1.0 / (1 + int(ahead.sum())))
        mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
        normalized = weights * len(families)
        regularization = 0.0015 * float(np.mean(np.square(normalized - 1.0)))
        return mrr - regularization

    rng = np.random.default_rng(seed)
    candidates = [np.ones(len(families), dtype=float)]
    candidates.extend(rng.dirichlet(np.full(len(families), 3.0), size=max(0, search_steps - 1)))
    best = candidates[0]
    best_score = evaluate(best)
    for candidate in candidates[1:]:
        score = evaluate(candidate)
        if score > best_score:
            best, best_score = candidate.copy(), score

    best = best / best.sum()
    step = 0.20
    while step >= 0.025:
        improved = False
        for index in range(len(families)):
            for direction in (-1.0, 1.0):
                candidate = best.copy()
                candidate[index] = max(0.01, candidate[index] + direction * step)
                candidate /= candidate.sum()
                score = evaluate(candidate)
                if score > best_score + 1e-12:
                    best, best_score, improved = candidate, score, True
        if not improved:
            step *= 0.5

    best *= len(families) / best.sum()
    return {family: float(best[index]) for index, family in enumerate(families)}


def backtest_te(
    stores: pd.DataFrame,
    family_weights: Mapping[str, float] | None = None,
    min_coverage: float = 0.40,
    progress: Callable[[float], None] | None = None,
) -> tuple[dict[str, float | int], pd.DataFrame, dict[str, int]]:
    """Check whether a leave-one-store-out ranking retrieves historical TE labels."""
    links, label_counts = resolve_te_labels(stores)
    linked = links.loc[links["status"].eq("enlazada")].copy()
    distances, coverage = _pairwise_distance_matrices(stores, family_weights)
    crs = stores["CR"].astype(str).to_numpy()
    rows = []
    total = len(linked)
    for position, (_, link) in enumerate(linked.iterrows(), start=1):
        target_idx = int(link["target_index"])
        mirror_idx = int(link["mirror_index"])
        eligible_candidates = np.isfinite(distances[target_idx]) & (coverage[target_idx] >= min_coverage)
        eligible_candidates[target_idx] = False
        if not eligible_candidates[mirror_idx]:
            rank = np.nan
        else:
            mirror_distance = distances[target_idx, mirror_idx]
            mirror_coverage = coverage[target_idx, mirror_idx]
            same_distance = np.isclose(distances[target_idx], mirror_distance, atol=1e-12)
            same_coverage = np.isclose(coverage[target_idx], mirror_coverage, atol=1e-12)
            ahead = eligible_candidates & (
                (distances[target_idx] < mirror_distance - 1e-12)
                | (same_distance & (
                    (coverage[target_idx] > mirror_coverage + 1e-12)
                    | (same_coverage & (crs < crs[mirror_idx]))
                ))
            )
            rank = int(1 + ahead.sum())
        rows.append({
            "CR": link["target_cr"],
            "PUNTO": link["target_name"],
            "ESPEJO_TE": link["mirror_name"],
            "CR_ESPEJO_TE": link["mirror_cr"],
            "RANGO_MODELO": rank,
            "SIMILITUD_ESPEJO_TE": float(100.0 * (1.0 - distances[target_idx, mirror_idx])) if pd.notna(rank) else np.nan,
        })
        if progress and (position % 10 == 0 or position == total):
            progress(position / max(total, 1))

    results = pd.DataFrame(rows)
    ranks = pd.to_numeric(results.get("RANGO_MODELO", pd.Series(dtype=float)), errors="coerce").dropna()
    eligible = int(len(results))
    metrics: dict[str, float | int] = {
        "n_etiquetas_enlazadas": int(len(linked)),
        "n_con_rango": int(len(ranks)),
        "top1": float((ranks <= 1).mean()) if len(ranks) else float("nan"),
        "top3": float((ranks <= 3).mean()) if len(ranks) else float("nan"),
        "top5": float((ranks <= 5).mean()) if len(ranks) else float("nan"),
        "mrr": float((1.0 / ranks).mean()) if len(ranks) else float("nan"),
        "rango_mediano": float(ranks.median()) if len(ranks) else float("nan"),
        "sin_rango_por_cobertura": int(eligible - len(ranks)),
    }
    return metrics, results, label_counts


def weighted_knn_forecast(
    ranking: pd.DataFrame,
    stores: pd.DataFrame,
    outcome: str,
    k: int = 10,
    include_cr: object | None = None,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Predict the selected sales outcome with an inverse-distance weighted mean."""
    if ranking.empty or outcome not in stores.columns:
        return {}, pd.DataFrame()
    lookup = stores.set_index(stores["CR"].astype(str), drop=False)
    ranked = ranking.copy()
    ranked["_key"] = ranked["CR"].astype(str)
    ranked["_outcome"] = ranked["_key"].map(lookup[outcome])
    ranked["_outcome"] = pd.to_numeric(ranked["_outcome"], errors="coerce")
    ranked = ranked.dropna(subset=["_outcome"]).copy()
    neighbors = ranked.head(max(1, int(k))).copy()
    if include_cr is not None and not neighbors["CR"].astype(str).eq(str(include_cr)).any():
        selected = ranked.loc[ranked["CR"].astype(str).eq(str(include_cr))]
        if not selected.empty:
            other_neighbors = ranked.loc[ranked["CR"].astype(str).ne(str(include_cr))].head(max(0, int(k) - 1))
            neighbors = pd.concat([other_neighbors, selected.head(1)]).sort_index().copy()
    if neighbors.empty:
        return {}, neighbors

    # The small floor prevents a single zero-distance point from dominating.
    neighbors["PESO"] = 1.0 / np.square(neighbors["DISTANCIA"].fillna(1.0) + 0.025)
    neighbors["PESO_RELATIVO"] = neighbors["PESO"] / neighbors["PESO"].sum()
    neighbors["APORTE_ESTIMACION"] = neighbors["PESO_RELATIVO"] * neighbors["_outcome"]
    estimate = float(np.average(neighbors["_outcome"], weights=neighbors["PESO"]))
    values = neighbors["_outcome"].astype(float)
    summary = {
        "estimacion_ponderada": estimate,
        "mediana_vecinos": float(values.median()),
        "p10_vecinos": float(values.quantile(0.10)),
        "p90_vecinos": float(values.quantile(0.90)),
        "n_vecinos": int(len(neighbors)),
    }
    neighbors["VENTA_OBJETIVO"] = neighbors["_outcome"]
    neighbors.drop(columns=["_key", "_outcome"], inplace=True)
    return summary, neighbors


def backtest_sales_loocv(
    stores: pd.DataFrame,
    outcome: str,
    family_weights: Mapping[str, float] | None = None,
    min_coverage: float = 0.40,
    k: int = 10,
    progress: Callable[[float], None] | None = None,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Leave-one-out cross-sectional validation for the analog-sales estimator."""
    if outcome not in stores.columns:
        raise WorkbookError(f"No existe el resultado de venta {outcome} en JUN.")
    valid = stores.loc[pd.to_numeric(stores[outcome], errors="coerce").notna()].copy()
    distances, coverage = _pairwise_distance_matrices(stores, family_weights)
    crs = stores["CR"].astype(str).to_numpy()
    sales = pd.to_numeric(stores[outcome], errors="coerce").to_numpy(dtype=float)
    predictions = []
    total = len(valid)
    for position, (target_idx, target_row) in enumerate(valid.iterrows(), start=1):
        target_idx = int(target_idx)
        eligible_candidates = (
            np.isfinite(distances[target_idx])
            & (coverage[target_idx] >= min_coverage)
            & np.isfinite(sales)
        )
        eligible_candidates[target_idx] = False
        candidate_indices = np.flatnonzero(eligible_candidates)
        if len(candidate_indices):
            ordered = np.lexsort((
                crs[candidate_indices],
                -coverage[target_idx, candidate_indices],
                distances[target_idx, candidate_indices],
            ))
            neighbor_indices = candidate_indices[ordered[:max(1, int(k))]]
            neighbor_distances = distances[target_idx, neighbor_indices]
            neighbor_weights = 1.0 / np.square(neighbor_distances + 0.025)
            estimate = float(np.average(sales[neighbor_indices], weights=neighbor_weights))
            other_sales = sales[np.arange(len(sales)) != target_idx]
            other_sales = other_sales[np.isfinite(other_sales)]
            predictions.append({
                "CR": target_row["CR"],
                "PUNTO": target_row["NAME"],
                "REAL": float(target_row[outcome]),
                "ESTIMADA": estimate,
                "MEDIANA_GLOBAL_LOO": float(np.median(other_sales)) if len(other_sales) else np.nan,
                "N_VECINOS": int(len(neighbor_indices)),
            })
        if progress and (position % 20 == 0 or position == total):
            progress(position / max(total, 1))

    result = pd.DataFrame(predictions)
    if result.empty:
        return {"n": 0}, result
    actual = result["REAL"].astype(float)
    predicted = result["ESTIMADA"].astype(float)
    baseline = result["MEDIANA_GLOBAL_LOO"].astype(float)
    abs_error = (actual - predicted).abs()
    baseline_error = (actual - baseline).abs()
    nonzero = actual.abs() > 1e-12
    metrics: dict[str, float | int] = {
        "n": int(len(result)),
        "mae": float(abs_error.mean()),
        "mae_baseline": float(baseline_error.mean()),
        "mejora_mae_pct": float((1.0 - abs_error.mean() / baseline_error.mean()) * 100.0)
        if baseline_error.mean() > 0 else float("nan"),
        "mape_pct": float((abs_error[nonzero] / actual[nonzero].abs()).mean() * 100.0) if nonzero.any() else float("nan"),
        "mediana_error_abs_pct": float((abs_error[nonzero] / actual[nonzero].abs()).median() * 100.0) if nonzero.any() else float("nan"),
    }
    return metrics, result
