from __future__ import annotations

from html import escape
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


PROJECT_DIR = Path(__file__).resolve().parent.parent

STATION_INPUT_FILE = (
    PROJECT_DIR / "data" / "processed" / "charging_station_level_spatial.csv"
)
BOUNDARY_SHP_FILE = (
    PROJECT_DIR / "data" / "raw" / "boundary_2024_incheon" / "DA_SIG_202406.shp"
)

PROCESSED_OUTPUT_FILE = (
    PROJECT_DIR / "data" / "processed" / "charging_station_dbscan.csv"
)
GEOJSON_OUTPUT_FILE = (
    PROJECT_DIR / "data" / "processed" / "charging_station_dbscan.geojson"
)
PARAMETER_COMPARISON_FILE = (
    PROJECT_DIR / "outputs" / "tables" / "dbscan_parameter_comparison.csv"
)
CLUSTER_SUMMARY_FILE = PROJECT_DIR / "outputs" / "tables" / "dbscan_cluster_summary.csv"
NOISE_STATIONS_FILE = PROJECT_DIR / "outputs" / "tables" / "dbscan_noise_stations.csv"
MAP_OUTPUT_FILE = PROJECT_DIR / "outputs" / "maps" / "dbscan_station_clusters_map.html"
FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"

MIN_SAMPLES_CANDIDATES = [4, 5, 8, 10]
BASE_EPS_CANDIDATES_METERS = [250, 500, 750, 1000, 1500, 2000]

# None이면 비교 결과에서 해석 가능한 후보를 추천해 사용한다.
# 사용자가 특정 값을 확정하고 싶으면 예: FINAL_EPS_METERS = 750, FINAL_MIN_SAMPLES = 5
FINAL_EPS_METERS: int | None = None
FINAL_MIN_SAMPLES: int | None = None

INCHEON_CENTER = [37.4563, 126.7052]
DEFAULT_ZOOM = 10
VALID_LAT_RANGE = (36.5, 38.5)
VALID_LNG_RANGE = (124.0, 127.8)

STATION_REQUIRED_COLUMNS = [
    "station_id",
    "station_name",
    "latitude",
    "longitude",
    "district_2024",
    "charger_count",
    "public_charger_count",
    "representative_address",
]

OUTPUT_COLUMNS = [
    "station_id",
    "station_name",
    "latitude",
    "longitude",
    "district_2024",
    "charger_count",
    "fast_charger_count",
    "slow_charger_count",
    "unknown_speed_count",
    "public_charger_count",
    "limited_charger_count",
    "public_charger_ratio_percent",
    "fast_charger_ratio_percent",
    "representative_address",
    "operator_name",
    "nearest_station_distance_m",
    "dbscan_cluster",
    "is_dbscan_noise",
    "cluster_station_count",
    "cluster_total_charger_count",
    "cluster_public_charger_count",
]


def ensure_output_dirs() -> None:
    """출력 디렉터리를 생성한다."""
    PROCESSED_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    PARAMETER_COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)
    MAP_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    """입력 파일 존재 여부를 확인한다."""
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")


def require_columns(df: pd.DataFrame, required_columns: list[str], name: str) -> None:
    """실제 열 이름 기준으로 필수 열을 확인한다."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"{name} 필수 열이 없습니다: {missing_columns}")


def configure_matplotlib_font() -> None:
    """macOS 한글 표시를 위해 AppleGothic을 우선 사용한다."""
    available_font_names = {font.name for font in fm.fontManager.ttflist}
    if "AppleGothic" in available_font_names:
        plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False


def load_station_data() -> pd.DataFrame:
    """충전소 단위 CSV를 읽고 관측 단위를 검증한다."""
    require_file(STATION_INPUT_FILE)
    station_df = pd.read_csv(STATION_INPUT_FILE, encoding="utf-8-sig", low_memory=False)

    print(f"\n[입력 충전소 단위 데이터] {STATION_INPUT_FILE}")
    print(f"행 수: {len(station_df):,}")
    print(f"열 이름: {station_df.columns.tolist()}")

    require_columns(station_df, STATION_REQUIRED_COLUMNS, "충전소 단위 데이터")

    station_df["station_id"] = station_df["station_id"].astype("string").str.strip()
    station_df["station_name"] = station_df["station_name"].astype("string").str.strip()
    station_df["district_2024"] = station_df["district_2024"].astype("string").str.strip()
    station_df["latitude"] = pd.to_numeric(station_df["latitude"], errors="coerce")
    station_df["longitude"] = pd.to_numeric(station_df["longitude"], errors="coerce")

    numeric_columns = [
        "charger_count",
        "fast_charger_count",
        "slow_charger_count",
        "unknown_speed_count",
        "public_charger_count",
        "limited_charger_count",
        "public_charger_ratio_percent",
        "fast_charger_ratio_percent",
    ]
    for column in numeric_columns:
        if column in station_df.columns:
            station_df[column] = pd.to_numeric(station_df[column], errors="coerce")

    validate_station_input(station_df, numeric_columns)

    return station_df


def validate_station_input(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    """DBSCAN 입력이 충전소 1행 단위인지 검증한다."""
    if df["station_id"].isna().any() or df["station_id"].eq("").any():
        raise ValueError("station_id 결측 또는 빈 값이 있습니다.")

    if df["station_id"].duplicated().any():
        duplicated = df.loc[df["station_id"].duplicated(keep=False), "station_id"].tolist()
        raise ValueError(
            "station_id 중복이 있어 충전소 1행 단위로 볼 수 없습니다: "
            f"{duplicated[:20]}"
        )

    missing_coord_mask = df["latitude"].isna() | df["longitude"].isna()
    if missing_coord_mask.any():
        missing_rows = df.loc[
            missing_coord_mask,
            ["station_id", "station_name", "latitude", "longitude"],
        ]
        raise ValueError(
            "충전소 좌표 결측이 있습니다:\n"
            f"{missing_rows.to_string(index=False)}"
        )

    invalid_coord_mask = (
        ~df["latitude"].between(*VALID_LAT_RANGE)
        | ~df["longitude"].between(*VALID_LNG_RANGE)
    )
    if invalid_coord_mask.any():
        invalid_rows = df.loc[
            invalid_coord_mask,
            ["station_id", "station_name", "latitude", "longitude"],
        ]
        raise ValueError(
            "인천 분석 범위를 벗어난 좌표가 있습니다:\n"
            f"{invalid_rows.to_string(index=False)}"
        )

    if df[numeric_columns].isna().any().any():
        missing_counts = df[numeric_columns].isna().sum()
        raise ValueError(
            "수치 열에 결측 또는 변환 불가 값이 있습니다:\n"
            f"{missing_counts[missing_counts > 0].to_string()}"
        )

    if (df[numeric_columns] < 0).any().any():
        negative_counts = (df[numeric_columns] < 0).sum()
        raise ValueError(
            "수치 열에 음수가 있습니다:\n"
            f"{negative_counts[negative_counts > 0].to_string()}"
        )

    duplicate_coordinate_count = int(df.duplicated(["latitude", "longitude"]).sum())
    print(f"동일 좌표를 공유하는 추가 충전소 행 수: {duplicate_coordinate_count:,}")
    print(
        "DBSCAN 입력은 station_id 1행 단위이며 charger_count를 반복 가중치로 사용하지 않습니다."
    )


def build_projected_geodataframe(station_df: pd.DataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, np.ndarray]:
    """EPSG:4326 GeoDataFrame을 만들고 EPSG:5179 좌표 배열을 반환한다."""
    geometry = [
        Point(longitude, latitude)
        for longitude, latitude in zip(station_df["longitude"], station_df["latitude"])
    ]
    station_gdf = gpd.GeoDataFrame(station_df.copy(), geometry=geometry, crs="EPSG:4326")

    if station_gdf.crs is None or station_gdf.crs.to_epsg() != 4326:
        raise ValueError(f"입력 CRS가 EPSG:4326이 아닙니다: {station_gdf.crs}")

    projected_gdf = station_gdf.to_crs("EPSG:5179")

    if projected_gdf.crs is None or projected_gdf.crs.to_epsg() != 5179:
        raise ValueError(f"투영 CRS가 EPSG:5179가 아닙니다: {projected_gdf.crs}")

    coords = np.column_stack([projected_gdf.geometry.x, projected_gdf.geometry.y])
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()

    # EPSG:5179는 미터 단위 투영좌표계이며 인천 범위에서 수십~수백 km 좌표 범위가 나와야 한다.
    if not (10_000 < x_range < 500_000 and 10_000 < y_range < 500_000):
        raise ValueError(
            "EPSG:5179 변환 후 x/y 범위가 미터 단위로 보기 어렵습니다. "
            f"x_range={x_range:.2f}, y_range={y_range:.2f}"
        )

    print(f"입력 CRS: {station_gdf.crs}")
    print(f"거리 계산 CRS: {projected_gdf.crs}")
    print(f"EPSG:5179 x 범위(m): {x_range:,.2f}")
    print(f"EPSG:5179 y 범위(m): {y_range:,.2f}")

    return station_gdf, projected_gdf, coords


def compute_k_distances(coords: np.ndarray, min_samples: int) -> np.ndarray:
    """min_samples번째 최근접 거리 배열을 반환한다."""
    neighbors = NearestNeighbors(n_neighbors=min_samples, metric="euclidean")
    neighbors.fit(coords)
    distances, _ = neighbors.kneighbors(coords)

    return np.sort(distances[:, -1])


def plot_k_distance(
    sorted_distances: np.ndarray,
    min_samples: int,
    suffix: str,
    ylim: float | None = None,
) -> Path:
    """k-distance plot을 PNG로 저장한다."""
    output_file = FIGURE_DIR / f"k_distance_min_samples_{min_samples}{suffix}.png"
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(np.arange(1, len(sorted_distances) + 1), sorted_distances, color="#2f6f9f")
    ax.set_title(f"k-distance plot (min_samples={min_samples})")
    ax.set_xlabel("충전소 정렬 순서")
    ax.set_ylabel(f"{min_samples}번째 최근접 거리(m)")
    ax.grid(alpha=0.25)

    if ylim is not None:
        ax.set_ylim(0, ylim)

    fig.tight_layout()
    fig.savefig(output_file, dpi=220)
    plt.close(fig)

    return output_file


def run_k_distance_analysis(coords: np.ndarray) -> tuple[pd.DataFrame, list[Path]]:
    """min_samples 후보별 k-distance 분석과 그래프 생성을 수행한다."""
    rows = []
    figure_paths = []

    for min_samples in MIN_SAMPLES_CANDIDATES:
        sorted_distances = compute_k_distances(coords, min_samples)
        p50 = float(np.percentile(sorted_distances, 50))
        p75 = float(np.percentile(sorted_distances, 75))
        p90 = float(np.percentile(sorted_distances, 90))
        p95 = float(np.percentile(sorted_distances, 95))
        p99 = float(np.percentile(sorted_distances, 99))
        max_distance = float(sorted_distances.max())

        rows.append(
            {
                "min_samples": min_samples,
                "p50_m": round(p50, 2),
                "p75_m": round(p75, 2),
                "p90_m": round(p90, 2),
                "p95_m": round(p95, 2),
                "p99_m": round(p99, 2),
                "max_m": round(max_distance, 2),
                "suggested_eps_low_m": int(round(p75 / 50) * 50),
                "suggested_eps_high_m": int(round(p95 / 50) * 50),
            }
        )

        figure_paths.append(plot_k_distance(sorted_distances, min_samples, ""))
        figure_paths.append(
            plot_k_distance(sorted_distances, min_samples, "_zoom_p95", ylim=p95)
        )
        figure_paths.append(
            plot_k_distance(sorted_distances, min_samples, "_zoom_p99", ylim=p99)
        )

    k_summary_df = pd.DataFrame(rows)

    print("\n[k-distance 후보 범위]")
    print(k_summary_df.to_string(index=False))
    print("eps는 자동 확정하지 않고 위 분위수와 실험 결과를 함께 검토합니다.")

    return k_summary_df, figure_paths


def build_eps_candidates(k_summary_df: pd.DataFrame) -> list[int]:
    """기본 eps 후보와 k-distance 분위수 기반 후보를 결합한다."""
    candidates = set(BASE_EPS_CANDIDATES_METERS)

    for column in ["suggested_eps_low_m", "suggested_eps_high_m"]:
        for value in k_summary_df[column].dropna().astype(int):
            if 100 <= value <= 3000:
                candidates.add(int(value))

    return sorted(candidates)


def evaluate_dbscan_parameters(coords: np.ndarray, eps_candidates: list[int]) -> pd.DataFrame:
    """eps와 min_samples 조합별 DBSCAN 지표를 계산한다."""
    rows = []
    station_count = len(coords)

    for eps_meters in eps_candidates:
        for min_samples in MIN_SAMPLES_CANDIDATES:
            labels = DBSCAN(
                eps=eps_meters,
                min_samples=min_samples,
                metric="euclidean",
            ).fit_predict(coords)

            cluster_labels = sorted(label for label in set(labels) if label != -1)
            cluster_count = len(cluster_labels)
            noise_count = int(np.sum(labels == -1))
            noise_ratio = noise_count / station_count * 100
            cluster_sizes = [
                int(np.sum(labels == label))
                for label in cluster_labels
            ]

            if cluster_sizes:
                largest_cluster_size = max(cluster_sizes)
                median_cluster_size = float(np.median(cluster_sizes))
            else:
                largest_cluster_size = 0
                median_cluster_size = np.nan

            largest_cluster_ratio = largest_cluster_size / station_count * 100

            non_noise_mask = labels != -1
            non_noise_labels = labels[non_noise_mask]
            unique_non_noise_labels = set(non_noise_labels)

            if len(unique_non_noise_labels) >= 2:
                silhouette = float(silhouette_score(coords[non_noise_mask], non_noise_labels))
                davies_bouldin = float(
                    davies_bouldin_score(coords[non_noise_mask], non_noise_labels)
                )
            else:
                # 군집이 2개 미만이면 군집 간 분리도를 정의할 수 없어 NaN으로 둔다.
                silhouette = np.nan
                davies_bouldin = np.nan

            rows.append(
                {
                    "eps_meters": eps_meters,
                    "min_samples": min_samples,
                    "cluster_count": cluster_count,
                    "noise_station_count": noise_count,
                    "noise_ratio_percent": round(noise_ratio, 2),
                    "largest_cluster_station_count": largest_cluster_size,
                    "largest_cluster_ratio_percent": round(largest_cluster_ratio, 2),
                    "median_cluster_size": round(median_cluster_size, 2)
                    if not np.isnan(median_cluster_size)
                    else np.nan,
                    "silhouette_score_without_noise": round(silhouette, 4)
                    if not np.isnan(silhouette)
                    else np.nan,
                    "davies_bouldin_without_noise": round(davies_bouldin, 4)
                    if not np.isnan(davies_bouldin)
                    else np.nan,
                }
            )

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(PARAMETER_COMPARISON_FILE, index=False, encoding="utf-8-sig")

    print("\n[DBSCAN 파라미터 실험 결과]")
    print(comparison_df.to_string(index=False))
    print(
        "군집이 2개 미만인 조합은 실루엣과 Davies-Bouldin을 계산할 수 없어 NaN으로 둡니다."
    )

    return comparison_df


def recommend_parameters(comparison_df: pd.DataFrame) -> tuple[int, int, pd.Series]:
    """해석 가능성을 고려해 최종 파라미터를 추천한다."""
    if FINAL_EPS_METERS is not None and FINAL_MIN_SAMPLES is not None:
        selected = comparison_df.loc[
            (comparison_df["eps_meters"] == FINAL_EPS_METERS)
            & (comparison_df["min_samples"] == FINAL_MIN_SAMPLES)
        ]
        if selected.empty:
            raise ValueError(
                "FINAL_EPS_METERS와 FINAL_MIN_SAMPLES 조합이 실험 결과에 없습니다: "
                f"{FINAL_EPS_METERS}, {FINAL_MIN_SAMPLES}"
            )

        row = selected.iloc[0]
        return int(row["eps_meters"]), int(row["min_samples"]), row

    candidate_df = comparison_df.loc[
        comparison_df["cluster_count"].between(5, 45)
        & comparison_df["noise_ratio_percent"].between(2, 45)
        & (comparison_df["largest_cluster_ratio_percent"] <= 50)
    ].copy()

    if candidate_df.empty:
        candidate_df = comparison_df.loc[
            (comparison_df["cluster_count"] >= 2)
            & (comparison_df["largest_cluster_ratio_percent"] <= 75)
        ].copy()

    if candidate_df.empty:
        candidate_df = comparison_df.copy()

    candidate_df["silhouette_rank"] = candidate_df["silhouette_score_without_noise"].rank(
        ascending=False,
        method="min",
        na_option="bottom",
    )
    candidate_df["noise_target_gap"] = (candidate_df["noise_ratio_percent"] - 15).abs()
    candidate_df["largest_target_gap"] = (
        candidate_df["largest_cluster_ratio_percent"] - 35
    ).abs()
    candidate_df["cluster_target_gap"] = (candidate_df["cluster_count"] - 25).abs()
    candidate_df["selection_score"] = (
        candidate_df["silhouette_rank"] * 2
        + candidate_df["noise_target_gap"] * 0.7
        + candidate_df["largest_target_gap"] * 0.5
        + candidate_df["cluster_target_gap"] * 1.5
    )

    row = candidate_df.sort_values(
        ["selection_score", "noise_ratio_percent", "largest_cluster_ratio_percent"]
    ).iloc[0]

    return int(row["eps_meters"]), int(row["min_samples"]), row


def fit_dbscan(coords: np.ndarray, eps_meters: int, min_samples: int) -> np.ndarray:
    """최종 DBSCAN 군집 라벨을 계산한다."""
    return DBSCAN(
        eps=eps_meters,
        min_samples=min_samples,
        metric="euclidean",
    ).fit_predict(coords)


def add_nearest_station_distance(df: pd.DataFrame, coords: np.ndarray) -> pd.DataFrame:
    """각 충전소의 최근접 충전소 거리를 추가한다."""
    result_df = df.copy()
    neighbors = NearestNeighbors(n_neighbors=2, metric="euclidean")
    neighbors.fit(coords)
    distances, _ = neighbors.kneighbors(coords)
    result_df["nearest_station_distance_m"] = distances[:, 1].round(2)

    return result_df


def add_cluster_attributes(station_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """최종 군집 라벨과 군집 단위 집계값을 추가한다."""
    result_df = station_df.copy()
    result_df["dbscan_cluster"] = labels.astype(int)
    result_df["is_dbscan_noise"] = result_df["dbscan_cluster"].eq(-1)

    cluster_stats = (
        result_df.loc[~result_df["is_dbscan_noise"]]
        .groupby("dbscan_cluster")
        .agg(
            cluster_station_count=("station_id", "count"),
            cluster_total_charger_count=("charger_count", "sum"),
            cluster_public_charger_count=("public_charger_count", "sum"),
        )
    )

    result_df = result_df.merge(
        cluster_stats,
        on="dbscan_cluster",
        how="left",
        validate="many_to_one",
    )

    noise_mask = result_df["is_dbscan_noise"]
    result_df.loc[noise_mask, "cluster_station_count"] = 1
    result_df.loc[noise_mask, "cluster_total_charger_count"] = result_df.loc[
        noise_mask,
        "charger_count",
    ]
    result_df.loc[noise_mask, "cluster_public_charger_count"] = result_df.loc[
        noise_mask,
        "public_charger_count",
    ]

    count_columns = [
        "cluster_station_count",
        "cluster_total_charger_count",
        "cluster_public_charger_count",
    ]
    result_df[count_columns] = result_df[count_columns].astype(int)

    return result_df


def build_cluster_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    """noise를 제외한 군집 요약표를 생성한다."""
    summary_df = (
        result_df.loc[~result_df["is_dbscan_noise"]]
        .groupby("dbscan_cluster", as_index=False)
        .agg(
            station_count=("station_id", "count"),
            total_charger_count=("charger_count", "sum"),
            public_charger_count=("public_charger_count", "sum"),
            avg_nearest_station_distance_m=("nearest_station_distance_m", "mean"),
            district_count=("district_2024", "nunique"),
        )
        .sort_values(["station_count", "total_charger_count"], ascending=False)
        .reset_index(drop=True)
    )

    if not summary_df.empty:
        summary_df["avg_nearest_station_distance_m"] = summary_df[
            "avg_nearest_station_distance_m"
        ].round(2)

    return summary_df


def build_noise_stations(result_df: pd.DataFrame) -> pd.DataFrame:
    """noise 충전소 목록을 생성한다."""
    columns = [
        "station_id",
        "station_name",
        "district_2024",
        "latitude",
        "longitude",
        "charger_count",
        "public_charger_count",
        "nearest_station_distance_m",
        "representative_address",
    ]

    return (
        result_df.loc[result_df["is_dbscan_noise"], columns]
        .sort_values("nearest_station_distance_m", ascending=False)
        .reset_index(drop=True)
    )


def validate_final_outputs(
    input_station_count: int,
    result_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
) -> None:
    """최종 DBSCAN 결과를 검증한다."""
    if len(result_df) != input_station_count:
        raise AssertionError(
            f"입력 충전소 수와 결과 충전소 수가 다릅니다: "
            f"{input_station_count:,} vs {len(result_df):,}"
        )

    if result_df["station_id"].duplicated().any():
        raise AssertionError("최종 결과에 station_id 중복이 있습니다.")

    if result_df["dbscan_cluster"].isna().any():
        raise AssertionError("군집 번호가 없는 행이 있습니다.")

    if not result_df.loc[result_df["is_dbscan_noise"], "dbscan_cluster"].eq(-1).all():
        raise AssertionError("noise 행의 dbscan_cluster가 -1이 아닙니다.")

    cluster_station_sum = int(cluster_summary_df["station_count"].sum())
    noise_count = int(result_df["is_dbscan_noise"].sum())

    if cluster_station_sum + noise_count != input_station_count:
        raise AssertionError(
            "군집 요약의 충전소 수 합계 + noise 수가 전체 충전소 수와 다릅니다. "
            f"cluster={cluster_station_sum:,}, noise={noise_count:,}, "
            f"total={input_station_count:,}"
        )


def save_final_outputs(
    result_df: pd.DataFrame,
    station_gdf: gpd.GeoDataFrame,
    cluster_summary_df: pd.DataFrame,
    noise_df: pd.DataFrame,
) -> list[Path]:
    """최종 CSV, GeoJSON, 요약표를 저장한다."""
    output_df = result_df[OUTPUT_COLUMNS].copy()
    output_df.to_csv(PROCESSED_OUTPUT_FILE, index=False, encoding="utf-8-sig")

    output_gdf = station_gdf[["geometry"]].join(output_df)
    output_gdf = output_gdf.set_geometry("geometry").set_crs("EPSG:4326")
    output_gdf.to_file(GEOJSON_OUTPUT_FILE, driver="GeoJSON", encoding="utf-8")

    cluster_summary_df.to_csv(CLUSTER_SUMMARY_FILE, index=False, encoding="utf-8-sig")
    noise_df.to_csv(NOISE_STATIONS_FILE, index=False, encoding="utf-8-sig")

    return [
        PROCESSED_OUTPUT_FILE,
        GEOJSON_OUTPUT_FILE,
        PARAMETER_COMPARISON_FILE,
        CLUSTER_SUMMARY_FILE,
        NOISE_STATIONS_FILE,
    ]


def load_boundary() -> gpd.GeoDataFrame:
    """2024년 인천 군·구 경계를 EPSG:4326으로 읽는다."""
    require_file(BOUNDARY_SHP_FILE)
    boundary_gdf = gpd.read_file(BOUNDARY_SHP_FILE, encoding="cp949")

    if "SIGUNGU_NM" not in boundary_gdf.columns:
        raise KeyError("경계 SHP에 SIGUNGU_NM 열이 없습니다.")

    boundary_gdf = boundary_gdf.rename(columns={"SIGUNGU_NM": "district_2024"})
    boundary_gdf["district_2024"] = boundary_gdf["district_2024"].astype("string").str.strip()

    if boundary_gdf.crs is None:
        raise ValueError("경계 데이터 CRS가 없습니다.")

    boundary_gdf = boundary_gdf.to_crs("EPSG:4326")
    return boundary_gdf[["district_2024", "geometry"]].copy()


def add_boundary_layer(fmap: folium.Map, boundary_gdf: gpd.GeoDataFrame) -> None:
    """지도에 행정구역 경계 레이어를 추가한다."""
    folium.GeoJson(
        boundary_gdf,
        name="2024년 인천 군·구 경계",
        style_function=lambda _: {
            "fillColor": "#ffffff",
            "color": "#333333",
            "weight": 1.2,
            "fillOpacity": 0.03,
        },
        tooltip=folium.GeoJsonTooltip(fields=["district_2024"], aliases=["군·구"]),
    ).add_to(fmap)


def cluster_color(cluster_label: int) -> str:
    """군집 번호별 색상을 반환한다."""
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#005f73",
        "#9b2226",
        "#0a9396",
        "#ee9b00",
        "#6a4c93",
    ]
    return colors[cluster_label % len(colors)]


def build_popup(row: pd.Series) -> folium.Popup:
    """DBSCAN 지도 팝업을 만든다."""
    cluster_text = "noise" if row["dbscan_cluster"] == -1 else str(row["dbscan_cluster"])
    html = f"""
    <div style="font-family:Arial, sans-serif; font-size:12px; line-height:1.45;">
      <b>{escape(str(row['station_name']))}</b><br>
      군집: {escape(cluster_text)}<br>
      충전기 수: {int(row['charger_count']):,}<br>
      일반 이용 가능: {int(row['public_charger_count']):,}<br>
      최근접 충전소 거리: {float(row['nearest_station_distance_m']):,.1f}m
    </div>
    """
    return folium.Popup(html, max_width=300)


def create_dbscan_map(result_df: pd.DataFrame, boundary_gdf: gpd.GeoDataFrame) -> Path:
    """DBSCAN 군집 결과 Folium 지도를 생성한다."""
    fmap = folium.Map(
        location=INCHEON_CENTER,
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    title_html = """
    <div style="
        position: fixed;
        top: 12px;
        left: 50px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.92);
        padding: 8px 12px;
        border: 1px solid #999;
        border-radius: 4px;
        font-family: Arial, sans-serif;
    ">
      <div style="font-size:16px;font-weight:700;">충전소 단위 DBSCAN 군집</div>
      <div style="font-size:12px;color:#555;">각 충전소는 1개 관측값이며 charger_count 반복 가중치를 사용하지 않음</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))
    add_boundary_layer(fmap, boundary_gdf)

    cluster_labels = sorted(label for label in result_df["dbscan_cluster"].unique() if label != -1)
    for label in cluster_labels:
        layer = MarkerCluster(name=f"군집 {label}").add_to(fmap)
        cluster_df = result_df.loc[result_df["dbscan_cluster"].eq(label)]
        color = cluster_color(int(label))

        for _, row in cluster_df.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=min(11, 3 + np.sqrt(float(row["charger_count"]))),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.72,
                weight=1,
                tooltip=f"{row['station_name']} | 군집 {label}",
                popup=build_popup(row),
            ).add_to(layer)

    noise_layer = folium.FeatureGroup(name="noise (-1)", show=True).add_to(fmap)
    noise_df = result_df.loc[result_df["is_dbscan_noise"]]
    for _, row in noise_df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=min(10, 3 + np.sqrt(float(row["charger_count"]))),
            color="#111111",
            fill=True,
            fill_color="#111111",
            fill_opacity=0.7,
            weight=1,
            tooltip=f"{row['station_name']} | noise",
            popup=build_popup(row),
        ).add_to(noise_layer)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(MAP_OUTPUT_FILE)

    return MAP_OUTPUT_FILE


def print_terminal_summary(
    k_summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    recommended_row: pd.Series,
    final_eps: int,
    final_min_samples: int,
    result_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    output_paths: list[Path],
    figure_paths: list[Path],
    map_path: Path,
) -> None:
    """분석 결과 요약을 출력한다."""
    print("\n[최종 파라미터 추천]")
    print(
        "최고 실루엣 점수만이 아니라 노이즈 비율, 최대 군집 비중, "
        "군집 수 해석 가능성을 함께 고려했습니다."
    )
    print(f"추천/적용 eps_meters: {final_eps}")
    print(f"추천/적용 min_samples: {final_min_samples}")
    print(recommended_row.to_string())

    print("\n[최종 군집 요약]")
    print(f"입력/결과 충전소 수: {len(result_df):,}")
    print(f"군집 수(noise 제외): {len(cluster_summary_df):,}")
    print(f"noise 충전소 수: {int(result_df['is_dbscan_noise'].sum()):,}")
    print(
        f"noise 비율: {result_df['is_dbscan_noise'].mean() * 100:.2f}%"
    )
    if not cluster_summary_df.empty:
        print("\n[상위 군집]")
        print(cluster_summary_df.head(10).to_string(index=False))

    print("\n[저장 파일]")
    for path in output_paths:
        print(path)

    print("\n[k-distance PNG]")
    for path in figure_paths:
        print(path)

    print("\n[DBSCAN 지도]")
    print(f"{map_path} ({map_path.stat().st_size / 1024:,.1f} KB)")


def main() -> None:
    ensure_output_dirs()
    configure_matplotlib_font()

    station_df = load_station_data()
    input_station_count = len(station_df)
    station_gdf, _, coords = build_projected_geodataframe(station_df)

    k_summary_df, figure_paths = run_k_distance_analysis(coords)
    eps_candidates = build_eps_candidates(k_summary_df)
    print(f"\n[실험 eps 후보(m)] {eps_candidates}")

    comparison_df = evaluate_dbscan_parameters(coords, eps_candidates)
    final_eps, final_min_samples, recommended_row = recommend_parameters(comparison_df)
    labels = fit_dbscan(coords, final_eps, final_min_samples)

    result_df = add_nearest_station_distance(station_df, coords)
    result_df = add_cluster_attributes(result_df, labels)
    cluster_summary_df = build_cluster_summary(result_df)
    noise_df = build_noise_stations(result_df)
    validate_final_outputs(input_station_count, result_df, cluster_summary_df)

    output_paths = save_final_outputs(
        result_df,
        station_gdf,
        cluster_summary_df,
        noise_df,
    )
    boundary_gdf = load_boundary()
    map_path = create_dbscan_map(result_df, boundary_gdf)

    print_terminal_summary(
        k_summary_df,
        comparison_df,
        recommended_row,
        final_eps,
        final_min_samples,
        result_df,
        cluster_summary_df,
        output_paths,
        figure_paths,
        map_path,
    )


if __name__ == "__main__":
    main()
