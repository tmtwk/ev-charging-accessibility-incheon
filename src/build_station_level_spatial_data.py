from __future__ import annotations

from itertools import combinations
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_CANDIDATES = [
    PROJECT_DIR / "data" / "processed" / "charging_stations_incheon_2024_district.csv",
    PROJECT_DIR / "data" / "processed" / "charging_stations_incheon_spatial.csv",
    PROJECT_DIR / "data" / "processed" / "charging_stations_incheon_final.csv",
]

CSV_OUTPUT_FILE = (
    PROJECT_DIR / "data" / "processed" / "charging_station_level_spatial.csv"
)
GEOJSON_OUTPUT_FILE = (
    PROJECT_DIR / "data" / "processed" / "charging_station_level_spatial.geojson"
)
COORDINATE_CONFLICT_OUTPUT_FILE = (
    PROJECT_DIR / "outputs" / "tables" / "station_coordinate_conflict_review.csv"
)
SUMMARY_OUTPUT_FILE = PROJECT_DIR / "outputs" / "tables" / "station_level_summary.csv"

REQUIRED_COLUMNS = [
    "charger_uid",
    "statId",
    "statNm",
    "lat",
    "lng",
    "district_2024",
    "output",
    "publicly_accessible",
    "is_limited",
]

OPTIONAL_TEXT_COLUMNS = {
    "representative_address": ["addr", "addrDetail", "location"],
    "operator_name": ["busiNm", "bnm"],
}

SPEED_CLASS_CANDIDATES = [
    "charge_speed_class",
    "charging_speed_class",
    "speed_class",
]

FAST_CHARGING_THRESHOLD_KW = 50
CONFLICT_DISTANCE_THRESHOLD_M = 100
EARTH_RADIUS_M = 6_371_000

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
]

CONFLICT_COLUMNS = [
    "station_id",
    "station_name",
    "district_2024",
    "charger_count",
    "unique_coordinate_count",
    "max_coordinate_distance_m",
    "coordinate_values",
    "representative_address",
]


def ensure_output_dirs() -> None:
    """결과 저장 디렉터리를 생성한다."""
    CSV_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    COORDINATE_CONFLICT_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    """CSV를 읽고 실제 열 이름을 출력한다."""
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    print(f"\n[입력 후보 확인] {path}")
    print(f"행 수: {len(df):,}")
    print(f"열 이름: {df.columns.tolist()}")
    return df


def choose_input_dataframe() -> tuple[pd.DataFrame, Path]:
    """2024년 경계 결합 파일을 우선 사용하고 필수 열을 확인한다."""
    loaded: list[tuple[Path, pd.DataFrame]] = []

    for path in INPUT_CANDIDATES:
        if not path.exists():
            print(f"\n[입력 후보 확인] {path}")
            print("파일 없음")
            continue

        df = read_csv(path)
        loaded.append((path, df))

    if not loaded:
        raise FileNotFoundError("사용 가능한 충전소 전처리 파일이 없습니다.")

    for path, df in loaded:
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if not missing_columns:
            print(f"\n선택한 입력 파일: {path}")
            return df.copy(), path

        print(f"필수 열 부족: {missing_columns}")

    raise KeyError("입력 후보에서 충전소 단위 집계에 필요한 열을 모두 찾지 못했습니다.")


def normalize_boolean_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """문자열 Boolean 값을 실제 Boolean으로 변환한다."""
    normalized_df = df.copy()
    boolean_map = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "y": True,
        "n": False,
        "yes": True,
        "no": False,
    }

    for column in columns:
        if normalized_df[column].dtype == "bool":
            continue

        converted = (
            normalized_df[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .map(boolean_map)
        )
        invalid_mask = normalized_df[column].notna() & converted.isna()

        if invalid_mask.any():
            invalid_values = (
                normalized_df.loc[invalid_mask, column].drop_duplicates().tolist()
            )
            raise ValueError(f"{column} 열에서 Boolean으로 변환할 수 없는 값: {invalid_values}")

        normalized_df[column] = converted.fillna(False).astype(bool)

    return normalized_df


def prepare_charger_level_data(df: pd.DataFrame) -> pd.DataFrame:
    """충전기 1행 단위 데이터를 분석 가능한 matched 데이터로 정리한다."""
    prepared_df = df.copy()

    if "boundary_match_status" in prepared_df.columns:
        before_count = len(prepared_df)
        prepared_df = prepared_df.loc[
            prepared_df["boundary_match_status"].eq("matched")
        ].copy()
        print(
            "2024년 군·구 경계 matched 행만 사용: "
            f"{len(prepared_df):,} / {before_count:,}"
        )

    prepared_df["charger_uid"] = prepared_df["charger_uid"].astype("string").str.strip()
    prepared_df["statId"] = prepared_df["statId"].astype("string").str.strip()
    prepared_df["statNm"] = prepared_df["statNm"].astype("string").str.strip()
    prepared_df["district_2024"] = prepared_df["district_2024"].astype("string").str.strip()
    prepared_df["lat"] = pd.to_numeric(prepared_df["lat"], errors="coerce")
    prepared_df["lng"] = pd.to_numeric(prepared_df["lng"], errors="coerce")

    validate_charger_level_rows(prepared_df)
    prepared_df = normalize_boolean_columns(
        prepared_df,
        ["publicly_accessible", "is_limited"],
    )
    prepared_df = add_speed_class(prepared_df)

    return prepared_df


def validate_charger_level_rows(df: pd.DataFrame) -> None:
    """충전기 1행 단위 여부와 좌표 유효성을 검증한다."""
    if df["charger_uid"].isna().any() or df["charger_uid"].eq("").any():
        raise ValueError("charger_uid 결측 또는 빈 값이 있습니다.")

    duplicate_mask = df["charger_uid"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_rows = df.loc[
            duplicate_mask,
            ["charger_uid", "statId", "statNm"],
        ].sort_values("charger_uid")
        raise ValueError(
            "charger_uid 중복이 있어 충전기 1행 단위로 볼 수 없습니다:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    missing_required = df[["statId", "statNm", "district_2024", "lat", "lng"]].isna()
    if missing_required.any().any():
        missing_counts = missing_required.sum()
        raise ValueError(
            "필수 열에 결측이 있습니다:\n"
            f"{missing_counts[missing_counts > 0].to_string()}"
        )

    invalid_coord_mask = ~df["lat"].between(-90, 90) | ~df["lng"].between(-180, 180)
    if invalid_coord_mask.any():
        invalid_rows = df.loc[
            invalid_coord_mask,
            ["charger_uid", "statId", "statNm", "lat", "lng"],
        ]
        raise ValueError(
            "위도·경도 범위를 벗어난 행이 있습니다:\n"
            f"{invalid_rows.to_string(index=False)}"
        )

    for column in ["coordinate_missing_final", "coordinate_outlier_final"]:
        if column in df.columns:
            normalized = normalize_boolean_columns(df[[column]].copy(), [column])
            if normalized[column].any():
                invalid_rows = df.loc[
                    normalized[column],
                    ["charger_uid", "statId", "statNm", "lat", "lng"],
                ]
                raise ValueError(
                    f"{column}=True인 좌표 검토 대상 행이 있습니다:\n"
                    f"{invalid_rows.to_string(index=False)}"
                )


def add_speed_class(df: pd.DataFrame) -> pd.DataFrame:
    """기존 속도 분류를 재사용하고, 없으면 output 기준 임시 분류를 추가한다."""
    speed_df = df.copy()
    existing_speed_column = next(
        (column for column in SPEED_CLASS_CANDIDATES if column in speed_df.columns),
        None,
    )

    if existing_speed_column is not None:
        speed_df["speed_class"] = (
            speed_df[existing_speed_column].astype("string").str.strip().str.lower()
        )
        return speed_df

    # 공식 충전기 유형 코드가 아니라 충전용량(output) 기준의 임시 분석 분류이다.
    speed_df["output"] = pd.to_numeric(speed_df["output"], errors="coerce")
    speed_df["speed_class"] = "unknown"
    speed_df.loc[speed_df["output"] >= FAST_CHARGING_THRESHOLD_KW, "speed_class"] = "fast"
    speed_df.loc[
        speed_df["output"].notna() & (speed_df["output"] < FAST_CHARGING_THRESHOLD_KW),
        "speed_class",
    ] = "slow"

    return speed_df


def haversine_distance_m(
    lat1: float,
    lng1: float,
    lat2: float,
    lng2: float,
) -> float:
    """두 WGS84 좌표 사이의 대권거리를 m 단위로 계산한다."""
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lng = radians(lng2 - lng1)

    a = (
        sin(delta_lat / 2) ** 2
        + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lng / 2) ** 2
    )
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return EARTH_RADIUS_M * c


def max_coordinate_distance_m(coordinates: pd.DataFrame) -> float:
    """충전소 내 고유 좌표 사이 최대 거리를 계산한다."""
    unique_coordinates = coordinates.drop_duplicates().to_numpy()

    if len(unique_coordinates) <= 1:
        return 0.0

    max_distance = 0.0
    for first, second in combinations(unique_coordinates, 2):
        distance = haversine_distance_m(first[0], first[1], second[0], second[1])
        max_distance = max(max_distance, distance)

    return max_distance


def first_non_empty(series: pd.Series) -> str:
    """대표 문자열 값을 반환한다."""
    valid_values = series.dropna().astype("string").str.strip()
    valid_values = valid_values.loc[valid_values.ne("")]

    if valid_values.empty:
        return ""

    return str(valid_values.iloc[0])


def mode_or_first(series: pd.Series) -> str:
    """최빈값을 우선 사용하고 동률이면 첫 값을 사용한다."""
    valid_values = series.dropna().astype("string").str.strip()
    valid_values = valid_values.loc[valid_values.ne("")]

    if valid_values.empty:
        return ""

    mode_values = valid_values.mode()
    if not mode_values.empty:
        return str(mode_values.iloc[0])

    return str(valid_values.iloc[0])


def build_representative_address(group: pd.DataFrame) -> str:
    """주소 관련 열에서 대표 주소를 만든다."""
    values = []
    for column in OPTIONAL_TEXT_COLUMNS["representative_address"]:
        if column in group.columns:
            value = first_non_empty(group[column])
            if value:
                values.append(value)

    return " ".join(dict.fromkeys(values))


def get_operator_name(group: pd.DataFrame) -> str:
    """운영기관명을 반환한다."""
    for column in OPTIONAL_TEXT_COLUMNS["operator_name"]:
        if column in group.columns:
            value = mode_or_first(group[column])
            if value:
                return value

    return ""


def summarize_station_group(group: pd.DataFrame) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """statId 단위로 충전소 지표를 만들고 좌표 충돌이면 review 행으로 분리한다."""
    coordinates = group[["lat", "lng"]].drop_duplicates()
    unique_coordinate_count = len(coordinates)
    max_distance = max_coordinate_distance_m(coordinates)

    station_id = str(group["statId"].iloc[0])
    station_name = mode_or_first(group["statNm"])
    representative_address = build_representative_address(group)
    district_2024 = mode_or_first(group["district_2024"])

    if max_distance >= CONFLICT_DISTANCE_THRESHOLD_M:
        coordinate_values = "; ".join(
            f"{row.lat:.8f},{row.lng:.8f}" for row in coordinates.itertuples(index=False)
        )
        conflict_row = {
            "station_id": station_id,
            "station_name": station_name,
            "district_2024": district_2024,
            "charger_count": int(group["charger_uid"].nunique()),
            "unique_coordinate_count": unique_coordinate_count,
            "max_coordinate_distance_m": round(max_distance, 2),
            "coordinate_values": coordinate_values,
            "representative_address": representative_address,
        }
        return None, conflict_row

    # 완전히 같은 좌표는 그대로 사용하고, 100m 미만의 미세 차이는 중앙값을 사용한다.
    if unique_coordinate_count == 1:
        latitude = float(coordinates["lat"].iloc[0])
        longitude = float(coordinates["lng"].iloc[0])
    else:
        latitude = float(group["lat"].median())
        longitude = float(group["lng"].median())

    charger_count = int(group["charger_uid"].nunique())
    fast_charger_count = int(group.loc[group["speed_class"].eq("fast"), "charger_uid"].nunique())
    slow_charger_count = int(group.loc[group["speed_class"].eq("slow"), "charger_uid"].nunique())
    unknown_speed_count = int(
        group.loc[group["speed_class"].eq("unknown"), "charger_uid"].nunique()
    )
    public_charger_count = int(
        group.loc[group["publicly_accessible"], "charger_uid"].nunique()
    )
    limited_charger_count = int(group.loc[group["is_limited"], "charger_uid"].nunique())

    station_row = {
        "station_id": station_id,
        "station_name": station_name,
        "latitude": latitude,
        "longitude": longitude,
        "district_2024": district_2024,
        "charger_count": charger_count,
        "fast_charger_count": fast_charger_count,
        "slow_charger_count": slow_charger_count,
        "unknown_speed_count": unknown_speed_count,
        "public_charger_count": public_charger_count,
        "limited_charger_count": limited_charger_count,
        "public_charger_ratio_percent": round(public_charger_count / charger_count * 100, 2),
        "fast_charger_ratio_percent": round(fast_charger_count / charger_count * 100, 2),
        "representative_address": representative_address,
        "operator_name": get_operator_name(group),
    }

    return station_row, None


def build_station_level_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """충전기 데이터를 충전소 단위로 집계한다."""
    station_rows: list[dict[str, object]] = []
    conflict_rows: list[dict[str, object]] = []

    for _, group in df.groupby("statId", sort=False):
        station_row, conflict_row = summarize_station_group(group)

        if station_row is not None:
            station_rows.append(station_row)

        if conflict_row is not None:
            conflict_rows.append(conflict_row)

    station_df = pd.DataFrame(station_rows, columns=OUTPUT_COLUMNS)
    conflict_df = pd.DataFrame(conflict_rows, columns=CONFLICT_COLUMNS)

    return station_df, conflict_df


def validate_station_level_output(
    station_df: pd.DataFrame,
    conflict_df: pd.DataFrame,
    input_charger_count: int,
) -> None:
    """충전소 단위 결과의 중복, 좌표, 합계를 검증한다."""
    if station_df["station_id"].duplicated().any():
        duplicated = station_df.loc[
            station_df["station_id"].duplicated(keep=False),
            "station_id",
        ].tolist()
        raise AssertionError(f"station_id 중복이 있습니다: {duplicated}")

    missing_required = station_df[OUTPUT_COLUMNS].isna()
    if missing_required.any().any():
        missing_counts = missing_required.sum()
        raise AssertionError(
            "충전소 단위 주요 열에 결측이 있습니다:\n"
            f"{missing_counts[missing_counts > 0].to_string()}"
        )

    invalid_coord_mask = (
        ~station_df["latitude"].between(-90, 90)
        | ~station_df["longitude"].between(-180, 180)
    )
    if invalid_coord_mask.any():
        invalid_rows = station_df.loc[
            invalid_coord_mask,
            ["station_id", "station_name", "latitude", "longitude"],
        ]
        raise AssertionError(
            "충전소 단위 결과에 비정상 좌표가 있습니다:\n"
            f"{invalid_rows.to_string(index=False)}"
        )

    station_charger_sum = int(station_df["charger_count"].sum())
    conflict_charger_sum = (
        int(conflict_df["charger_count"].sum()) if not conflict_df.empty else 0
    )

    if conflict_charger_sum > 0:
        print(
            "좌표 충돌 충전소가 있어 최종 충전소 공간 데이터에서는 제외했습니다. "
            f"충돌 충전기 수: {conflict_charger_sum:,}"
        )

    if station_charger_sum + conflict_charger_sum != input_charger_count:
        raise AssertionError(
            "충전소별 충전기 수 합계가 입력 충전기 수와 일치하지 않습니다. "
            f"station={station_charger_sum:,}, conflict={conflict_charger_sum:,}, "
            f"input={input_charger_count:,}"
        )

    if conflict_charger_sum == 0 and station_charger_sum != input_charger_count:
        raise AssertionError(
            "충전소별 충전기 수 합계가 입력 충전기 수와 일치하지 않습니다. "
            f"station={station_charger_sum:,}, input={input_charger_count:,}"
        )

    speed_total = (
        station_df["fast_charger_count"]
        + station_df["slow_charger_count"]
        + station_df["unknown_speed_count"]
    )
    if not speed_total.eq(station_df["charger_count"]).all():
        invalid_rows = station_df.loc[
            ~speed_total.eq(station_df["charger_count"]),
            [
                "station_id",
                "charger_count",
                "fast_charger_count",
                "slow_charger_count",
                "unknown_speed_count",
            ],
        ]
        raise AssertionError(
            "충전소별 fast + slow + unknown 합계가 charger_count와 다릅니다:\n"
            f"{invalid_rows.to_string(index=False)}"
        )

    numeric_df = station_df.select_dtypes(include=["number"])
    if np.isinf(numeric_df.to_numpy()).any():
        raise AssertionError("충전소 단위 수치 열에 무한대 값이 있습니다.")

    if (numeric_df < 0).any().any():
        negative_counts = (numeric_df < 0).sum()
        raise AssertionError(
            "충전소 단위 수치 열에 음수가 있습니다:\n"
            f"{negative_counts[negative_counts > 0].to_string()}"
        )


def build_station_summary(station_df: pd.DataFrame) -> pd.DataFrame:
    """군·구별 충전소 단위 요약표를 생성한다."""
    summary_df = (
        station_df.groupby("district_2024", as_index=False)
        .agg(
            station_count=("station_id", "nunique"),
            charger_count=("charger_count", "sum"),
            avg_chargers_per_station=("charger_count", "mean"),
            median_chargers_per_station=("charger_count", "median"),
            max_chargers_per_station=("charger_count", "max"),
            public_charger_count=("public_charger_count", "sum"),
            fast_charger_count=("fast_charger_count", "sum"),
        )
        .sort_values("district_2024")
        .reset_index(drop=True)
    )

    summary_df["avg_chargers_per_station"] = summary_df[
        "avg_chargers_per_station"
    ].round(2)
    summary_df["median_chargers_per_station"] = summary_df[
        "median_chargers_per_station"
    ].round(2)

    return summary_df


def save_outputs(
    station_df: pd.DataFrame,
    conflict_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> list[Path]:
    """CSV, GeoJSON, 검토 파일, 요약표를 저장한다."""
    station_df.to_csv(CSV_OUTPUT_FILE, index=False, encoding="utf-8-sig")
    conflict_df.to_csv(
        COORDINATE_CONFLICT_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(SUMMARY_OUTPUT_FILE, index=False, encoding="utf-8-sig")

    geometry = [
        Point(longitude, latitude)
        for longitude, latitude in zip(station_df["longitude"], station_df["latitude"])
    ]
    station_gdf = gpd.GeoDataFrame(station_df.copy(), geometry=geometry, crs="EPSG:4326")
    station_gdf.to_file(GEOJSON_OUTPUT_FILE, driver="GeoJSON", encoding="utf-8")

    return [
        CSV_OUTPUT_FILE,
        GEOJSON_OUTPUT_FILE,
        COORDINATE_CONFLICT_OUTPUT_FILE,
        SUMMARY_OUTPUT_FILE,
    ]


def print_terminal_summary(
    input_charger_count: int,
    station_df: pd.DataFrame,
    conflict_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_paths: list[Path],
) -> None:
    """요구된 터미널 요약을 출력한다."""
    charger_sum = int(station_df["charger_count"].sum())
    conflict_charger_sum = (
        int(conflict_df["charger_count"].sum()) if not conflict_df.empty else 0
    )

    print("\n[충전소 단위 공간 데이터 생성 결과]")
    print(f"입력 충전기 수: {input_charger_count:,}")
    print(f"최종 충전소 수: {len(station_df):,}")
    print(f"충전소당 평균 충전기 수: {station_df['charger_count'].mean():.2f}")
    print(f"충전소당 중앙값 충전기 수: {station_df['charger_count'].median():.2f}")
    print(f"충전소당 최대 충전기 수: {int(station_df['charger_count'].max()):,}")
    print(f"좌표 충돌 충전소 수: {len(conflict_df):,}")

    print("\n[군·구별 충전소 수]")
    print(summary_df[["district_2024", "station_count"]].to_string(index=False))

    print("\n[충전기 수 합계 검증]")
    print(f"최종 충전소 데이터 충전기 수 합계: {charger_sum:,}")
    print(f"좌표 충돌 검토 충전기 수 합계: {conflict_charger_sum:,}")
    print(f"검증 합계: {charger_sum + conflict_charger_sum:,} / {input_charger_count:,}")

    print("\n[생성 파일]")
    for path in output_paths:
        print(path)


def main() -> None:
    ensure_output_dirs()

    raw_df, _ = choose_input_dataframe()
    charger_df = prepare_charger_level_data(raw_df)
    input_charger_count = len(charger_df)

    station_df, conflict_df = build_station_level_data(charger_df)
    validate_station_level_output(station_df, conflict_df, input_charger_count)
    summary_df = build_station_summary(station_df)
    output_paths = save_outputs(station_df, conflict_df, summary_df)
    print_terminal_summary(
        input_charger_count,
        station_df,
        conflict_df,
        summary_df,
        output_paths,
    )


if __name__ == "__main__":
    main()
