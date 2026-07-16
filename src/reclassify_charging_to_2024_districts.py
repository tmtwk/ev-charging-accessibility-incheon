from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


# --------------------------------------------------
# 1. 경로 설정
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHARGING_INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "charging_stations_incheon_spatial.csv"
)

BOUNDARY_INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "boundary_2024_incheon"
    / "DA_SIG_202406.shp"
)

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_TABLE_DIR = PROJECT_DIR / "outputs" / "tables"

CSV_OUTPUT_FILE = PROCESSED_DIR / "charging_stations_incheon_2024_district.csv"
GEOJSON_OUTPUT_FILE = PROCESSED_DIR / "charging_stations_incheon_2024_district.geojson"
SUMMARY_OUTPUT_FILE = OUTPUT_TABLE_DIR / "charging_district_2024_summary.csv"
CROSSTAB_OUTPUT_FILE = OUTPUT_TABLE_DIR / "district_2026_to_2024_crosstab.csv"
UNMATCHED_OUTPUT_FILE = OUTPUT_TABLE_DIR / "charging_boundary_unmatched.csv"

EXPECTED_DISTRICTS_2024 = {
    "중구",
    "동구",
    "미추홀구",
    "연수구",
    "남동구",
    "부평구",
    "계양구",
    "서구",
    "강화군",
    "옹진군",
}

CHARGING_REQUIRED_COLUMNS = [
    "charger_uid",
    "statId",
    "statNm",
    "addr",
    "lat",
    "lng",
    "district_final",
]

BOUNDARY_REQUIRED_COLUMNS = [
    "BASE_DATE",
    "SIGUNGU_NM",
    "SIGUNGU_CD",
    "geometry",
]


def ensure_directories() -> None:
    """출력 디렉터리를 생성한다."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)


def validate_required_columns(df: pd.DataFrame, required_columns: list[str], name: str) -> None:
    """필수 열이 없으면 명시적으로 오류를 발생시킨다."""
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise KeyError(f"{name} 필수 열이 없습니다: {missing_columns}")


def load_charging_data() -> gpd.GeoDataFrame:
    """충전소 CSV를 읽고 WGS84 점 GeoDataFrame으로 변환한다."""
    if not CHARGING_INPUT_FILE.exists():
        raise FileNotFoundError(f"충전소 입력 파일을 찾을 수 없습니다: {CHARGING_INPUT_FILE}")

    charging_df = pd.read_csv(CHARGING_INPUT_FILE, encoding="utf-8-sig", low_memory=False)
    validate_required_columns(charging_df, CHARGING_REQUIRED_COLUMNS, "충전소 데이터")

    print(f"충전소 입력 행 수: {len(charging_df):,}")

    # 좌표 열은 원본 문자열이 섞여 있어도 분석 가능하도록 숫자형으로 변환한다.
    charging_df["lat"] = pd.to_numeric(charging_df["lat"], errors="coerce")
    charging_df["lng"] = pd.to_numeric(charging_df["lng"], errors="coerce")

    missing_coord_mask = charging_df["lat"].isna() | charging_df["lng"].isna()
    missing_coord_count = int(missing_coord_mask.sum())
    print(f"좌표 결측 행 수: {missing_coord_count:,}")

    if missing_coord_count > 0:
        print("좌표 결측 행:")
        print(
            charging_df.loc[
                missing_coord_mask,
                ["charger_uid", "statId", "statNm", "addr", "lat", "lng", "district_final"],
            ].to_string(index=False)
        )

    duplicate_charger_count = int(charging_df["charger_uid"].duplicated().sum())
    print(f"charger_uid 중복 행 수: {duplicate_charger_count:,}")

    if duplicate_charger_count > 0:
        duplicate_rows = charging_df.loc[
            charging_df["charger_uid"].duplicated(keep=False),
            ["charger_uid", "statId", "statNm", "addr"],
        ].sort_values("charger_uid")
        print("charger_uid 중복 행:")
        print(duplicate_rows.to_string(index=False))
        raise ValueError(f"charger_uid 중복이 {duplicate_charger_count:,}건 발견되었습니다.")

    # Point는 경도(lng)를 x, 위도(lat)를 y로 생성한다.
    geometry = [
        None if pd.isna(lng) or pd.isna(lat) else Point(lng, lat)
        for lng, lat in zip(charging_df["lng"], charging_df["lat"])
    ]

    return gpd.GeoDataFrame(charging_df, geometry=geometry, crs="EPSG:4326")


def load_and_validate_boundary() -> gpd.GeoDataFrame:
    """2024년 인천 군·구 경계 데이터를 읽고 검증한다."""
    if not BOUNDARY_INPUT_FILE.exists():
        raise FileNotFoundError(f"경계 입력 파일을 찾을 수 없습니다: {BOUNDARY_INPUT_FILE}")

    # DBF 한글 속성값은 기본 인코딩 추론이 깨질 수 있어 cp949를 명시한다.
    boundary_gdf = gpd.read_file(BOUNDARY_INPUT_FILE, encoding="cp949")
    validate_required_columns(boundary_gdf, BOUNDARY_REQUIRED_COLUMNS, "경계 데이터")

    print(f"경계 입력 행 수: {len(boundary_gdf):,}")
    print(f"경계 CRS: {boundary_gdf.crs}")

    if boundary_gdf.crs is None or boundary_gdf.crs.to_epsg() != 5179:
        raise ValueError(f"경계 CRS가 EPSG:5179가 아닙니다: {boundary_gdf.crs}")

    base_dates = sorted(boundary_gdf["BASE_DATE"].dropna().astype(str).unique().tolist())
    print(f"BASE_DATE 고유값: {base_dates}")

    district_values = set(boundary_gdf["SIGUNGU_NM"].dropna().astype(str).unique())
    print(f"군·구 고유값 수: {len(district_values):,}")
    print(f"군·구 고유값: {sorted(district_values)}")

    if len(boundary_gdf) != 10:
        raise ValueError(f"경계 행 수가 10개가 아닙니다: {len(boundary_gdf):,}")

    if district_values != EXPECTED_DISTRICTS_2024:
        missing = sorted(EXPECTED_DISTRICTS_2024 - district_values)
        unexpected = sorted(district_values - EXPECTED_DISTRICTS_2024)
        raise ValueError(f"군·구명이 기대값과 다릅니다. 누락={missing}, 예상 외={unexpected}")

    empty_geometry_mask = boundary_gdf.geometry.is_empty | boundary_gdf.geometry.isna()
    invalid_geometry_mask = ~boundary_gdf.geometry.is_valid

    empty_geometry_count = int(empty_geometry_mask.sum())
    invalid_geometry_count = int(invalid_geometry_mask.sum())

    print(f"빈 geometry 행 수: {empty_geometry_count:,}")
    print(f"유효하지 않은 geometry 행 수: {invalid_geometry_count:,}")

    if empty_geometry_count > 0:
        print("빈 geometry 행:")
        print(boundary_gdf.loc[empty_geometry_mask, ["BASE_DATE", "SIGUNGU_NM", "SIGUNGU_CD"]].to_string(index=False))
        raise ValueError("빈 geometry가 있어 공간 결합을 진행할 수 없습니다.")

    boundary_gdf = boundary_gdf.copy()
    boundary_gdf["geometry_fixed"] = False

    if invalid_geometry_count > 0:
        print("유효하지 않은 geometry 행:")
        print(
            boundary_gdf.loc[
                invalid_geometry_mask,
                ["BASE_DATE", "SIGUNGU_NM", "SIGUNGU_CD"],
            ].to_string(index=False)
        )

        # 유효하지 않은 경계만 수정하고, 수정 여부를 별도 열에 남긴다.
        boundary_gdf.loc[invalid_geometry_mask, "geometry"] = (
            boundary_gdf.loc[invalid_geometry_mask, "geometry"].buffer(0)
        )
        boundary_gdf.loc[invalid_geometry_mask, "geometry_fixed"] = True

        remaining_invalid_count = int((~boundary_gdf.geometry.is_valid).sum())
        print(f"geometry 수정 후 유효하지 않은 행 수: {remaining_invalid_count:,}")

        if remaining_invalid_count > 0:
            raise ValueError("geometry 수정 후에도 유효하지 않은 경계가 남아 있습니다.")

    return boundary_gdf


def reproject_charging_to_boundary_crs(
    charging_gdf: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """충전소 점을 경계 좌표계로 변환한다."""
    charging_projected_gdf = charging_gdf.to_crs(boundary_gdf.crs)

    print(f"충전소 변환 후 CRS: {charging_projected_gdf.crs}")
    print(f"경계 CRS: {boundary_gdf.crs}")

    if charging_projected_gdf.crs != boundary_gdf.crs:
        raise ValueError(
            f"공간 결합 전 CRS가 일치하지 않습니다: "
            f"charging={charging_projected_gdf.crs}, boundary={boundary_gdf.crs}"
        )

    return charging_projected_gdf


def spatial_join_districts(
    charging_gdf: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """within 공간 결합으로 2024년 군·구를 부여한다."""
    boundary_columns = [
        "BASE_DATE",
        "SIGUNGU_NM",
        "SIGUNGU_CD",
        "geometry_fixed",
        "geometry",
    ]

    joined_gdf = gpd.sjoin(
        charging_gdf,
        boundary_gdf[boundary_columns],
        how="left",
        predicate="within",
    )

    match_count_by_charger = joined_gdf.groupby("charger_uid", dropna=False).size()
    multiple_match_uids = match_count_by_charger.loc[match_count_by_charger > 1].index.tolist()
    multiple_match_count = len(multiple_match_uids)

    joined_gdf["boundary_match_status"] = "unmatched"
    joined_gdf.loc[joined_gdf["SIGUNGU_NM"].notna(), "boundary_match_status"] = "matched"

    if multiple_match_count > 0:
        joined_gdf.loc[
            joined_gdf["charger_uid"].isin(multiple_match_uids),
            "boundary_match_status",
        ] = "multiple_match"
        print(f"중복 매칭 charger_uid 수: {multiple_match_count:,}")
        print(joined_gdf.loc[joined_gdf["charger_uid"].isin(multiple_match_uids)].to_string())
        raise ValueError("하나의 charger_uid가 둘 이상의 경계와 결합되었습니다.")

    if len(joined_gdf) > len(charging_gdf):
        raise ValueError(
            f"공간 결합 결과 행 수가 입력 행 수보다 많습니다: "
            f"{len(joined_gdf):,} > {len(charging_gdf):,}"
        )

    joined_gdf = joined_gdf.rename(
        columns={
            "SIGUNGU_NM": "district_2024",
            "SIGUNGU_CD": "district_code_2024",
            "BASE_DATE": "boundary_base_date",
        }
    )

    return joined_gdf


def inspect_unmatched_intersections(
    joined_gdf: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """미매칭 점이 경계선 위에 있는지 intersects 결과를 별도로 확인한다."""
    unmatched_gdf = joined_gdf.loc[
        joined_gdf["boundary_match_status"] == "unmatched"
    ].copy()

    if unmatched_gdf.empty:
        print("미매칭 충전기 intersects 결합 행 수: 0")
        print("미매칭 충전기 intersects 고유 charger_uid 수: 0")
        return unmatched_gdf

    boundary_columns = [
        "BASE_DATE",
        "SIGUNGU_NM",
        "SIGUNGU_CD",
        "geometry",
    ]
    intersected_gdf = gpd.sjoin(
        unmatched_gdf.drop(columns=["index_right"], errors="ignore"),
        boundary_gdf[boundary_columns],
        how="inner",
        predicate="intersects",
    )

    print(f"미매칭 충전기 intersects 결합 행 수: {len(intersected_gdf):,}")
    print(
        "미매칭 충전기 intersects 고유 charger_uid 수: "
        f"{intersected_gdf['charger_uid'].nunique():,}"
    )

    return unmatched_gdf


def build_summary(joined_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """2024년 군·구별 충전소 수와 충전기 수를 집계한다."""
    matched_gdf = joined_gdf.loc[joined_gdf["boundary_match_status"] == "matched"].copy()

    station_count = (
        matched_gdf.groupby("district_2024")["statId"]
        .nunique()
        .rename("station_count")
    )
    charger_count = (
        matched_gdf.groupby("district_2024")["charger_uid"]
        .nunique()
        .rename("charger_count")
    )

    summary_df = (
        pd.concat([station_count, charger_count], axis=1)
        .reindex(sorted(EXPECTED_DISTRICTS_2024))
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"district_2024": "district_2024"})
    )

    return summary_df


def print_result_validation(
    charging_gdf: gpd.GeoDataFrame,
    joined_gdf: gpd.GeoDataFrame,
    summary_df: pd.DataFrame,
    crosstab_df: pd.DataFrame,
) -> None:
    """공간 결합 결과와 2026년-2024년 행정구역 대응을 출력한다."""
    input_count = len(charging_gdf)
    joined_count = len(joined_gdf)
    matched_count = int((joined_gdf["boundary_match_status"] == "matched").sum())
    unmatched_count = int((joined_gdf["boundary_match_status"] == "unmatched").sum())
    multiple_match_count = int((joined_gdf["boundary_match_status"] == "multiple_match").sum())
    match_rate = matched_count / input_count * 100 if input_count else 0

    print("\n[결과 검증]")
    print(f"입력 충전기 수: {input_count:,}")
    print(f"공간 결합 결과 행 수: {joined_count:,}")
    print(f"매칭 성공 수: {matched_count:,}")
    print(f"미매칭 수: {unmatched_count:,}")
    print(f"중복 매칭 수: {multiple_match_count:,}")
    print(f"매칭 성공률: {match_rate:.2f}%")

    if joined_count != input_count:
        raise ValueError(f"입력 행 수와 결과 행 수가 다릅니다: {input_count:,} != {joined_count:,}")

    if joined_gdf["charger_uid"].duplicated().any():
        raise ValueError("결과 데이터에서 charger_uid 고유성이 깨졌습니다.")

    print("\n2024년 군·구별 충전소 수:")
    print(summary_df[["district_2024", "station_count"]].to_string(index=False))

    print("\n2024년 군·구별 충전기 수:")
    print(summary_df[["district_2024", "charger_count"]].to_string(index=False))

    print("\n2026년 district_final과 2024년 district_2024 교차표:")
    print(crosstab_df.to_string())

    print("\n[행정구역 대응 확인 - 교차표 기반]")
    for source_district in ["영종구", "서해구", "검단구", "제물포구"]:
        if source_district in crosstab_df.index:
            observed = crosstab_df.loc[source_district]
            observed = observed[observed > 0].sort_values(ascending=False)
            print(f"{source_district}: {observed.to_dict()}")
        else:
            print(f"{source_district}: 입력 데이터에 없음")


def save_outputs(
    joined_gdf: gpd.GeoDataFrame,
    summary_df: pd.DataFrame,
    crosstab_df: pd.DataFrame,
    unmatched_gdf: gpd.GeoDataFrame,
) -> None:
    """분석 결과 파일을 저장한다."""
    output_columns_to_drop = ["geometry", "index_right"]
    csv_df = pd.DataFrame(joined_gdf.drop(columns=output_columns_to_drop, errors="ignore"))
    csv_df.to_csv(CSV_OUTPUT_FILE, index=False, encoding="utf-8-sig")

    joined_gdf.drop(columns=["index_right"], errors="ignore").to_crs("EPSG:4326").to_file(
        GEOJSON_OUTPUT_FILE,
        driver="GeoJSON",
    )

    summary_df.to_csv(SUMMARY_OUTPUT_FILE, index=False, encoding="utf-8-sig")
    crosstab_df.to_csv(CROSSTAB_OUTPUT_FILE, encoding="utf-8-sig")

    unmatched_csv_df = pd.DataFrame(unmatched_gdf.drop(columns=output_columns_to_drop, errors="ignore"))
    unmatched_csv_df.to_csv(UNMATCHED_OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n[생성 파일]")
    for output_file in [
        CSV_OUTPUT_FILE,
        GEOJSON_OUTPUT_FILE,
        SUMMARY_OUTPUT_FILE,
        CROSSTAB_OUTPUT_FILE,
        UNMATCHED_OUTPUT_FILE,
    ]:
        print(output_file)


def main() -> None:
    ensure_directories()

    charging_gdf = load_charging_data()
    boundary_gdf = load_and_validate_boundary()
    charging_projected_gdf = reproject_charging_to_boundary_crs(charging_gdf, boundary_gdf)

    joined_gdf = spatial_join_districts(charging_projected_gdf, boundary_gdf)
    unmatched_gdf = inspect_unmatched_intersections(joined_gdf, boundary_gdf)

    summary_df = build_summary(joined_gdf)
    crosstab_df = pd.crosstab(
        joined_gdf["district_final"],
        joined_gdf["district_2024"].fillna("미매칭"),
        dropna=False,
    )

    print_result_validation(charging_gdf, joined_gdf, summary_df, crosstab_df)
    save_outputs(joined_gdf, summary_df, crosstab_df, unmatched_gdf)


if __name__ == "__main__":
    main()
