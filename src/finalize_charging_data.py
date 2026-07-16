from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. 경로 설정
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "charging_stations_incheon_clean_step1.csv"
)

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FINAL_FILE = (
    PROCESSED_DIR
    / "charging_stations_incheon_final.csv"
)

SPATIAL_FILE = (
    PROCESSED_DIR
    / "charging_stations_incheon_spatial.csv"
)

CORRECTION_LOG_FILE = (
    PROCESSED_DIR
    / "charging_stations_correction_log.csv"
)

UNRESOLVED_COORDINATE_FILE = (
    PROCESSED_DIR
    / "charging_stations_coordinate_unresolved.csv"
)


# --------------------------------------------------
# 2. 입력 파일 확인 및 불러오기
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"1차 전처리 파일을 찾을 수 없습니다:\n{INPUT_FILE}"
    )

charging_df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig",
    low_memory=False,
)

original_row_count = len(charging_df)

print(f"입력 데이터 행 수: {original_row_count:,}")


# --------------------------------------------------
# 3. Boolean 열 복원
# --------------------------------------------------

boolean_columns = [
    "coordinate_missing",
    "coordinate_outlier",
    "district_mismatch",
    "is_deleted",
    "is_limited",
    "is_active_record",
    "usable_for_spatial_analysis",
]

boolean_map = {
    "true": True,
    "false": False,
    "1": True,
    "0": False,
    "y": True,
    "n": False,
}

for column in boolean_columns:
    if column not in charging_df.columns:
        continue

    if charging_df[column].dtype != "bool":
        converted = (
            charging_df[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .map(boolean_map)
        )

        invalid_mask = converted.isna() & charging_df[column].notna()

        if invalid_mask.any():
            invalid_values = (
                charging_df.loc[invalid_mask, column]
                .drop_duplicates()
                .tolist()
            )

            raise ValueError(
                f"{column} 열에서 Boolean으로 변환할 수 없는 값: "
                f"{invalid_values}"
            )

        charging_df[column] = converted.fillna(False).astype(bool)


# --------------------------------------------------
# 4. 수정 전 값을 기록할 열 생성
# --------------------------------------------------

charging_df["district_before_correction"] = (
    charging_df["district_by_code"]
)

charging_df["lat_before_correction"] = charging_df["lat"]
charging_df["lng_before_correction"] = charging_df["lng"]

charging_df["district_corrected"] = False
charging_df["coordinate_corrected"] = False
charging_df["correction_reason"] = pd.NA


# --------------------------------------------------
# 5. 코드와 주소가 충돌하는 남동구 51행 수정
# --------------------------------------------------
# zscode가 제물포구 코드로 표시됐지만
# 주소와 좌표가 명확히 남동구에 해당하는 사례이다.

namdong_mismatch_mask = (
    charging_df["district_by_code"].eq("제물포구")
    & charging_df["district_address_normalized"].eq("남동구")
)

charging_df.loc[
    namdong_mismatch_mask,
    "district_by_code"
] = "남동구"

charging_df.loc[
    namdong_mismatch_mask,
    "district_corrected"
] = True

charging_df.loc[
    namdong_mismatch_mask,
    "correction_reason"
] = "행정구역 코드와 주소 충돌: 주소 기준 남동구로 수정"


# --------------------------------------------------
# 6. 최종 군·구 열 생성
# --------------------------------------------------

charging_df["district_final"] = charging_df["district_by_code"]


# --------------------------------------------------
# 7. 동일 충전소의 정상 좌표를 이용한 이상치 보정
# --------------------------------------------------
# 같은 statId 안에 좌표 이상치가 아닌 행이 있으면,
# 그 정상 좌표의 중앙값을 해당 충전소 대표 좌표로 사용한다.

valid_coordinate_rows = charging_df.loc[
    ~charging_df["coordinate_outlier"]
    & ~charging_df["coordinate_missing"],
    ["statId", "lat", "lng"],
].copy()

station_valid_coordinates = (
    valid_coordinate_rows
    .groupby("statId", as_index=False)
    .agg(
        valid_station_lat=("lat", "median"),
        valid_station_lng=("lng", "median"),
    )
)

charging_df = charging_df.merge(
    station_valid_coordinates,
    on="statId",
    how="left",
    validate="many_to_one",
)

repairable_coordinate_mask = (
    charging_df["coordinate_outlier"]
    & charging_df["valid_station_lat"].notna()
    & charging_df["valid_station_lng"].notna()
)

charging_df.loc[
    repairable_coordinate_mask,
    "lat"
] = charging_df.loc[
    repairable_coordinate_mask,
    "valid_station_lat"
]

charging_df.loc[
    repairable_coordinate_mask,
    "lng"
] = charging_df.loc[
    repairable_coordinate_mask,
    "valid_station_lng"
]

charging_df.loc[
    repairable_coordinate_mask,
    "coordinate_corrected"
] = True

# 기존 수정 사유가 있으면 뒤에 좌표 수정 내용을 추가한다.
existing_reason_mask = (
    repairable_coordinate_mask
    & charging_df["correction_reason"].notna()
)

charging_df.loc[
    existing_reason_mask,
    "correction_reason"
] = (
    charging_df.loc[
        existing_reason_mask,
        "correction_reason",
    ]
    + "; 동일 충전소의 정상 좌표 중앙값으로 보정"
)

new_reason_mask = (
    repairable_coordinate_mask
    & charging_df["correction_reason"].isna()
)

charging_df.loc[
    new_reason_mask,
    "correction_reason"
] = "동일 충전소의 정상 좌표 중앙값으로 보정"


# --------------------------------------------------
# 8. 좌표 유효성 다시 판정
# --------------------------------------------------
# 강화군과 옹진군을 포함하도록 넉넉하게 설정한 1차 범위이다.

LAT_MIN = 37.0
LAT_MAX = 38.1
LNG_MIN = 124.0
LNG_MAX = 127.2

charging_df["coordinate_missing_final"] = (
    charging_df[["lat", "lng"]]
    .isna()
    .any(axis=1)
)

charging_df["coordinate_outlier_final"] = (
    ~charging_df["lat"].between(LAT_MIN, LAT_MAX)
    | ~charging_df["lng"].between(LNG_MIN, LNG_MAX)
)


# --------------------------------------------------
# 9. 최종 분석 가능 여부 생성
# --------------------------------------------------

charging_df["usable_for_district_analysis"] = (
    ~charging_df["is_deleted"]
    & charging_df["district_final"].notna()
)

charging_df["usable_for_spatial_analysis_final"] = (
    ~charging_df["is_deleted"]
    & ~charging_df["coordinate_missing_final"]
    & ~charging_df["coordinate_outlier_final"]
)

# 누구나 이용 가능한 충전기 분석용 기준
charging_df["publicly_accessible"] = (
    ~charging_df["is_deleted"]
    & ~charging_df["is_limited"]
)


# --------------------------------------------------
# 10. 수정·미해결 자료 분리
# --------------------------------------------------

correction_log_df = charging_df.loc[
    charging_df["district_corrected"]
    | charging_df["coordinate_corrected"],
    [
        "charger_uid",
        "statId",
        "statNm",
        "addr",
        "district_before_correction",
        "district_final",
        "lat_before_correction",
        "lng_before_correction",
        "lat",
        "lng",
        "district_corrected",
        "coordinate_corrected",
        "correction_reason",
    ],
].copy()

unresolved_coordinate_df = charging_df.loc[
    charging_df["coordinate_outlier_final"]
    | charging_df["coordinate_missing_final"],
    [
        "charger_uid",
        "statId",
        "statNm",
        "addr",
        "district_final",
        "lat",
        "lng",
        "is_deleted",
    ],
].copy()


# --------------------------------------------------
# 11. 불필요한 보조 열 제거
# --------------------------------------------------

charging_df = charging_df.drop(
    columns=[
        "valid_station_lat",
        "valid_station_lng",
    ]
)


# --------------------------------------------------
# 12. 공간 분석용 데이터 분리
# --------------------------------------------------

spatial_df = charging_df.loc[
    charging_df["usable_for_spatial_analysis_final"]
].copy()


# --------------------------------------------------
# 13. 결과 검증
# --------------------------------------------------

print("\n[전처리 2단계 결과]")
print(f"전체 행 수: {len(charging_df):,}")

print(
    "행정구역 수정 행:",
    f"{charging_df['district_corrected'].sum():,}"
)

print(
    "좌표 자동 보정 행:",
    f"{charging_df['coordinate_corrected'].sum():,}"
)

print(
    "좌표 미해결 행:",
    f"{len(unresolved_coordinate_df):,}"
)

print(
    "삭제 제외 군·구 분석 가능 행:",
    f"{charging_df['usable_for_district_analysis'].sum():,}"
)

print(
    "최종 공간 분석 가능 행:",
    f"{len(spatial_df):,}"
)

print(
    "일반 이용 가능 충전기:",
    f"{charging_df['publicly_accessible'].sum():,}"
)

if len(charging_df) != original_row_count:
    raise RuntimeError(
        "전처리 과정에서 전체 행 수가 변경되었습니다."
    )


# --------------------------------------------------
# 14. 파일 저장
# --------------------------------------------------

charging_df.to_csv(
    FINAL_FILE,
    index=False,
    encoding="utf-8-sig",
)

spatial_df.to_csv(
    SPATIAL_FILE,
    index=False,
    encoding="utf-8-sig",
)

correction_log_df.to_csv(
    CORRECTION_LOG_FILE,
    index=False,
    encoding="utf-8-sig",
)

unresolved_coordinate_df.to_csv(
    UNRESOLVED_COORDINATE_FILE,
    index=False,
    encoding="utf-8-sig",
)

print("\n[파일 저장 완료]")
print(FINAL_FILE)
print(SPATIAL_FILE)
print(CORRECTION_LOG_FILE)
print(UNRESOLVED_COORDINATE_FILE)
