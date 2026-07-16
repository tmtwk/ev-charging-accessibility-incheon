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
    / "charging_stations_incheon_final.csv"
)

OUTPUT_TABLE_DIR = PROJECT_DIR / "outputs" / "tables"
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

DISTRICT_SUMMARY_FILE = (
    OUTPUT_TABLE_DIR
    / "district_charging_summary.csv"
)

OUTPUT_SUMMARY_FILE = (
    OUTPUT_TABLE_DIR
    / "charger_output_summary.csv"
)

LEGACY_DISTRICT_FILE = (
    OUTPUT_TABLE_DIR
    / "legacy_district_chargers.csv"
)

LEGACY_DISTRICT_NAMES = [
    "중구_과거코드",
    "서구_과거코드",
]


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"분석 파일을 찾을 수 없습니다:\n{INPUT_FILE}"
    )

charging_df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig",
    low_memory=False,
)

print(f"불러온 행 수: {len(charging_df):,}")
print(f"불러온 열 수: {charging_df.shape[1]:,}")


# --------------------------------------------------
# 3. Boolean 열 복원
# --------------------------------------------------

boolean_columns = [
    "is_deleted",
    "is_limited",
    "publicly_accessible",
    "usable_for_district_analysis",
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

        invalid_mask = (
            charging_df[column].notna()
            & converted.isna()
        )

        if invalid_mask.any():
            invalid_values = (
                charging_df.loc[invalid_mask, column]
                .drop_duplicates()
                .tolist()
            )

            raise ValueError(
                f"{column} 열에 변환할 수 없는 값이 있습니다: "
                f"{invalid_values}"
            )

        charging_df[column] = (
            converted
            .fillna(False)
            .astype(bool)
        )


# --------------------------------------------------
# 4. 분석용 데이터 선택
# --------------------------------------------------

active_df = charging_df.loc[
    charging_df["usable_for_district_analysis"]
].copy()

district_df = active_df.loc[
    ~active_df["district_final"].isin(LEGACY_DISTRICT_NAMES)
].copy()

legacy_district_df = active_df.loc[
    active_df["district_final"].isin(LEGACY_DISTRICT_NAMES)
].copy()

print(
    "전체 운영 인프라 행 수:",
    f"{len(active_df):,}"
)

print(
    "군·구 비교 분석 대상 행 수:",
    f"{len(district_df):,}"
)

print(
    "과거 코드로 분리한 행 수:",
    f"{len(legacy_district_df):,}"
)


# --------------------------------------------------
# 5. 충전기 고유성 재검증
# --------------------------------------------------

duplicate_charger_count = (
    district_df["charger_uid"]
    .duplicated()
    .sum()
)

if duplicate_charger_count > 0:
    raise ValueError(
        f"charger_uid 중복이 "
        f"{duplicate_charger_count:,}건 발견되었습니다."
    )


# --------------------------------------------------
# 6. 군·구별 충전소 수
# --------------------------------------------------
# statId의 고유 개수로 계산한다.

station_count = (
    district_df
    .groupby("district_final")["statId"]
    .nunique()
    .rename("station_count")
)


# --------------------------------------------------
# 7. 군·구별 충전기 수
# --------------------------------------------------
# charger_uid의 고유 개수로 계산한다.

charger_count = (
    district_df
    .groupby("district_final")["charger_uid"]
    .nunique()
    .rename("charger_count")
)


# --------------------------------------------------
# 8. 일반 이용 가능 충전기 수
# --------------------------------------------------

public_charger_count = (
    district_df.loc[
        district_df["publicly_accessible"]
    ]
    .groupby("district_final")["charger_uid"]
    .nunique()
    .rename("public_charger_count")
)


# --------------------------------------------------
# 9. 이용 제한 충전기 수
# --------------------------------------------------

limited_charger_count = (
    district_df.loc[
        district_df["is_limited"]
    ]
    .groupby("district_final")["charger_uid"]
    .nunique()
    .rename("limited_charger_count")
)


# --------------------------------------------------
# 10. 군·구별 요약표 결합
# --------------------------------------------------

district_summary = pd.concat(
    [
        station_count,
        charger_count,
        public_charger_count,
        limited_charger_count,
    ],
    axis=1,
).fillna(0)

count_columns = [
    "station_count",
    "charger_count",
    "public_charger_count",
    "limited_charger_count",
]

district_summary[count_columns] = (
    district_summary[count_columns]
    .astype(int)
)


# --------------------------------------------------
# 11. 파생 지표 계산
# --------------------------------------------------

district_summary[
    "avg_chargers_per_station"
] = (
    district_summary["charger_count"]
    / district_summary["station_count"]
).round(2)

district_summary[
    "public_charger_ratio_percent"
] = (
    district_summary["public_charger_count"]
    / district_summary["charger_count"]
    * 100
).round(2)

district_summary[
    "limited_charger_ratio_percent"
] = (
    district_summary["limited_charger_count"]
    / district_summary["charger_count"]
    * 100
).round(2)


# --------------------------------------------------
# 12. 정렬 및 인덱스 정리
# --------------------------------------------------

district_summary = (
    district_summary
    .sort_values(
        "charger_count",
        ascending=False,
    )
    .reset_index()
)


# --------------------------------------------------
# 13. 충전용량 분포
# --------------------------------------------------

active_df["output"] = pd.to_numeric(
    active_df["output"],
    errors="coerce",
)

output_summary = (
    active_df["output"]
    .value_counts(dropna=False)
    .rename_axis("output_kw")
    .reset_index(name="charger_count")
)

output_summary[
    "ratio_percent"
] = (
    output_summary["charger_count"]
    / len(active_df)
    * 100
).round(2)


# --------------------------------------------------
# 14. 전체 요약 출력
# --------------------------------------------------

total_station_count = active_df["statId"].nunique()
total_charger_count = active_df["charger_uid"].nunique()

total_public_charger_count = (
    active_df.loc[
        active_df["publicly_accessible"],
        "charger_uid",
    ]
    .nunique()
)

total_limited_charger_count = (
    active_df.loc[
        active_df["is_limited"],
        "charger_uid",
    ]
    .nunique()
)

print("\n" + "=" * 70)
print("전체 기초 통계")
print("=" * 70)

print(f"운영 중 충전소 수: {total_station_count:,}")
print(f"운영 중 충전기 수: {total_charger_count:,}")
print(
    "일반 이용 가능 충전기 수:",
    f"{total_public_charger_count:,}"
)
print(
    "이용 제한 충전기 수:",
    f"{total_limited_charger_count:,}"
)

print("\n" + "=" * 70)
print("군·구별 충전 인프라 요약")
print("=" * 70)

print(district_summary.to_string(index=False))

print("\n" + "=" * 70)
print("충전용량 분포 상위 20개")
print("=" * 70)

print(output_summary.head(20).to_string(index=False))


# --------------------------------------------------
# 15. 결과 저장
# --------------------------------------------------

district_summary.to_csv(
    DISTRICT_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig",
)

output_summary.to_csv(
    OUTPUT_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig",
)

legacy_district_df.to_csv(
    LEGACY_DISTRICT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print("\n[저장 완료]")
print(DISTRICT_SUMMARY_FILE)
print(OUTPUT_SUMMARY_FILE)
print(LEGACY_DISTRICT_FILE)
