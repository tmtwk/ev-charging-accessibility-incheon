from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. 파일 경로 설정
# --------------------------------------------------


# parent는 src, parent.parent는 프로젝트 최상위 폴더
PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "charging_stations_incheon_raw.csv"
)

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_FILE = (
    PROCESSED_DIR
    / "charging_stations_incheon_clean_step1.csv"
)

REVIEW_FILE = (
    PROCESSED_DIR
    / "charging_stations_incheon_review_required.csv"
)


# --------------------------------------------------
# 2. 원본 데이터 불러오기
# --------------------------------------------------

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"원본 파일을 찾을 수 없습니다: {RAW_FILE}"
    )

charging_df = pd.read_csv(
    RAW_FILE,
    encoding="utf-8-sig",
    low_memory=False,
)

original_row_count = len(charging_df)

print(f"전처리 전 행 수: {original_row_count:,}")
print(f"전처리 전 열 수: {charging_df.shape[1]:,}")


# --------------------------------------------------
# 3. 문자열 열의 앞뒤 공백 정리
# --------------------------------------------------

text_columns = charging_df.select_dtypes(
    include=["object", "string"]
).columns

for column in text_columns:
    charging_df[column] = (
        charging_df[column]
        .astype("string")
        .str.strip()
    )


# --------------------------------------------------
# 4. 충전기 고유 식별자 생성
# --------------------------------------------------

charging_df["charger_uid"] = (
    charging_df["statId"].astype("string")
    + "_"
    + charging_df["chgerId"]
    .astype("Int64")
    .astype("string")
    .str.zfill(2)
)

duplicate_charger_count = (
    charging_df["charger_uid"]
    .duplicated()
    .sum()
)

print(
    f"고유 충전기 ID 중복 수: "
    f"{duplicate_charger_count:,}"
)


# --------------------------------------------------
# 5. 좌표를 숫자형으로 변환
# --------------------------------------------------

charging_df["lat"] = pd.to_numeric(
    charging_df["lat"],
    errors="coerce",
)

charging_df["lng"] = pd.to_numeric(
    charging_df["lng"],
    errors="coerce",
)

charging_df["coordinate_missing"] = (
    charging_df[["lat", "lng"]]
    .isna()
    .any(axis=1)
)


# --------------------------------------------------
# 6. 인천 범위를 넉넉하게 설정해 이상치 표시
# --------------------------------------------------
# 강화군·옹진군 등 도서지역까지 고려한 1차 점검 범위이다.
# 이 조건은 최종 삭제 기준이 아니라 검토 대상 표시용이다.

LAT_MIN = 37.0
LAT_MAX = 38.1
LNG_MIN = 124.0
LNG_MAX = 127.2

charging_df["coordinate_outlier"] = (
    ~charging_df["lat"].between(LAT_MIN, LAT_MAX)
    | ~charging_df["lng"].between(LNG_MIN, LNG_MAX)
)


# --------------------------------------------------
# 7. 행정구역 코드 기준 군·구명 표준화
# --------------------------------------------------

district_code_map = {
    28110: "중구_과거코드",
    28125: "제물포구",
    28155: "영종구",
    28177: "미추홀구",
    28185: "연수구",
    28200: "남동구",
    28237: "부평구",
    28245: "계양구",
    28260: "서구_과거코드",
    28275: "서해구",
    28290: "검단구",
    28710: "강화군",
    28720: "옹진군",
}

charging_df["district_by_code"] = (
    charging_df["zscode"]
    .map(district_code_map)
)

charging_df["district_from_address"] = (
    charging_df["addr"]
    .str.split()
    .str[1]
)


# --------------------------------------------------
# 8. 코드와 주소의 군·구명이 충돌하는지 표시
# --------------------------------------------------

legacy_name_map = {
    "남구": "미추홀구",
    "서구": "서구_과거코드",
    "중구": "중구_과거코드",
}

charging_df["district_address_normalized"] = (
    charging_df["district_from_address"]
    .replace(legacy_name_map)
)

charging_df["district_mismatch"] = (
    charging_df["district_by_code"]
    != charging_df["district_address_normalized"]
)


# --------------------------------------------------
# 9. 삭제·이용 제한 여부 파생 변수
# --------------------------------------------------

charging_df["is_deleted"] = (
    charging_df["delYn"].eq("Y")
)

charging_df["is_limited"] = (
    charging_df["limitYn"].eq("Y")
)

charging_df["is_active_record"] = (
    ~charging_df["is_deleted"]
)


# --------------------------------------------------
# 10. 날짜 형식 변환
# --------------------------------------------------

datetime_columns = [
    "statUpdDt",
    "lastTsdt",
    "lastTedt",
    "nowTsdt",
]

for column in datetime_columns:
    charging_df[column] = (
        charging_df[column]
        .astype("Int64")
        .astype("string")
    )

    charging_df[column] = pd.to_datetime(
        charging_df[column],
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )


# --------------------------------------------------
# 11. 충전용량 숫자형 변환
# --------------------------------------------------

charging_df["output"] = pd.to_numeric(
    charging_df["output"],
    errors="coerce",
)


# --------------------------------------------------
# 12. 공간 분석 가능 여부 표시
# --------------------------------------------------

charging_df["usable_for_spatial_analysis"] = (
    ~charging_df["coordinate_missing"]
    & ~charging_df["coordinate_outlier"]
    & ~charging_df["is_deleted"]
)


# --------------------------------------------------
# 13. 검토 필요 행 분리
# --------------------------------------------------

review_mask = (
    charging_df["coordinate_missing"]
    | charging_df["coordinate_outlier"]
    | charging_df["district_mismatch"]
    | charging_df["is_deleted"]
)

review_df = charging_df.loc[review_mask].copy()


# --------------------------------------------------
# 14. 결과 출력
# --------------------------------------------------

print("\n[전처리 결과]")

print(
    "좌표 결측 행:",
    f"{charging_df['coordinate_missing'].sum():,}"
)

print(
    "좌표 이상치 후보:",
    f"{charging_df['coordinate_outlier'].sum():,}"
)

print(
    "행정구역 불일치 후보:",
    f"{charging_df['district_mismatch'].sum():,}"
)

print(
    "삭제 표시 충전기:",
    f"{charging_df['is_deleted'].sum():,}"
)

print(
    "공간 분석 가능 행:",
    f"{charging_df['usable_for_spatial_analysis'].sum():,}"
)

print(
    "검토 필요 행:",
    f"{len(review_df):,}"
)


# --------------------------------------------------
# 15. 파일 저장
# --------------------------------------------------

charging_df.to_csv(
    CLEAN_FILE,
    index=False,
    encoding="utf-8-sig",
)

review_df.to_csv(
    REVIEW_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(f"\n전체 전처리 파일 저장: {CLEAN_FILE}")
print(f"검토 대상 파일 저장: {REVIEW_FILE}")