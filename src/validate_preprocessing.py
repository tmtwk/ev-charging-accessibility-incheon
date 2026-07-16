from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "charging_stations_incheon_raw.csv"
)
CLEAN_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "charging_stations_incheon_clean_step1.csv"
)
REVIEW_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "charging_stations_incheon_review_required.csv"
)
OUTPUT_TABLE_DIR = PROJECT_DIR / "outputs" / "tables"

REQUIRED_COLUMNS = [
    "charger_uid",
    "statId",
    "chgerId",
    "statNm",
    "addr",
    "lat",
    "lng",
    "zscode",
    "district_by_code",
    "district_from_address",
    "district_address_normalized",
    "district_mismatch",
    "coordinate_missing",
    "coordinate_outlier",
    "is_deleted",
    "is_limited",
    "usable_for_spatial_analysis",
]

BOOLEAN_COLUMNS = [
    "district_mismatch",
    "coordinate_missing",
    "coordinate_outlier",
    "is_deleted",
    "is_limited",
    "usable_for_spatial_analysis",
]


def require_files(paths: list[Path]) -> None:
    """검증에 필요한 입력 파일이 모두 있는지 확인한다."""
    missing_files = [path for path in paths if not path.exists()]

    if missing_files:
        missing_text = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "필요한 파일을 찾을 수 없습니다:\n"
            f"{missing_text}"
        )


def read_csv(path: Path) -> pd.DataFrame:
    """프로젝트 CSV 파일을 공통 옵션으로 읽는다."""
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )


def require_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
    file_path: Path,
) -> None:
    """필수 열 누락 여부를 확인한다."""
    missing_columns = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{file_path}에서 다음 열을 찾을 수 없습니다:\n"
            f"{missing_columns}"
        )


def normalize_boolean_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """CSV 재로딩 후 문자열로 바뀐 Boolean 값을 다시 보정한다."""
    normalized_df = dataframe.copy()

    boolean_map = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "y": True,
        "n": False,
    }

    for column in columns:
        if normalized_df[column].dtype == "bool":
            continue

        original_values = (
            normalized_df[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        converted_values = original_values.map(boolean_map)

        invalid_mask = (
            original_values.notna()
            & converted_values.isna()
        )

        if invalid_mask.any():
            invalid_values = (
                original_values.loc[invalid_mask]
                .drop_duplicates()
                .tolist()
            )

            raise ValueError(
                f"{column} 열에서 Boolean으로 변환할 수 없는 "
                f"값을 발견했습니다: {invalid_values}"
            )

        normalized_df[column] = (
            converted_values
            .fillna(False)
            .astype(bool)
        )

    return normalized_df


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def build_validation_summary(clean_df: pd.DataFrame) -> pd.DataFrame:
    """전처리 판정 결과를 집계한다."""
    usable_count = int(
        clean_df["usable_for_spatial_analysis"].sum()
    )
    excluded_count = int(
        (~clean_df["usable_for_spatial_analysis"]).sum()
    )

    summary = pd.DataFrame({
        "검증항목": [
            "좌표 결측",
            "좌표 이상치 후보",
            "행정구역 불일치 후보",
            "삭제 표시 충전기",
            "이용 제한 충전기",
            "공간 분석 가능 충전기",
            "공간 분석 제외 충전기",
        ],
        "행수": [
            int(clean_df["coordinate_missing"].sum()),
            int(clean_df["coordinate_outlier"].sum()),
            int(clean_df["district_mismatch"].sum()),
            int(clean_df["is_deleted"].sum()),
            int(clean_df["is_limited"].sum()),
            usable_count,
            excluded_count,
        ],
    })

    summary["비율_퍼센트"] = (
        summary["행수"]
        / len(clean_df)
        * 100
    ).round(2)

    return summary


def get_coordinate_outliers(clean_df: pd.DataFrame) -> pd.DataFrame:
    return clean_df.loc[
        clean_df["coordinate_outlier"],
        [
            "charger_uid",
            "statNm",
            "addr",
            "lat",
            "lng",
            "district_by_code",
            "zscode",
            "is_deleted",
        ],
    ].copy()


def get_district_mismatches(clean_df: pd.DataFrame) -> pd.DataFrame:
    return clean_df.loc[
        clean_df["district_mismatch"],
        [
            "charger_uid",
            "statNm",
            "addr",
            "zscode",
            "district_by_code",
            "district_from_address",
            "district_address_normalized",
            "lat",
            "lng",
        ],
    ].copy()


def get_deleted_rows(clean_df: pd.DataFrame) -> pd.DataFrame:
    return clean_df.loc[
        clean_df["is_deleted"],
        [
            "charger_uid",
            "statNm",
            "addr",
            "delYn",
            "delDetail",
            "stat",
            "statUpdDt",
        ],
    ].copy()


def print_row_count_validation(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    review_df: pd.DataFrame,
) -> None:
    print_section("1. 전처리 전후 행 수")

    print(f"원본 데이터 행 수: {len(raw_df):,}")
    print(f"전처리 데이터 행 수: {len(clean_df):,}")
    print(f"검토 대상 행 수: {len(review_df):,}")

    if len(raw_df) == len(clean_df):
        print("결과: 전처리 과정에서 원본 행이 삭제되지 않았습니다.")
    else:
        print("주의: 원본과 전처리 데이터의 행 수가 다릅니다.")


def print_charger_uid_validation(clean_df: pd.DataFrame) -> None:
    print_section("2. 충전기 고유 ID 검증")

    print(f"charger_uid 결측 수: {clean_df['charger_uid'].isna().sum():,}")
    print(
        "charger_uid 중복 수: "
        f"{clean_df['charger_uid'].duplicated().sum():,}"
    )


def print_validation_summary(summary: pd.DataFrame) -> None:
    print_section("3. 전처리 판정 결과")
    print(summary.to_string(index=False))


def print_coordinate_outliers(coordinate_outliers: pd.DataFrame) -> None:
    print_section("4. 좌표 이상치 후보")

    if coordinate_outliers.empty:
        print("좌표 이상치 후보가 없습니다.")
        return

    print(f"좌표 이상치 후보: {len(coordinate_outliers):,}행")
    print(coordinate_outliers.head(20).to_string(index=False))


def print_district_mismatches(
    district_mismatches: pd.DataFrame,
) -> None:
    print_section("5. 행정구역 불일치 후보")

    if district_mismatches.empty:
        print("행정구역 불일치 후보가 없습니다.")
        return

    print(f"행정구역 불일치 후보: {len(district_mismatches):,}행")

    mismatch_counts = (
        district_mismatches
        .groupby(
            [
                "district_by_code",
                "district_from_address",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="행수")
        .sort_values("행수", ascending=False)
    )

    print("\n[불일치 유형별 행 수]")
    print(mismatch_counts.head(30).to_string(index=False))

    print("\n[불일치 데이터 예시]")
    print(district_mismatches.head(20).to_string(index=False))


def print_deleted_rows(deleted_rows: pd.DataFrame) -> None:
    print_section("6. 삭제 표시 충전기")

    if deleted_rows.empty:
        print("삭제 표시 충전기가 없습니다.")
        return

    print(f"삭제 표시 충전기: {len(deleted_rows):,}행")
    print(deleted_rows.head(20).to_string(index=False))


def print_review_file_validation(
    clean_df: pd.DataFrame,
    review_df: pd.DataFrame,
) -> None:
    expected_review_mask = (
        clean_df["coordinate_missing"]
        | clean_df["coordinate_outlier"]
        | clean_df["district_mismatch"]
        | clean_df["is_deleted"]
    )
    expected_review_count = int(expected_review_mask.sum())

    print_section("7. 검토 대상 파일 검증")

    print(f"조건으로 계산한 검토 대상: {expected_review_count:,}행")
    print(f"실제 review 파일 행 수: {len(review_df):,}행")

    if expected_review_count == len(review_df):
        print("결과: 검토 대상 파일이 조건과 일치합니다.")
    else:
        print(
            "주의: 검토 조건으로 계산한 행 수와 "
            "review 파일의 행 수가 다릅니다."
        )


def save_outputs(
    validation_summary: pd.DataFrame,
    coordinate_outliers: pd.DataFrame,
    district_mismatches: pd.DataFrame,
    deleted_rows: pd.DataFrame,
) -> None:
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    output_files = {
        "preprocessing_validation_summary.csv": validation_summary,
        "coordinate_outlier_review.csv": coordinate_outliers,
        "district_mismatch_review.csv": district_mismatches,
        "deleted_charger_review.csv": deleted_rows,
    }

    for file_name, dataframe in output_files.items():
        dataframe.to_csv(
            OUTPUT_TABLE_DIR / file_name,
            index=False,
            encoding="utf-8-sig",
        )

    print_section("검증 결과 저장 완료")

    for file_name in output_files:
        print(OUTPUT_TABLE_DIR / file_name)


def main() -> None:
    require_files([RAW_FILE, CLEAN_FILE, REVIEW_FILE])

    raw_df = read_csv(RAW_FILE)
    clean_df = read_csv(CLEAN_FILE)
    review_df = read_csv(REVIEW_FILE)

    require_columns(clean_df, REQUIRED_COLUMNS, CLEAN_FILE)
    clean_df = normalize_boolean_columns(clean_df, BOOLEAN_COLUMNS)

    validation_summary = build_validation_summary(clean_df)
    coordinate_outliers = get_coordinate_outliers(clean_df)
    district_mismatches = get_district_mismatches(clean_df)
    deleted_rows = get_deleted_rows(clean_df)

    print_row_count_validation(raw_df, clean_df, review_df)
    print_charger_uid_validation(clean_df)
    print_validation_summary(validation_summary)
    print_coordinate_outliers(coordinate_outliers)
    print_district_mismatches(district_mismatches)
    print_deleted_rows(deleted_rows)
    print_review_file_validation(clean_df, review_df)

    save_outputs(
        validation_summary,
        coordinate_outliers,
        district_mismatches,
        deleted_rows,
    )


if __name__ == "__main__":
    main()
