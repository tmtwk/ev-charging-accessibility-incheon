from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "ev_registration_2026-02-19.csv"
)

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_TABLE_DIR = PROJECT_DIR / "outputs" / "tables"

CLEAN_FILE = (
    PROCESSED_DIR
    / "ev_registration_2026_02_19_clean.csv"
)

DISTRICT_SUMMARY_FILE = (
    OUTPUT_TABLE_DIR
    / "ev_registration_2026_district_summary.csv"
)

VEHICLE_TYPE_SUMMARY_FILE = (
    OUTPUT_TABLE_DIR
    / "ev_registration_2026_vehicle_type_summary.csv"
)

ENCODING_CANDIDATES = [
    "cp949",
    "utf-8-sig",
    "utf-8",
    "euc-kr",
]

REQUIRED_COLUMNS = [
    "시군구별",
    "연료별",
    "용도별",
    "승용",
    "승합",
    "화물",
    "특수",
]

NUMERIC_COLUMNS = [
    "승용",
    "승합",
    "화물",
    "특수",
    "계",
]

EXPECTED_DISTRICT_COUNT = 10
EXPECTED_TOTAL_EV_COUNT = 79860


def require_input_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"전기차 등록 원본 파일을 찾을 수 없습니다:\n{path}"
        )


def read_csv_with_fallback(
    path: Path,
) -> tuple[pd.DataFrame, str]:
    errors = []

    for encoding in ENCODING_CANDIDATES:
        try:
            dataframe = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
            )
            return dataframe, encoding
        except UnicodeDecodeError as error:
            errors.append(f"{encoding}: {error}")

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        "지원한 인코딩으로 파일을 읽지 못했습니다: "
        + " | ".join(errors),
    )


def standardize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = dataframe.copy()
    column_map = {
        column: "".join(str(column).strip().split())
        for column in cleaned_df.columns
    }

    cleaned_df = cleaned_df.rename(columns=column_map)

    return cleaned_df


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
) -> None:
    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "전기차 등록 데이터에서 필요한 열을 찾을 수 없습니다:\n"
            f"{missing_columns}"
        )


def add_total_column_if_missing(dataframe: pd.DataFrame) -> pd.DataFrame:
    """2026 원본처럼 계 열이 없으면 차종별 합계로 계 열을 만든다."""
    total_df = dataframe.copy()

    if "계" not in total_df.columns:
        total_df["계"] = (
            total_df["승용"]
            + total_df["승합"]
            + total_df["화물"]
            + total_df["특수"]
        )

    return total_df


def validate_fuel_type(dataframe: pd.DataFrame) -> None:
    fuel_values = (
        dataframe["연료별"]
        .astype("string")
        .str.strip()
        .dropna()
        .unique()
        .tolist()
    )

    invalid_values = [
        value
        for value in fuel_values
        if value != "전기"
    ]

    if invalid_values:
        raise ValueError(
            "연료별 열에 전기 이외의 값이 있습니다:\n"
            f"{invalid_values}"
        )


def normalize_text_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized_df = dataframe.copy()

    for column in ["시군구별", "연료별", "용도별"]:
        normalized_df[column] = (
            normalized_df[column]
            .astype("string")
            .str.strip()
        )

    return normalized_df


def convert_numeric_columns(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
) -> pd.DataFrame:
    converted_df = dataframe.copy()

    for column in numeric_columns:
        if column not in converted_df.columns:
            continue

        numeric_series = pd.to_numeric(
            converted_df[column],
            errors="coerce",
        )

        invalid_mask = numeric_series.isna() & converted_df[column].notna()

        if invalid_mask.any():
            invalid_values = (
                converted_df.loc[invalid_mask, column]
                .drop_duplicates()
                .tolist()
            )
            raise ValueError(
                f"{column} 열에서 숫자로 변환할 수 없는 값: "
                f"{invalid_values}"
            )

        converted_df[column] = numeric_series.astype(int)

    return converted_df


def validate_row_totals(dataframe: pd.DataFrame) -> None:
    vehicle_type_sum = (
        dataframe["승용"]
        + dataframe["승합"]
        + dataframe["화물"]
        + dataframe["특수"]
    )

    invalid_mask = vehicle_type_sum != dataframe["계"]

    if invalid_mask.any():
        invalid_rows = dataframe.loc[
            invalid_mask,
            [
                "시군구별",
                "용도별",
                "승용",
                "승합",
                "화물",
                "특수",
                "계",
            ],
        ].copy()
        invalid_rows["차종합계"] = vehicle_type_sum.loc[invalid_mask]

        raise ValueError(
            "승용+승합+화물+특수 합계가 계와 다른 행이 있습니다:\n"
            f"{invalid_rows.to_string(index=False)}"
        )


def build_district_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    vehicle_summary = (
        dataframe
        .groupby("시군구별", as_index=False)
        .agg(
            passenger_ev_count=("승용", "sum"),
            van_ev_count=("승합", "sum"),
            truck_ev_count=("화물", "sum"),
            special_ev_count=("특수", "sum"),
            total_ev_count=("계", "sum"),
        )
    )

    purpose_summary = (
        dataframe
        .pivot_table(
            index="시군구별",
            columns="용도별",
            values="계",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    for purpose in ["비사업용", "사업용"]:
        if purpose not in purpose_summary.columns:
            raise ValueError(
                f"용도별 열에서 {purpose} 값을 찾을 수 없습니다."
            )

    purpose_summary = purpose_summary.rename(
        columns={
            "비사업용": "private_ev_count",
            "사업용": "commercial_ev_count",
        }
    )

    district_summary = vehicle_summary.merge(
        purpose_summary[
            [
                "시군구별",
                "private_ev_count",
                "commercial_ev_count",
            ]
        ],
        on="시군구별",
        how="left",
        validate="one_to_one",
    )

    district_summary["commercial_ev_ratio_percent"] = (
        district_summary["commercial_ev_count"]
        / district_summary["total_ev_count"]
        * 100
    ).round(2)

    district_summary = district_summary.rename(
        columns={
            "시군구별": "district_2024",
        }
    )

    return district_summary.sort_values(
        "total_ev_count",
        ascending=False,
    ).reset_index(drop=True)


def build_vehicle_type_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    total_by_type = {
        "passenger_ev_count": int(dataframe["승용"].sum()),
        "van_ev_count": int(dataframe["승합"].sum()),
        "truck_ev_count": int(dataframe["화물"].sum()),
        "special_ev_count": int(dataframe["특수"].sum()),
    }

    total_count = int(dataframe["계"].sum())

    vehicle_type_summary = pd.DataFrame({
        "vehicle_type": list(total_by_type.keys()),
        "ev_count": list(total_by_type.values()),
    })

    vehicle_type_summary["ratio_percent"] = (
        vehicle_type_summary["ev_count"]
        / total_count
        * 100
    ).round(2)

    return vehicle_type_summary


def validate_district_summary(
    district_summary: pd.DataFrame,
) -> None:
    district_count = district_summary["district_2024"].nunique()

    if district_count != EXPECTED_DISTRICT_COUNT:
        districts = sorted(
            district_summary["district_2024"]
            .dropna()
            .tolist()
        )
        raise ValueError(
            f"군·구 수가 {EXPECTED_DISTRICT_COUNT}개가 아닙니다. "
            f"실제 군·구 수: {district_count}, 목록: {districts}"
        )

    total_ev_count = int(district_summary["total_ev_count"].sum())

    if total_ev_count != EXPECTED_TOTAL_EV_COUNT:
        print(
            "경고: 전기차 전체 등록 대수가 예상값과 다릅니다. "
            "원본 파일이 변경되었을 수 있습니다."
        )
        print(f"예상 합계: {EXPECTED_TOTAL_EV_COUNT:,}")
        print(f"실제 합계: {total_ev_count:,}")


def save_outputs(
    clean_df: pd.DataFrame,
    district_summary: pd.DataFrame,
    vehicle_type_summary: pd.DataFrame,
) -> list[Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    clean_df.to_csv(
        CLEAN_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    district_summary.to_csv(
        DISTRICT_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    vehicle_type_summary.to_csv(
        VEHICLE_TYPE_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return [
        CLEAN_FILE,
        DISTRICT_SUMMARY_FILE,
        VEHICLE_TYPE_SUMMARY_FILE,
    ]


def print_summary(
    encoding: str,
    raw_shape: tuple[int, int],
    district_summary: pd.DataFrame,
    output_paths: list[Path],
) -> None:
    total_ev_count = int(district_summary["total_ev_count"].sum())
    private_ev_count = int(district_summary["private_ev_count"].sum())
    commercial_ev_count = int(
        district_summary["commercial_ev_count"].sum()
    )

    print("\n" + "=" * 70)
    print("전기차 등록 데이터 전처리 결과")
    print("=" * 70)
    print(f"사용 인코딩: {encoding}")
    print(f"원본 행 수: {raw_shape[0]:,}")
    print(f"원본 열 수: {raw_shape[1]:,}")
    print(
        "군·구 수:",
        f"{district_summary['district_2024'].nunique():,}"
    )
    print(f"전기차 전체 등록 대수: {total_ev_count:,}")
    print(f"비사업용 합계: {private_ev_count:,}")
    print(f"사업용 합계: {commercial_ev_count:,}")

    print("\n[군·구별 요약표]")
    print(district_summary.to_string(index=False))

    print("\n[생성 파일]")
    for path in output_paths:
        print(path)


def main() -> None:
    require_input_file(RAW_FILE)

    raw_df, used_encoding = read_csv_with_fallback(RAW_FILE)
    raw_shape = raw_df.shape

    clean_df = standardize_columns(raw_df)
    require_columns(clean_df, REQUIRED_COLUMNS)
    clean_df = normalize_text_columns(clean_df)
    validate_fuel_type(clean_df)
    clean_df = convert_numeric_columns(clean_df, NUMERIC_COLUMNS)
    clean_df = add_total_column_if_missing(clean_df)
    validate_row_totals(clean_df)

    district_summary = build_district_summary(clean_df)
    vehicle_type_summary = build_vehicle_type_summary(clean_df)
    validate_district_summary(district_summary)

    output_paths = save_outputs(
        clean_df,
        district_summary,
        vehicle_type_summary,
    )

    print_summary(
        used_encoding,
        raw_shape,
        district_summary,
        output_paths,
    )


if __name__ == "__main__":
    main()
