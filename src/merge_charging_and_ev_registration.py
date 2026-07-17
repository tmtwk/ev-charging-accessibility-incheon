from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. 경로와 상수
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHARGING_SUMMARY_FILE = (
    PROJECT_DIR / "outputs" / "tables" / "charging_district_2024_summary.csv"
)
EV_REGISTRATION_FILE = (
    PROJECT_DIR / "outputs" / "tables" / "ev_registration_2026_district_summary.csv"
)
CHARGING_DETAIL_FILE = (
    PROJECT_DIR / "data" / "processed" / "charging_stations_incheon_2024_district.csv"
)

PROCESSED_OUTPUT_FILE = (
    PROJECT_DIR / "data" / "processed" / "charging_ev_demand_merged_2024_basis.csv"
)
SUMMARY_OUTPUT_FILE = (
    PROJECT_DIR / "outputs" / "tables" / "district_ev_charging_supply_demand_summary.csv"
)
SHORTAGE_RANKING_OUTPUT_FILE = (
    PROJECT_DIR / "outputs" / "tables" / "district_infrastructure_shortage_ranking.csv"
)

FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"
EV_AND_CHARGERS_FIGURE = FIGURE_DIR / "ev_and_chargers_by_district.png"
CHARGERS_PER_100_EV_FIGURE = FIGURE_DIR / "chargers_per_100_ev_by_district.png"
PUBLIC_CHARGERS_PER_100_EV_FIGURE = (
    FIGURE_DIR / "public_chargers_per_100_ev_by_district.png"
)
FAST_CHARGERS_PER_100_EV_FIGURE = (
    FIGURE_DIR / "fast_chargers_per_100_ev_by_district.png"
)

EXPECTED_DISTRICTS = {
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

EXPECTED_CHARGER_TOTAL = 32322
EXPECTED_EV_TOTAL = 79860
FAST_CHARGING_THRESHOLD_KW = 50

REGISTRATION_BASE_DATE = "2026-02-19"
REGISTRATION_BASE_LABEL = "2026년 2월 19일"
CHARGING_DATA_BASE_DATE = "2026-07"
CHARGING_DATA_BASE_LABEL = "2026년 7월"
BOUNDARY_BASE_DATE = "2024-06-30"

CHARGING_INFRA_COLUMNS = [
    "district_2024",
    "station_count",
    "charger_count",
    "public_charger_count",
    "limited_charger_count",
    "fast_charger_count",
    "slow_charger_count",
    "unknown_speed_count",
    "avg_chargers_per_station",
    "public_charger_ratio_percent",
    "fast_charger_ratio_percent",
]

EV_REGISTRATION_COLUMNS = [
    "total_ev_count",
    "passenger_ev_count",
    "van_ev_count",
    "truck_ev_count",
    "special_ev_count",
    "private_ev_count",
    "commercial_ev_count",
    "commercial_ev_ratio_percent",
]

OUTPUT_COLUMNS = [
    "district_2024",
    "station_count",
    "charger_count",
    "public_charger_count",
    "limited_charger_count",
    "fast_charger_count",
    "slow_charger_count",
    "unknown_speed_count",
    "avg_chargers_per_station",
    "public_charger_ratio_percent",
    "fast_charger_ratio_percent",
    "total_ev_count",
    "passenger_ev_count",
    "van_ev_count",
    "truck_ev_count",
    "special_ev_count",
    "private_ev_count",
    "commercial_ev_count",
    "commercial_ev_ratio_percent",
    "chargers_per_100_ev",
    "public_chargers_per_100_ev",
    "fast_chargers_per_100_ev",
    "ev_per_charger",
    "ev_per_public_charger",
    "stations_per_1000_ev",
    "weighted_ev_demand_equal",
    "weighted_ev_demand_commercial_1_5",
    "chargers_per_100_weighted_ev",
    "charger_count_rank",
    "ev_count_rank",
    "chargers_per_100_ev_rank",
    "public_chargers_per_100_ev_rank",
    "fast_chargers_per_100_ev_rank",
    "registration_base_date",
    "boundary_base_date",
    "charging_data_base_date",
]


def ensure_output_dirs() -> None:
    """결과 저장 디렉터리를 생성한다."""
    PROCESSED_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    """입력 파일 존재 여부를 확인한다."""
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")


def read_csv(path: Path) -> pd.DataFrame:
    """CSV 파일을 프로젝트 표준 옵션으로 읽는다."""
    require_file(path)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    print(f"\n[{path}]")
    print(f"행 수: {len(df):,}")
    print(f"열 이름: {df.columns.tolist()}")
    return df


def normalize_district_column(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """군·구 열을 district_2024로 맞추고 공백을 제거한다."""
    normalized_df = df.copy()

    if "district_2024" not in normalized_df.columns:
        candidate_columns = [
            column
            for column in normalized_df.columns
            if "district" in str(column).lower()
            or "군" in str(column)
            or "구" in str(column)
            or "시군구" in str(column)
        ]

        if len(candidate_columns) != 1:
            raise KeyError(
                f"{source_name}에서 군·구 열을 하나로 특정할 수 없습니다: "
                f"{candidate_columns}"
            )

        normalized_df = normalized_df.rename(columns={candidate_columns[0]: "district_2024"})
        print(
            f"{source_name} 군·구 열 이름을 "
            f"{candidate_columns[0]} -> district_2024로 변경했습니다."
        )

    normalized_df["district_2024"] = (
        normalized_df["district_2024"].astype("string").str.strip()
    )

    return normalized_df


def print_district_lists(charging_df: pd.DataFrame, registration_df: pd.DataFrame) -> None:
    """양쪽 군·구 목록을 출력한다."""
    charging_districts = sorted(charging_df["district_2024"].dropna().unique().tolist())
    registration_districts = sorted(registration_df["district_2024"].dropna().unique().tolist())

    print("\n[군·구 목록]")
    print(f"충전소 데이터: {charging_districts}")
    print(f"등록 데이터: {registration_districts}")


def validate_districts(charging_df: pd.DataFrame, registration_df: pd.DataFrame) -> None:
    """병합 전 군·구 수, 중복, 목록 일치 여부를 검증한다."""
    charging_district_set = set(charging_df["district_2024"].dropna().tolist())
    registration_district_set = set(registration_df["district_2024"].dropna().tolist())

    charging_only = sorted(charging_district_set - registration_district_set)
    registration_only = sorted(registration_district_set - charging_district_set)

    print(f"충전소 데이터에만 있는 군·구: {charging_only}")
    print(f"등록 데이터에만 있는 군·구: {registration_only}")

    if charging_only or registration_only:
        raise ValueError("양쪽 군·구 목록이 일치하지 않아 병합하지 않습니다.")

    for name, df in [("충전소 데이터", charging_df), ("등록 데이터", registration_df)]:
        districts = set(df["district_2024"].dropna().tolist())
        duplicate_count = int(df["district_2024"].duplicated().sum())

        if len(districts) != 10:
            raise ValueError(f"{name} 군·구 수가 10개가 아닙니다: {len(districts)}")

        if duplicate_count > 0:
            duplicates = (
                df.loc[df["district_2024"].duplicated(keep=False), "district_2024"]
                .dropna()
                .tolist()
            )
            raise ValueError(f"{name} 군·구 중복이 있습니다: {duplicates}")

        if districts != EXPECTED_DISTRICTS:
            missing = sorted(EXPECTED_DISTRICTS - districts)
            unexpected = sorted(districts - EXPECTED_DISTRICTS)
            raise ValueError(
                f"{name} 군·구 목록이 기대값과 다릅니다. "
                f"누락={missing}, 예상 외={unexpected}"
            )


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
        if column not in normalized_df.columns:
            raise KeyError(f"상세 충전소 데이터에 필요한 열이 없습니다: {column}")

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
            raise ValueError(f"{column} 열에서 Boolean 변환 불가 값: {invalid_values}")

        normalized_df[column] = converted.fillna(False).astype(bool)

    return normalized_df


def add_speed_class(df: pd.DataFrame) -> pd.DataFrame:
    """충전용량 기준 급속·완속 임시 분류를 추가한다."""
    classified_df = df.copy()
    classified_df["output"] = pd.to_numeric(classified_df["output"], errors="coerce")
    classified_df["charge_speed_class"] = "unknown"

    # 공식 충전기 유형 코드가 아니라 충전용량(output) 기준의 임시 분석 분류이다.
    classified_df.loc[
        classified_df["output"] >= FAST_CHARGING_THRESHOLD_KW,
        "charge_speed_class",
    ] = "fast"
    classified_df.loc[
        classified_df["output"].notna()
        & (classified_df["output"] < FAST_CHARGING_THRESHOLD_KW),
        "charge_speed_class",
    ] = "slow"

    return classified_df


def aggregate_charging_detail() -> pd.DataFrame:
    """공간 결합 상세 파일에서 부족한 충전 인프라 지표를 집계한다."""
    detail_df = read_csv(CHARGING_DETAIL_FILE)

    required_columns = [
        "district_2024",
        "boundary_match_status",
        "charger_uid",
        "statId",
        "publicly_accessible",
        "is_limited",
        "output",
    ]
    missing_columns = [column for column in required_columns if column not in detail_df.columns]
    if missing_columns:
        raise KeyError(f"상세 충전소 데이터 필수 열이 없습니다: {missing_columns}")

    detail_df["district_2024"] = detail_df["district_2024"].astype("string").str.strip()
    detail_df = normalize_boolean_columns(detail_df, ["publicly_accessible", "is_limited"])
    detail_df = add_speed_class(detail_df)

    matched_df = detail_df.loc[detail_df["boundary_match_status"].eq("matched")].copy()

    if matched_df["charger_uid"].duplicated().any():
        raise ValueError("상세 충전소 matched 데이터에서 charger_uid 중복이 있습니다.")

    station_count = matched_df.groupby("district_2024")["statId"].nunique()
    charger_count = matched_df.groupby("district_2024")["charger_uid"].nunique()
    public_charger_count = (
        matched_df.loc[matched_df["publicly_accessible"]]
        .groupby("district_2024")["charger_uid"]
        .nunique()
    )
    limited_charger_count = (
        matched_df.loc[matched_df["is_limited"]]
        .groupby("district_2024")["charger_uid"]
        .nunique()
    )

    speed_counts = (
        matched_df.pivot_table(
            index="district_2024",
            columns="charge_speed_class",
            values="charger_uid",
            aggfunc="nunique",
            fill_value=0,
        )
        .rename(
            columns={
                "fast": "fast_charger_count",
                "slow": "slow_charger_count",
                "unknown": "unknown_speed_count",
            }
        )
    )

    for column in ["fast_charger_count", "slow_charger_count", "unknown_speed_count"]:
        if column not in speed_counts.columns:
            speed_counts[column] = 0

    detail_summary = pd.concat(
        [
            station_count.rename("station_count"),
            charger_count.rename("charger_count"),
            public_charger_count.rename("public_charger_count"),
            limited_charger_count.rename("limited_charger_count"),
            speed_counts[
                ["fast_charger_count", "slow_charger_count", "unknown_speed_count"]
            ],
        ],
        axis=1,
    ).fillna(0)

    count_columns = [
        "station_count",
        "charger_count",
        "public_charger_count",
        "limited_charger_count",
        "fast_charger_count",
        "slow_charger_count",
        "unknown_speed_count",
    ]
    detail_summary[count_columns] = detail_summary[count_columns].astype(int)

    detail_summary["avg_chargers_per_station"] = safe_divide(
        detail_summary["charger_count"],
        detail_summary["station_count"],
        multiplier=1,
        metric_name="avg_chargers_per_station",
    )
    detail_summary["public_charger_ratio_percent"] = safe_divide(
        detail_summary["public_charger_count"],
        detail_summary["charger_count"],
        multiplier=100,
        metric_name="public_charger_ratio_percent",
    )
    detail_summary["fast_charger_ratio_percent"] = safe_divide(
        detail_summary["fast_charger_count"],
        detail_summary["charger_count"],
        multiplier=100,
        metric_name="fast_charger_ratio_percent",
    )

    return detail_summary.reset_index()


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    multiplier: float,
    metric_name: str,
) -> pd.Series:
    """0 나눗셈을 검증하고 소수점 둘째 자리까지 계산한다."""
    zero_mask = denominator.eq(0)
    if zero_mask.any():
        raise ValueError(f"{metric_name} 계산에서 0으로 나누는 행이 있습니다.")

    result = numerator / denominator * multiplier
    if np.isinf(result).any():
        raise ValueError(f"{metric_name} 계산 결과에 무한대가 있습니다.")

    return result.round(2)


def build_charging_infra_summary(charging_summary_df: pd.DataFrame) -> pd.DataFrame:
    """요약 파일을 우선 사용하고 누락 열은 상세 파일 집계로 보강한다."""
    charging_df = normalize_district_column(charging_summary_df, "충전소 요약")

    missing_infra_columns = [
        column for column in CHARGING_INFRA_COLUMNS if column not in charging_df.columns
    ]

    if missing_infra_columns:
        print(f"충전소 요약 파일에 없는 인프라 열: {missing_infra_columns}")
        detail_summary = aggregate_charging_detail()

        comparison_df = charging_df.merge(
            detail_summary[["district_2024", "station_count", "charger_count"]],
            on="district_2024",
            how="inner",
            suffixes=("_summary", "_detail"),
            validate="one_to_one",
        )

        for column in ["station_count", "charger_count"]:
            mismatch_mask = (
                comparison_df[f"{column}_summary"] != comparison_df[f"{column}_detail"]
            )
            if mismatch_mask.any():
                raise ValueError(
                    f"충전소 요약 파일과 상세 재집계의 {column} 값이 다릅니다:\n"
                    f"{comparison_df.loc[mismatch_mask].to_string(index=False)}"
                )

        charging_df = detail_summary

    missing_after_fill = [
        column for column in CHARGING_INFRA_COLUMNS if column not in charging_df.columns
    ]
    if missing_after_fill:
        raise KeyError(f"충전 인프라 최종 지표 열이 부족합니다: {missing_after_fill}")

    return charging_df[CHARGING_INFRA_COLUMNS].copy()


def prepare_registration_summary(registration_df: pd.DataFrame) -> pd.DataFrame:
    """등록 현황 데이터를 병합 가능한 형태로 정리한다."""
    prepared_df = normalize_district_column(registration_df, "등록 현황")
    missing_columns = [
        column for column in EV_REGISTRATION_COLUMNS if column not in prepared_df.columns
    ]
    if missing_columns:
        raise KeyError(f"등록 현황 필수 열이 없습니다: {missing_columns}")

    return prepared_df[["district_2024", *EV_REGISTRATION_COLUMNS]].copy()


def validate_totals(charging_df: pd.DataFrame, registration_df: pd.DataFrame) -> None:
    """핵심 합계가 기대값과 일치하는지 확인한다."""
    charger_total = int(charging_df["charger_count"].sum())
    ev_total = int(registration_df["total_ev_count"].sum())

    if charger_total != EXPECTED_CHARGER_TOTAL:
        raise ValueError(
            f"충전기 수 합계가 {EXPECTED_CHARGER_TOTAL:,}이 아닙니다. "
            f"실제 합계: {charger_total:,}"
        )

    if ev_total != EXPECTED_EV_TOTAL:
        raise ValueError(
            f"전기차 등록 대수 합계가 {EXPECTED_EV_TOTAL:,}이 아닙니다. "
            f"실제 합계: {ev_total:,}"
        )


def merge_datasets(charging_df: pd.DataFrame, registration_df: pd.DataFrame) -> pd.DataFrame:
    """outer merge와 indicator로 누락 여부를 확인한 뒤 one-to-one 병합 결과를 반환한다."""
    merged_df = charging_df.merge(
        registration_df,
        on="district_2024",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    not_both_df = merged_df.loc[~merged_df["_merge"].eq("both")]
    if not not_both_df.empty:
        raise ValueError(
            "병합 누락 행이 있습니다:\n"
            f"{not_both_df[['district_2024', '_merge']].to_string(index=False)}"
        )

    merged_df = merged_df.drop(columns=["_merge"])

    if len(merged_df) != 10:
        raise AssertionError(f"병합 결과가 10행이 아닙니다: {len(merged_df):,}")

    return merged_df


def add_supply_demand_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """수요 대비 공급 지표와 순위를 계산한다."""
    result_df = df.copy()

    result_df["chargers_per_100_ev"] = safe_divide(
        result_df["charger_count"],
        result_df["total_ev_count"],
        multiplier=100,
        metric_name="chargers_per_100_ev",
    )
    result_df["public_chargers_per_100_ev"] = safe_divide(
        result_df["public_charger_count"],
        result_df["total_ev_count"],
        multiplier=100,
        metric_name="public_chargers_per_100_ev",
    )
    result_df["fast_chargers_per_100_ev"] = safe_divide(
        result_df["fast_charger_count"],
        result_df["total_ev_count"],
        multiplier=100,
        metric_name="fast_chargers_per_100_ev",
    )
    result_df["ev_per_charger"] = safe_divide(
        result_df["total_ev_count"],
        result_df["charger_count"],
        multiplier=1,
        metric_name="ev_per_charger",
    )
    result_df["ev_per_public_charger"] = safe_divide(
        result_df["total_ev_count"],
        result_df["public_charger_count"],
        multiplier=1,
        metric_name="ev_per_public_charger",
    )
    result_df["stations_per_1000_ev"] = safe_divide(
        result_df["station_count"],
        result_df["total_ev_count"],
        multiplier=1000,
        metric_name="stations_per_1000_ev",
    )

    result_df["weighted_ev_demand_equal"] = (
        result_df["private_ev_count"] + result_df["commercial_ev_count"]
    )

    # 사업용 가중치 1.5는 공식 수요량이 아니라 민감도 분석용 가정이다.
    # 따라서 최종 정답처럼 해석하지 않고 보조 지표로만 사용한다.
    result_df["weighted_ev_demand_commercial_1_5"] = (
        result_df["private_ev_count"] + result_df["commercial_ev_count"] * 1.5
    ).round(2)
    result_df["chargers_per_100_weighted_ev"] = safe_divide(
        result_df["charger_count"],
        result_df["weighted_ev_demand_commercial_1_5"],
        multiplier=100,
        metric_name="chargers_per_100_weighted_ev",
    )

    result_df["charger_count_rank"] = (
        result_df["charger_count"].rank(method="min", ascending=False).astype(int)
    )
    result_df["ev_count_rank"] = (
        result_df["total_ev_count"].rank(method="min", ascending=False).astype(int)
    )

    # 아래 100대당 공급 지표 순위는 값이 낮을수록 1위이다.
    # 이는 인프라 부족 가능성이 높은 지역 순위를 뜻한다.
    result_df["chargers_per_100_ev_rank"] = (
        result_df["chargers_per_100_ev"].rank(method="min", ascending=True).astype(int)
    )
    result_df["public_chargers_per_100_ev_rank"] = (
        result_df["public_chargers_per_100_ev"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    result_df["fast_chargers_per_100_ev_rank"] = (
        result_df["fast_chargers_per_100_ev"].rank(method="min", ascending=True).astype(int)
    )

    result_df["registration_base_date"] = REGISTRATION_BASE_DATE
    result_df["boundary_base_date"] = BOUNDARY_BASE_DATE
    result_df["charging_data_base_date"] = CHARGING_DATA_BASE_DATE

    return result_df[OUTPUT_COLUMNS].copy()


def validate_final_output(df: pd.DataFrame) -> None:
    """최종 결과의 행 수, 합계, 결측, 음수, 비율 무한대를 검증한다."""
    if len(df) != 10:
        raise AssertionError(f"최종 결과가 10행이 아닙니다: {len(df):,}")

    if df["district_2024"].duplicated().any():
        raise AssertionError("최종 결과에 군·구 중복이 있습니다.")

    total_ev_sum = int(df["total_ev_count"].sum())
    charger_sum = int(df["charger_count"].sum())

    if total_ev_sum != EXPECTED_EV_TOTAL:
        raise AssertionError(
            f"total_ev_count 합계가 {EXPECTED_EV_TOTAL:,}이 아닙니다: {total_ev_sum:,}"
        )

    if charger_sum != EXPECTED_CHARGER_TOTAL:
        raise AssertionError(
            f"charger_count 합계가 {EXPECTED_CHARGER_TOTAL:,}이 아닙니다: {charger_sum:,}"
        )

    if (df["public_charger_count"] > df["charger_count"]).any():
        raise AssertionError("public_charger_count가 charger_count보다 큰 행이 있습니다.")

    speed_total = (
        df["fast_charger_count"] + df["slow_charger_count"] + df["unknown_speed_count"]
    )
    if not speed_total.eq(df["charger_count"]).all():
        invalid_rows = df.loc[
            ~speed_total.eq(df["charger_count"]),
            [
                "district_2024",
                "charger_count",
                "fast_charger_count",
                "slow_charger_count",
                "unknown_speed_count",
            ],
        ]
        raise AssertionError(
            "fast + slow + unknown이 charger_count와 다릅니다:\n"
            f"{invalid_rows.to_string(index=False)}"
        )

    if df[OUTPUT_COLUMNS].isna().any().any():
        missing_counts = df[OUTPUT_COLUMNS].isna().sum()
        raise AssertionError(
            "주요 열에 결측값이 있습니다:\n"
            f"{missing_counts[missing_counts > 0].to_string()}"
        )

    numeric_df = df.select_dtypes(include=["number"])

    if np.isinf(numeric_df.to_numpy()).any():
        raise AssertionError("수치 열에 무한대 값이 있습니다.")

    if (numeric_df < 0).any().any():
        negative_counts = (numeric_df < 0).sum()
        raise AssertionError(
            "음수 수치가 있습니다:\n"
            f"{negative_counts[negative_counts > 0].to_string()}"
        )


def configure_matplotlib_font() -> None:
    """macOS 한글 표시를 위해 AppleGothic을 우선 사용한다."""
    available_font_names = {font.name for font in fm.fontManager.ttflist}
    if "AppleGothic" in available_font_names:
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        print("경고: AppleGothic 폰트를 찾지 못했습니다. 기본 폰트로 그래프를 생성합니다.")

    plt.rcParams["axes.unicode_minus"] = False


def annotate_bars(ax: plt.Axes, values: pd.Series) -> None:
    """막대 위에 값 라벨을 표시한다."""
    max_value = float(values.max()) if len(values) else 0
    offset = max_value * 0.01 if max_value else 0.1

    for bar, value in zip(ax.patches, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_ev_and_chargers(df: pd.DataFrame) -> Path:
    """군·구별 등록 전기차 수와 충전기 수를 이중 y축으로 비교한다."""
    plot_df = df.sort_values("total_ev_count", ascending=False).reset_index(drop=True)

    fig, ax_ev = plt.subplots(figsize=(12, 6))
    x = np.arange(len(plot_df))
    width = 0.38

    ev_bars = ax_ev.bar(
        x - width / 2,
        plot_df["total_ev_count"],
        width=width,
        color="#4c78a8",
        label=f"전기차 등록 대수({REGISTRATION_BASE_DATE})",
    )
    ax_charger = ax_ev.twinx()
    charger_bars = ax_charger.bar(
        x + width / 2,
        plot_df["charger_count"],
        width=width,
        color="#f58518",
        label=f"충전기 수({CHARGING_DATA_BASE_DATE})",
    )

    ax_ev.set_title(
        "군·구별 전기차 등록 대수와 충전기 수 비교"
        f"(등록 {REGISTRATION_BASE_DATE}, 충전 {CHARGING_DATA_BASE_DATE})"
    )
    ax_ev.set_xlabel("군·구")
    ax_ev.set_ylabel("전기차 등록 대수")
    ax_charger.set_ylabel("충전기 수")
    ax_ev.set_xticks(x)
    ax_ev.set_xticklabels(plot_df["district_2024"], rotation=45)

    handles = [ev_bars, charger_bars]
    labels = [handle.get_label() for handle in handles]
    ax_ev.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(EV_AND_CHARGERS_FIGURE, dpi=220)
    plt.close(fig)

    return EV_AND_CHARGERS_FIGURE


def plot_metric_bar(
    df: pd.DataFrame,
    metric_column: str,
    title: str,
    ylabel: str,
    output_file: Path,
    annotate: bool = True,
) -> Path:
    """낮은 값 순으로 공급 지표 막대그래프를 생성한다."""
    plot_df = df.sort_values(metric_column, ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(plot_df["district_2024"], plot_df[metric_column], color="#59a14f")

    if annotate:
        annotate_bars(ax, plot_df[metric_column])

    ax.set_title(title)
    ax.set_xlabel("군·구")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, plot_df[metric_column].max() * 1.15)

    fig.tight_layout()
    fig.savefig(output_file, dpi=220)
    plt.close(fig)

    return output_file


def create_figures(df: pd.DataFrame) -> list[Path]:
    """요구된 PNG 그래프를 생성한다."""
    configure_matplotlib_font()

    figure_paths = [
        plot_ev_and_chargers(df),
        plot_metric_bar(
            df,
            "chargers_per_100_ev",
            f"{REGISTRATION_BASE_LABEL} 등록 전기차 100대당 "
            f"{CHARGING_DATA_BASE_LABEL} 충전기 수",
            "충전기 수 / 전기차 100대",
            CHARGERS_PER_100_EV_FIGURE,
            annotate=True,
        ),
        plot_metric_bar(
            df,
            "public_chargers_per_100_ev",
            f"{REGISTRATION_BASE_LABEL} 등록 전기차 100대당 "
            f"{CHARGING_DATA_BASE_LABEL} 일반 이용 가능 충전기 수",
            "일반 이용 가능 충전기 수 / 전기차 100대",
            PUBLIC_CHARGERS_PER_100_EV_FIGURE,
            annotate=True,
        ),
        plot_metric_bar(
            df,
            "fast_chargers_per_100_ev",
            f"{REGISTRATION_BASE_LABEL} 등록 전기차 100대당 "
            f"{CHARGING_DATA_BASE_LABEL} 급속 충전기 수",
            "급속 충전기 수 / 전기차 100대",
            FAST_CHARGERS_PER_100_EV_FIGURE,
            annotate=False,
        ),
    ]

    return figure_paths


def save_outputs(df: pd.DataFrame) -> tuple[list[Path], list[Path]]:
    """CSV와 shortage ranking 파일을 저장한다."""
    shortage_ranking_df = df.sort_values(
        "public_chargers_per_100_ev",
        ascending=True,
    ).reset_index(drop=True)

    df.to_csv(PROCESSED_OUTPUT_FILE, index=False, encoding="utf-8-sig")
    df.to_csv(SUMMARY_OUTPUT_FILE, index=False, encoding="utf-8-sig")
    shortage_ranking_df.to_csv(
        SHORTAGE_RANKING_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    figure_paths = create_figures(df)
    csv_paths = [PROCESSED_OUTPUT_FILE, SUMMARY_OUTPUT_FILE, SHORTAGE_RANKING_OUTPUT_FILE]

    return csv_paths, figure_paths


def print_terminal_summary(df: pd.DataFrame, csv_paths: list[Path], figure_paths: list[Path]) -> None:
    """요구된 터미널 요약을 출력한다."""
    print("\n[병합 결과]")
    print(f"병합 결과 행 수: {len(df):,}")
    print(f"전기차 등록 대수 합계: {int(df['total_ev_count'].sum()):,}")
    print(f"충전기 수 합계: {int(df['charger_count'].sum()):,}")

    summary_columns = [
        "district_2024",
        "total_ev_count",
        "charger_count",
        "public_charger_count",
        "fast_charger_count",
        "chargers_per_100_ev",
        "public_chargers_per_100_ev",
        "fast_chargers_per_100_ev",
        "ev_per_charger",
    ]
    print("\n[군·구별 수요·공급 요약표]")
    print(df.sort_values("district_2024")[summary_columns].to_string(index=False))

    print("\n[chargers_per_100_ev 낮은 5개 지역]")
    print(
        df.sort_values("chargers_per_100_ev")
        [["district_2024", "chargers_per_100_ev"]]
        .head(5)
        .to_string(index=False)
    )

    print("\n[public_chargers_per_100_ev 낮은 5개 지역]")
    print(
        df.sort_values("public_chargers_per_100_ev")
        [["district_2024", "public_chargers_per_100_ev"]]
        .head(5)
        .to_string(index=False)
    )

    print("\n[fast_chargers_per_100_ev 낮은 5개 지역]")
    print(
        df.sort_values("fast_chargers_per_100_ev")
        [["district_2024", "fast_chargers_per_100_ev"]]
        .head(5)
        .to_string(index=False)
    )

    print("\n[지표 해석]")
    print(
        "모든 100대당 지표는 "
        f"'{REGISTRATION_BASE_LABEL} 등록 전기차 100대당 "
        f"{CHARGING_DATA_BASE_LABEL} 충전기 수'입니다."
    )

    print("\n[생성 CSV]")
    for path in csv_paths:
        print(path)

    print("\n[생성 PNG]")
    for path in figure_paths:
        print(path)


def main() -> None:
    ensure_output_dirs()

    charging_summary_raw_df = read_csv(CHARGING_SUMMARY_FILE)
    registration_raw_df = read_csv(EV_REGISTRATION_FILE)

    charging_df = build_charging_infra_summary(charging_summary_raw_df)
    registration_df = prepare_registration_summary(registration_raw_df)

    print_district_lists(charging_df, registration_df)
    validate_districts(charging_df, registration_df)
    validate_totals(charging_df, registration_df)

    merged_df = merge_datasets(charging_df, registration_df)
    final_df = add_supply_demand_metrics(merged_df)
    validate_final_output(final_df)

    csv_paths, figure_paths = save_outputs(final_df)
    print_terminal_summary(final_df, csv_paths, figure_paths)


if __name__ == "__main__":
    main()
