from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "charging_stations_incheon_final.csv"
)

OUTPUT_TABLE_DIR = PROJECT_DIR / "outputs" / "tables"
OUTPUT_FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"

DISTRICT_TYPE_SUMMARY_FILE = (
    OUTPUT_TABLE_DIR
    / "district_charging_type_summary.csv"
)
SPEED_SUMMARY_FILE = (
    OUTPUT_TABLE_DIR
    / "charging_speed_summary.csv"
)

CHARGERS_BY_DISTRICT_FIGURE = (
    OUTPUT_FIGURE_DIR
    / "chargers_by_district.png"
)
FAST_SLOW_BY_DISTRICT_FIGURE = (
    OUTPUT_FIGURE_DIR
    / "fast_slow_chargers_by_district.png"
)
PUBLIC_RATIO_BY_DISTRICT_FIGURE = (
    OUTPUT_FIGURE_DIR
    / "public_charger_ratio_by_district.png"
)

REQUIRED_COLUMNS = [
    "charger_uid",
    "statId",
    "district_final",
    "chgerType",
    "output",
    "is_deleted",
    "is_limited",
    "publicly_accessible",
    "usable_for_district_analysis",
]

BOOLEAN_COLUMNS = [
    "is_deleted",
    "is_limited",
    "publicly_accessible",
    "usable_for_district_analysis",
]

CURRENT_DISTRICT_COUNT = 11
LEGACY_DISTRICT_NAMES = [
    "중구_과거코드",
    "서구_과거코드",
]

FAST_CHARGING_THRESHOLD_KW = 50


def require_input_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"입력 파일을 찾을 수 없습니다:\n{path}"
        )


def read_charging_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )


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
            "분석에 필요한 열을 찾을 수 없습니다:\n"
            f"{missing_columns}"
        )


def normalize_boolean_columns(
    dataframe: pd.DataFrame,
    boolean_columns: list[str],
) -> pd.DataFrame:
    normalized_df = dataframe.copy()
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

    for column in boolean_columns:
        if normalized_df[column].dtype == "bool":
            continue

        converted = (
            normalized_df[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .map(boolean_map)
        )

        invalid_mask = (
            normalized_df[column].notna()
            & converted.isna()
        )

        if invalid_mask.any():
            invalid_values = (
                normalized_df.loc[invalid_mask, column]
                .drop_duplicates()
                .tolist()
            )
            raise ValueError(
                f"{column} 열에서 Boolean으로 변환할 수 없는 값: "
                f"{invalid_values}"
            )

        normalized_df[column] = (
            converted
            .fillna(False)
            .astype(bool)
        )

    return normalized_df


def split_analysis_data(
    charging_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active_df = charging_df.loc[
        ~charging_df["is_deleted"]
        & charging_df["usable_for_district_analysis"]
    ].copy()

    legacy_district_df = active_df.loc[
        active_df["district_final"].isin(LEGACY_DISTRICT_NAMES)
    ].copy()

    district_df = active_df.loc[
        ~active_df["district_final"].isin(LEGACY_DISTRICT_NAMES)
    ].copy()

    districts = sorted(
        district_df["district_final"]
        .dropna()
        .unique()
        .tolist()
    )

    if len(districts) != CURRENT_DISTRICT_COUNT:
        print(
            "경고: 현재 군·구 수가 "
            f"{CURRENT_DISTRICT_COUNT}개가 아닙니다."
        )
        print(f"실제 군·구 수: {len(districts)}")
        print(f"실제 군·구 목록: {districts}")

    return active_df, district_df, legacy_district_df


def print_charger_type_diagnostics(dataframe: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("충전기 유형 현황 점검")
    print("=" * 70)

    print("\n[chgerType 빈도]")
    print(
        dataframe["chgerType"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    output_numeric = pd.to_numeric(
        dataframe["output"],
        errors="coerce",
    )

    type_output_summary = (
        dataframe
        .assign(output_numeric=output_numeric)
        .groupby("chgerType", dropna=False)
        .agg(
            charger_count=("charger_uid", "nunique"),
            output_min=("output_numeric", "min"),
            output_median=("output_numeric", "median"),
            output_max=("output_numeric", "max"),
            output_missing_count=(
                "output_numeric",
                lambda series: int(series.isna().sum()),
            ),
        )
        .reset_index()
    )

    print("\n[chgerType별 output 요약]")
    print(type_output_summary.to_string(index=False))

    output_missing_count = int(output_numeric.isna().sum())
    output_missing_ratio = (
        output_missing_count
        / len(dataframe)
        * 100
    )

    print("\n[output 결측]")
    print(f"output 결측 행 수: {output_missing_count:,}")
    print(f"output 결측 비율: {output_missing_ratio:.2f}%")


def add_charge_speed_class(dataframe: pd.DataFrame) -> pd.DataFrame:
    classified_df = dataframe.copy()
    classified_df["output"] = pd.to_numeric(
        classified_df["output"],
        errors="coerce",
    )

    classified_df["charge_speed_class"] = "unknown"

    # 충전용량 기준의 프로젝트 분석용 임시 분류이다.
    # 공식 chgerType 코드 기반 급속·완속 분류와 다를 수 있다.
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


def count_unique_by_class(
    dataframe: pd.DataFrame,
    class_name: str,
) -> pd.Series:
    return (
        dataframe.loc[
            dataframe["charge_speed_class"].eq(class_name)
        ]
        .groupby("district_final")["charger_uid"]
        .nunique()
    )


def build_district_summary(district_df: pd.DataFrame) -> pd.DataFrame:
    station_count = (
        district_df
        .groupby("district_final")["statId"]
        .nunique()
        .rename("station_count")
    )

    charger_count = (
        district_df
        .groupby("district_final")["charger_uid"]
        .nunique()
        .rename("charger_count")
    )

    fast_charger_count = count_unique_by_class(
        district_df,
        "fast",
    ).rename("fast_charger_count")

    slow_charger_count = count_unique_by_class(
        district_df,
        "slow",
    ).rename("slow_charger_count")

    unknown_speed_count = count_unique_by_class(
        district_df,
        "unknown",
    ).rename("unknown_speed_count")

    public_charger_count = (
        district_df.loc[
            district_df["publicly_accessible"]
        ]
        .groupby("district_final")["charger_uid"]
        .nunique()
        .rename("public_charger_count")
    )

    limited_charger_count = (
        district_df.loc[
            district_df["is_limited"]
        ]
        .groupby("district_final")["charger_uid"]
        .nunique()
        .rename("limited_charger_count")
    )

    summary = pd.concat(
        [
            station_count,
            charger_count,
            fast_charger_count,
            slow_charger_count,
            unknown_speed_count,
            public_charger_count,
            limited_charger_count,
        ],
        axis=1,
    ).fillna(0)

    count_columns = [
        "station_count",
        "charger_count",
        "fast_charger_count",
        "slow_charger_count",
        "unknown_speed_count",
        "public_charger_count",
        "limited_charger_count",
    ]

    summary[count_columns] = summary[count_columns].astype(int)

    summary["fast_charger_ratio_percent"] = (
        summary["fast_charger_count"]
        / summary["charger_count"]
        * 100
    ).round(2)

    summary["public_charger_ratio_percent"] = (
        summary["public_charger_count"]
        / summary["charger_count"]
        * 100
    ).round(2)

    summary["avg_chargers_per_station"] = (
        summary["charger_count"]
        / summary["station_count"]
    ).round(2)

    return (
        summary
        .sort_values("charger_count", ascending=False)
        .reset_index()
    )


def build_speed_summary(active_df: pd.DataFrame) -> pd.DataFrame:
    speed_order = ["fast", "slow", "unknown"]
    total_charger_count = active_df["charger_uid"].nunique()

    speed_summary = (
        active_df
        .groupby("charge_speed_class")["charger_uid"]
        .nunique()
        .reindex(speed_order, fill_value=0)
        .rename("charger_count")
        .reset_index()
    )

    speed_summary["ratio_percent"] = (
        speed_summary["charger_count"]
        / total_charger_count
        * 100
    ).round(2)

    return speed_summary


def configure_matplotlib_font() -> None:
    available_font_names = {
        font.name
        for font in fm.fontManager.ttflist
    }

    if "AppleGothic" in available_font_names:
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        print(
            "경고: AppleGothic 폰트를 찾지 못했습니다. "
            "기본 폰트로 그래프를 생성합니다."
        )

    plt.rcParams["axes.unicode_minus"] = False


def annotate_bars(
    ax: plt.Axes,
    values: pd.Series,
    percent: bool = False,
) -> None:
    for bar, value in zip(ax.patches, values):
        if percent:
            label = f"{value:.1f}%"
        else:
            label = f"{int(value):,}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_chargers_by_district(
    district_summary: pd.DataFrame,
) -> Path:
    plot_df = district_summary.sort_values(
        "charger_count",
        ascending=False,
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        plot_df["district_final"],
        plot_df["charger_count"],
        color="#4c78a8",
    )

    annotate_bars(ax, plot_df["charger_count"])

    ax.set_title("인천광역시 군·구별 운영 중 전기차 충전기 수")
    ax.set_xlabel("군·구")
    ax.set_ylabel("충전기 수")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(CHARGERS_BY_DISTRICT_FIGURE, dpi=200)
    plt.close(fig)

    return CHARGERS_BY_DISTRICT_FIGURE


def plot_fast_slow_by_district(
    district_summary: pd.DataFrame,
) -> Path:
    plot_df = district_summary.sort_values(
        "charger_count",
        ascending=False,
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    bottom = pd.Series(0, index=plot_df.index)
    stack_columns = [
        ("slow_charger_count", "완속", "#72b7b2"),
        ("fast_charger_count", "급속", "#f58518"),
        ("unknown_speed_count", "미분류", "#bab0ac"),
    ]

    for column, label, color in stack_columns:
        values = plot_df[column]

        if column == "unknown_speed_count" and values.sum() == 0:
            continue

        bars = ax.bar(
            plot_df["district_final"],
            values,
            bottom=bottom,
            label=label,
            color=color,
        )

        for bar, value, base in zip(bars, values, bottom):
            if value < 100:
                continue

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                base + value / 2,
                f"{int(value):,}",
                ha="center",
                va="center",
                fontsize=7,
            )

        bottom = bottom + values

    for index, total in zip(plot_df.index, plot_df["charger_count"]):
        ax.text(
            plot_df.index.get_loc(index),
            total + plot_df["charger_count"].max() * 0.01,
            f"{int(total):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title("인천광역시 군·구별 급속·완속 충전기 구성")
    ax.set_xlabel("군·구")
    ax.set_ylabel("충전기 수")
    ax.set_ylim(0, plot_df["charger_count"].max() * 1.12)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FAST_SLOW_BY_DISTRICT_FIGURE, dpi=200)
    plt.close(fig)

    return FAST_SLOW_BY_DISTRICT_FIGURE


def plot_public_ratio_by_district(
    district_summary: pd.DataFrame,
) -> Path:
    plot_df = district_summary.sort_values(
        "public_charger_ratio_percent",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        plot_df["district_final"],
        plot_df["public_charger_ratio_percent"],
        color="#59a14f",
    )

    annotate_bars(
        ax,
        plot_df["public_charger_ratio_percent"],
        percent=True,
    )

    ax.set_title("인천광역시 군·구별 일반 이용 가능 충전기 비율")
    ax.set_xlabel("군·구")
    ax.set_ylabel("일반 이용 가능 충전기 비율(%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(PUBLIC_RATIO_BY_DISTRICT_FIGURE, dpi=200)
    plt.close(fig)

    return PUBLIC_RATIO_BY_DISTRICT_FIGURE


def validate_outputs(
    active_df: pd.DataFrame,
    district_df: pd.DataFrame,
    district_summary: pd.DataFrame,
    speed_summary: pd.DataFrame,
) -> None:
    active_charger_count = active_df["charger_uid"].nunique()
    district_charger_count = district_df["charger_uid"].nunique()

    if district_df["charger_uid"].duplicated().any():
        raise ValueError("분석 대상에서 charger_uid 중복이 발견되었습니다.")

    speed_total = int(speed_summary["charger_count"].sum())
    if speed_total != active_charger_count:
        raise AssertionError(
            "급속+완속+미분류 합계가 전체 운영 충전기 수와 "
            f"다릅니다: {speed_total:,} != {active_charger_count:,}"
        )

    district_total = int(district_summary["charger_count"].sum())
    if district_total != district_charger_count:
        raise AssertionError(
            "군·구별 charger_count 합계가 현재 군·구 비교 대상 "
            "충전기 수와 다릅니다: "
            f"{district_total:,} != {district_charger_count:,}"
        )

    public_limited_total = (
        district_summary["public_charger_count"]
        + district_summary["limited_charger_count"]
    )
    invalid_public_limited = (
        public_limited_total
        != district_summary["charger_count"]
    )

    if invalid_public_limited.any():
        invalid_rows = district_summary.loc[
            invalid_public_limited,
            [
                "district_final",
                "charger_count",
                "public_charger_count",
                "limited_charger_count",
            ],
        ]
        raise AssertionError(
            "public_charger_count + limited_charger_count가 "
            "charger_count와 다른 군·구가 있습니다:\n"
            f"{invalid_rows.to_string(index=False)}"
        )

    if (district_summary["station_count"] == 0).any():
        invalid_districts = (
            district_summary.loc[
                district_summary["station_count"].eq(0),
                "district_final",
            ]
            .tolist()
        )
        raise AssertionError(
            "station_count가 0인 군·구가 있습니다:\n"
            f"{invalid_districts}"
        )


def save_tables(
    district_summary: pd.DataFrame,
    speed_summary: pd.DataFrame,
) -> list[Path]:
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    district_summary.to_csv(
        DISTRICT_TYPE_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    speed_summary.to_csv(
        SPEED_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return [
        DISTRICT_TYPE_SUMMARY_FILE,
        SPEED_SUMMARY_FILE,
    ]


def save_figures(district_summary: pd.DataFrame) -> list[Path]:
    OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib_font()

    return [
        plot_chargers_by_district(district_summary),
        plot_fast_slow_by_district(district_summary),
        plot_public_ratio_by_district(district_summary),
    ]


def print_final_summary(
    active_df: pd.DataFrame,
    district_df: pd.DataFrame,
    district_summary: pd.DataFrame,
    speed_summary: pd.DataFrame,
    table_paths: list[Path],
    figure_paths: list[Path],
) -> None:
    districts = sorted(
        district_df["district_final"]
        .dropna()
        .unique()
        .tolist()
    )

    speed_counts = (
        speed_summary
        .set_index("charge_speed_class")["charger_count"]
        .to_dict()
    )

    fast_count = int(speed_counts.get("fast", 0))
    slow_count = int(speed_counts.get("slow", 0))
    unknown_count = int(speed_counts.get("unknown", 0))
    total_count = active_df["charger_uid"].nunique()
    district_total_count = district_df["charger_uid"].nunique()
    fast_ratio = fast_count / total_count * 100

    print("\n" + "=" * 70)
    print("최종 출력 검증")
    print("=" * 70)
    print(f"전체 운영 분석 대상 행 수: {len(active_df):,}")
    print(f"현재 군·구 비교 분석 대상 행 수: {len(district_df):,}")
    print(f"현재 군·구 수: {len(districts)}")
    print(f"현재 군·구 목록: {districts}")
    print(f"전체 충전기 수: {total_count:,}")
    print(f"현재 군·구 비교 충전기 수: {district_total_count:,}")
    print(f"급속 충전기 수: {fast_count:,}")
    print(f"완속 충전기 수: {slow_count:,}")
    print(f"미분류 수: {unknown_count:,}")
    print(f"급속 비율: {fast_ratio:.2f}%")

    print("\n[군·구별 요약표]")
    print(district_summary.to_string(index=False))

    print("\n[생성 CSV]")
    for path in table_paths:
        print(path)

    print("\n[생성 PNG]")
    for path in figure_paths:
        print(path)


def main() -> None:
    require_input_file(INPUT_FILE)
    charging_df = read_charging_data(INPUT_FILE)
    require_columns(charging_df, REQUIRED_COLUMNS)
    charging_df = normalize_boolean_columns(
        charging_df,
        BOOLEAN_COLUMNS,
    )

    active_df, district_df, legacy_district_df = split_analysis_data(
        charging_df
    )

    print_charger_type_diagnostics(active_df)

    active_df = add_charge_speed_class(active_df)
    district_df = add_charge_speed_class(district_df)

    district_summary = build_district_summary(district_df)
    speed_summary = build_speed_summary(active_df)

    validate_outputs(
        active_df,
        district_df,
        district_summary,
        speed_summary,
    )

    table_paths = save_tables(district_summary, speed_summary)
    figure_paths = save_figures(district_summary)

    print(
        "\n과거 코드 보존 행 수:",
        f"{len(legacy_district_df):,}"
    )

    print_final_summary(
        active_df,
        district_df,
        district_summary,
        speed_summary,
        table_paths,
        figure_paths,
    )


if __name__ == "__main__":
    main()


# 실행 방법:
# .venv/bin/python src/classify_and_visualize_chargers.py
#
# 생성 파일:
# outputs/tables/district_charging_type_summary.csv
# outputs/tables/charging_speed_summary.csv
# outputs/figures/chargers_by_district.png
# outputs/figures/fast_slow_chargers_by_district.png
# outputs/figures/public_charger_ratio_by_district.png
