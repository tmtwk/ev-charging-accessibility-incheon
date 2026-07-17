from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent

DISTRICT_ANALYSIS_FILE = (
    PROJECT_DIR / "outputs" / "tables" / "district_ev_charging_supply_demand_summary.csv"
)
CHARGING_FINAL_FILE = (
    PROJECT_DIR / "data" / "processed" / "charging_stations_incheon_final.csv"
)
CHARGING_2024_DISTRICT_FILE = (
    PROJECT_DIR / "data" / "processed" / "charging_stations_incheon_2024_district.csv"
)
STATION_LEVEL_FILE = (
    PROJECT_DIR / "data" / "processed" / "charging_station_level_spatial.csv"
)
DBSCAN_PARAMETER_FILE = (
    PROJECT_DIR / "outputs" / "tables" / "dbscan_parameter_comparison.csv"
)

OUTPUT_TABLE_DIR = PROJECT_DIR / "outputs" / "tables"
OUTPUT_FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"
OUTPUT_MAP_DIR = PROJECT_DIR / "outputs" / "maps"
REPORT_DIR = PROJECT_DIR / "reports"

FINAL_DISTRICT_SUMMARY_FILE = OUTPUT_TABLE_DIR / "final_district_analysis_summary.csv"
FINAL_MAP_INVENTORY_FILE = OUTPUT_TABLE_DIR / "final_map_inventory.csv"
FINAL_FIGURE_INVENTORY_FILE = OUTPUT_TABLE_DIR / "final_figure_inventory.csv"
DBSCAN_EXPLORATORY_SUMMARY_FILE = OUTPUT_TABLE_DIR / "dbscan_exploratory_summary.csv"
FINAL_DATA_QUALITY_FILE = OUTPUT_TABLE_DIR / "final_data_quality_summary.csv"
FINAL_NOTES_FILE = REPORT_DIR / "final_analysis_notes.md"

FINAL_FIGURES = {
    "ev_vs_chargers": OUTPUT_FIGURE_DIR / "final_ev_and_chargers_by_district.png",
    "public_per_100": OUTPUT_FIGURE_DIR / "final_public_chargers_per_100_ev_by_district.png",
    "fast_per_100": OUTPUT_FIGURE_DIR / "final_fast_chargers_per_100_ev_by_district.png",
    "public_ratio": OUTPUT_FIGURE_DIR / "final_public_charger_ratio_by_district.png",
}

FINAL_MAPS = [
    "station_marker_cluster_map.html",
    "station_capacity_circle_map.html",
    "public_charger_heatmap.html",
    "district_supply_choropleth_map.html",
]

REQUIRED_DISTRICT_COLUMNS = [
    "district_2024",
    "total_ev_count",
    "station_count",
    "charger_count",
    "public_charger_count",
    "limited_charger_count",
    "fast_charger_count",
    "slow_charger_count",
    "public_charger_ratio_percent",
    "fast_charger_ratio_percent",
    "chargers_per_100_ev",
    "public_chargers_per_100_ev",
    "fast_chargers_per_100_ev",
    "ev_per_public_charger",
    "registration_base_date",
    "boundary_base_date",
    "charging_data_base_date",
]

FINAL_DISTRICT_COLUMNS = [
    "district",
    "total_ev_count",
    "station_count",
    "charger_count",
    "public_charger_count",
    "fast_charger_count",
    "slow_charger_count",
    "public_charger_ratio_percent",
    "fast_charger_ratio_percent",
    "chargers_per_100_ev",
    "public_chargers_per_100_ev",
    "fast_chargers_per_100_ev",
    "ev_per_public_charger",
    "ev_count_rank",
    "charger_count_rank",
    "public_supply_shortage_rank",
    "fast_supply_shortage_rank",
    "registration_base_date",
    "boundary_base_date",
    "charging_data_base_date",
]


def ensure_dirs() -> None:
    """최종 산출물 저장 폴더를 생성한다."""
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    """입력 파일 존재 여부를 확인한다."""
    if not path.exists():
        raise FileNotFoundError(f"필요한 파일을 찾을 수 없습니다: {path}")


def read_csv(path: Path) -> pd.DataFrame:
    """CSV를 읽고 실제 열 이름을 출력한다."""
    require_file(path)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    print(f"\n[파일 확인] {path}")
    print(f"행 수: {len(df):,}")
    print(f"열 이름: {df.columns.tolist()}")
    return df


def require_columns(df: pd.DataFrame, required_columns: list[str], name: str) -> None:
    """필수 열이 실제 데이터에 있는지 확인한다."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"{name} 필수 열이 없습니다: {missing_columns}")


def configure_matplotlib_font() -> None:
    """macOS 한글 표시를 위해 AppleGothic을 우선 사용한다."""
    available_font_names = {font.name for font in fm.fontManager.ttflist}
    if "AppleGothic" in available_font_names:
        plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False


def build_final_district_summary(district_df: pd.DataFrame) -> pd.DataFrame:
    """최종 보고서용 군·구 핵심 분석표를 만든다."""
    require_columns(district_df, REQUIRED_DISTRICT_COLUMNS, "군·구 병합 분석표")
    final_df = district_df[REQUIRED_DISTRICT_COLUMNS].copy()
    final_df = final_df.rename(columns={"district_2024": "district"})

    final_df["ev_count_rank"] = (
        final_df["total_ev_count"].rank(method="min", ascending=False).astype(int)
    )
    final_df["charger_count_rank"] = (
        final_df["charger_count"].rank(method="min", ascending=False).astype(int)
    )
    final_df["public_supply_shortage_rank"] = (
        final_df["public_chargers_per_100_ev"].rank(method="min", ascending=True).astype(int)
    )
    final_df["fast_supply_shortage_rank"] = (
        final_df["fast_chargers_per_100_ev"].rank(method="min", ascending=True).astype(int)
    )

    final_df = final_df[FINAL_DISTRICT_COLUMNS].sort_values("district").reset_index(drop=True)
    validate_final_district_summary(final_df, district_df)
    final_df.to_csv(FINAL_DISTRICT_SUMMARY_FILE, index=False, encoding="utf-8-sig")

    return final_df


def validate_final_district_summary(final_df: pd.DataFrame, source_df: pd.DataFrame) -> None:
    """최종 군·구 분석표의 핵심 검증을 수행한다."""
    if len(final_df) != 10:
        raise AssertionError(f"최종 군·구 수가 10개가 아닙니다: {len(final_df):,}")

    if final_df["district"].duplicated().any():
        raise AssertionError("최종 분석표에 군·구 중복이 있습니다.")

    numeric_columns = final_df.select_dtypes(include=["number"]).columns.tolist()
    if final_df[FINAL_DISTRICT_COLUMNS].isna().any().any():
        missing_counts = final_df[FINAL_DISTRICT_COLUMNS].isna().sum()
        raise AssertionError(
            "최종 분석표 주요 열에 결측이 있습니다:\n"
            f"{missing_counts[missing_counts > 0].to_string()}"
        )

    if (final_df[numeric_columns] < 0).any().any():
        negative_counts = (final_df[numeric_columns] < 0).sum()
        raise AssertionError(
            "최종 분석표 수치 열에 음수가 있습니다:\n"
            f"{negative_counts[negative_counts > 0].to_string()}"
        )

    if (final_df["public_charger_count"] > final_df["charger_count"]).any():
        raise AssertionError("public_charger_count가 charger_count보다 큰 군·구가 있습니다.")

    if (final_df["fast_charger_count"] > final_df["charger_count"]).any():
        raise AssertionError("fast_charger_count가 charger_count보다 큰 군·구가 있습니다.")

    for column in [
        "total_ev_count",
        "station_count",
        "charger_count",
        "public_charger_count",
        "fast_charger_count",
        "slow_charger_count",
    ]:
        source_sum = int(source_df[column].sum())
        final_sum = int(final_df[column].sum())
        if source_sum != final_sum:
            raise AssertionError(
                f"{column} 합계가 기존 검증 표와 일치하지 않습니다: "
                f"final={final_sum:,}, source={source_sum:,}"
            )


def annotate_bars(ax: plt.Axes, values: pd.Series, fmt: str = "{:.2f}") -> None:
    """막대 위에 값 라벨을 표시한다."""
    max_value = float(values.max())
    offset = max_value * 0.01 if max_value else 0.1
    for bar, value in zip(ax.patches, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def create_final_figures(final_df: pd.DataFrame) -> pd.DataFrame:
    """최종 보고서용 핵심 그래프 4개를 새 파일명으로 생성한다."""
    configure_matplotlib_font()
    selected_figure_metadata = {}

    sorted_ev_df = final_df.sort_values("total_ev_count", ascending=False).reset_index(drop=True)
    fig, ax_ev = plt.subplots(figsize=(12, 6))
    x = np.arange(len(sorted_ev_df))
    width = 0.38
    ev_bars = ax_ev.bar(
        x - width / 2,
        sorted_ev_df["total_ev_count"],
        width=width,
        color="#4c78a8",
        label="전기차 등록 대수",
    )
    ax_charger = ax_ev.twinx()
    charger_bars = ax_charger.bar(
        x + width / 2,
        sorted_ev_df["charger_count"],
        width=width,
        color="#f58518",
        label="전체 충전기 수",
    )
    ax_ev.set_title("군·구별 전기차 등록 대수와 전체 충전기 수 비교")
    ax_ev.set_xlabel("군·구")
    ax_ev.set_ylabel("전기차 등록 대수")
    ax_charger.set_ylabel("전체 충전기 수")
    ax_ev.set_xticks(x)
    ax_ev.set_xticklabels(sorted_ev_df["district"], rotation=45)
    handles = [ev_bars, charger_bars]
    ax_ev.legend(handles, [handle.get_label() for handle in handles], loc="upper right")
    fig.tight_layout()
    fig.savefig(FINAL_FIGURES["ev_vs_chargers"], dpi=220)
    plt.close(fig)
    selected_figure_metadata[FINAL_FIGURES["ev_vs_chargers"].name] = {
        "purpose": "군·구별 전기차 등록 대수와 전체 충전기 수 비교",
        "source_note": "등록 2026-02-19, 충전소 2026-07",
    }

    bar_specs = [
        (
            "public_per_100",
            "public_chargers_per_100_ev",
            "전기차 100대당 일반 이용 가능 충전기 수",
            "일반 이용 가능 충전기 수 / 전기차 100대",
        ),
        (
            "fast_per_100",
            "fast_chargers_per_100_ev",
            "전기차 100대당 급속충전기 수",
            "급속충전기 수 / 전기차 100대",
        ),
        (
            "public_ratio",
            "public_charger_ratio_percent",
            "군·구별 전체 충전기 중 일반 이용 가능 충전기 비율",
            "일반 이용 가능 비율(%)",
        ),
    ]
    for key, metric, title, ylabel in bar_specs:
        plot_df = final_df.sort_values(metric, ascending=True).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(plot_df["district"], plot_df[metric], color="#59a14f")
        annotate_bars(ax, plot_df[metric])
        ax.set_title(title)
        ax.set_xlabel("군·구")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, float(plot_df[metric].max()) * 1.15)
        fig.tight_layout()
        fig.savefig(FINAL_FIGURES[key], dpi=220)
        plt.close(fig)
        selected_figure_metadata[FINAL_FIGURES[key].name] = {
            "purpose": title,
            "source_note": "등록 2026-02-19, 충전소 2026-07",
        }

    inventory_rows = []
    for path in sorted(OUTPUT_FIGURE_DIR.glob("*.png")):
        selected = path.name in selected_figure_metadata
        metadata = selected_figure_metadata.get(path.name, {})
        inventory_rows.append(
            {
                "figure_file": path.name,
                "path": str(path),
                "file_size_bytes": path.stat().st_size,
                "selected_for_report": selected,
                "purpose": metadata.get("purpose", "최종 핵심 그래프에서는 제외"),
                "source_note": metadata.get(
                    "source_note",
                    "기존 탐색·중간 산출물 또는 DBSCAN k-distance 보조 그래프",
                ),
            }
        )

    figure_inventory_df = pd.DataFrame(inventory_rows)
    figure_inventory_df.to_csv(FINAL_FIGURE_INVENTORY_FILE, index=False, encoding="utf-8-sig")

    return figure_inventory_df


def build_map_inventory() -> pd.DataFrame:
    """최종 지도 산출물 존재 여부와 해석 포인트를 정리한다."""
    map_descriptions = {
        "station_marker_cluster_map.html": (
            "충전소 위치 MarkerCluster",
            "충전소의 전체 위치 분포와 지역별 밀집 정도를 빠르게 확인한다.",
        ),
        "station_capacity_circle_map.html": (
            "충전소별 충전 규모 CircleMarker",
            "충전기 수가 많은 거점과 급속충전 비율이 높은 지점을 함께 확인한다.",
        ),
        "public_charger_heatmap.html": (
            "일반 이용 가능 충전기 HeatMap",
            "전체 충전기가 아니라 시민이 이용 가능한 충전기 밀집권을 확인한다.",
        ),
        "district_supply_choropleth_map.html": (
            "군·구별 일반 이용 가능 충전기 공급 단계구분도",
            "전기차 100대당 일반 이용 가능 충전기 수가 낮은 취약 후보 군·구를 확인한다.",
        ),
    }

    rows = []
    for filename in FINAL_MAPS:
        path = OUTPUT_MAP_DIR / filename
        rows.append(
            {
                "map_file": filename,
                "path": str(path),
                "exists": path.exists(),
                "file_size_bytes": path.stat().st_size if path.exists() else 0,
                "purpose": map_descriptions[filename][0],
                "interpretation_point": map_descriptions[filename][1],
                "selected_for_report": True,
            }
        )

    inventory_df = pd.DataFrame(rows)
    invalid_rows = inventory_df.loc[~inventory_df["exists"] | inventory_df["file_size_bytes"].eq(0)]
    if not invalid_rows.empty:
        raise FileNotFoundError(
            "최종 지도 HTML 파일이 없거나 크기가 0입니다:\n"
            f"{invalid_rows.to_string(index=False)}"
        )

    inventory_df.to_csv(FINAL_MAP_INVENTORY_FILE, index=False, encoding="utf-8-sig")
    return inventory_df


def build_dbscan_exploratory_summary(dbscan_df: pd.DataFrame) -> pd.DataFrame:
    """DBSCAN 실험 결과를 탐색적 보조 분석 요약표로 정리한다."""
    require_columns(
        dbscan_df,
        [
            "eps_meters",
            "min_samples",
            "cluster_count",
            "noise_station_count",
            "noise_ratio_percent",
            "largest_cluster_ratio_percent",
        ],
        "DBSCAN 파라미터 비교표",
    )
    summary_df = dbscan_df[
        [
            "eps_meters",
            "min_samples",
            "cluster_count",
            "noise_station_count",
            "noise_ratio_percent",
            "largest_cluster_ratio_percent",
        ]
    ].copy()
    summary_df["adopted_for_final_analysis"] = False
    summary_df["non_adoption_reason"] = (
        "인천의 도심·외곽·섬 지역 밀도 차이와 eps/min_samples 민감성 때문에 "
        "최종 취약지역 지표로 사용하지 않고 탐색적 보조 분석으로만 활용"
    )
    summary_df.to_csv(DBSCAN_EXPLORATORY_SUMMARY_FILE, index=False, encoding="utf-8-sig")
    return summary_df


def count_true(df: pd.DataFrame, column: str) -> int:
    """Boolean 또는 문자열 Boolean 열에서 True 개수를 센다."""
    if column not in df.columns:
        return 0
    if df[column].dtype == "bool":
        return int(df[column].sum())
    return int(df[column].astype("string").str.strip().str.lower().isin(["true", "1", "y", "yes"]).sum())


def build_data_quality_summary(
    charging_final_df: pd.DataFrame,
    charging_2024_df: pd.DataFrame,
    station_df: pd.DataFrame,
    final_df: pd.DataFrame,
) -> pd.DataFrame:
    """최종 데이터 품질 요약 CSV를 만든다."""
    boundary_counts = charging_2024_df["boundary_match_status"].value_counts(dropna=False)
    matched_count = int(boundary_counts.get("matched", 0))
    unmatched_count = int(boundary_counts.get("unmatched", 0))
    match_rate = round(matched_count / len(charging_2024_df) * 100, 2)

    rows = [
        {
            "item": "원본 충전기 행 수",
            "value": int(len(charging_final_df)),
            "unit": "행",
            "source": str(CHARGING_FINAL_FILE),
            "note": "삭제 표시 충전기를 포함한 최종 전처리 전 충전기 단위 행",
        },
        {
            "item": "삭제 표시 충전기 수",
            "value": count_true(charging_final_df, "is_deleted"),
            "unit": "기",
            "source": str(CHARGING_FINAL_FILE),
            "note": "is_deleted=True",
        },
        {
            "item": "좌표 이상치 수",
            "value": count_true(charging_final_df, "coordinate_outlier_final"),
            "unit": "기",
            "source": str(CHARGING_FINAL_FILE),
            "note": "coordinate_outlier_final=True",
        },
        {
            "item": "최종 공간 분석 충전기 수",
            "value": int(len(charging_2024_df)),
            "unit": "기",
            "source": str(CHARGING_2024_DISTRICT_FILE),
            "note": "삭제·좌표 이상치 처리 후 2024 경계 결합 입력",
        },
        {
            "item": "2024년 경계 매칭 성공 수",
            "value": matched_count,
            "unit": "기",
            "source": str(CHARGING_2024_DISTRICT_FILE),
            "note": "boundary_match_status=matched",
        },
        {
            "item": "경계 미매칭 수",
            "value": unmatched_count,
            "unit": "기",
            "source": str(CHARGING_2024_DISTRICT_FILE),
            "note": "boundary_match_status=unmatched",
        },
        {
            "item": "경계 매칭률",
            "value": match_rate,
            "unit": "%",
            "source": str(CHARGING_2024_DISTRICT_FILE),
            "note": "matched / 2024 경계 결합 입력 행 수",
        },
        {
            "item": "충전소 수",
            "value": int(len(station_df)),
            "unit": "개소",
            "source": str(STATION_LEVEL_FILE),
            "note": "statId 기준 충전소 단위 집계",
        },
        {
            "item": "전기차 등록 대수",
            "value": int(final_df["total_ev_count"].sum()),
            "unit": "대",
            "source": str(DISTRICT_ANALYSIS_FILE),
            "note": "2026-02-19 등록 기준",
        },
        {
            "item": "일반 이용 가능 충전기 수",
            "value": int(final_df["public_charger_count"].sum()),
            "unit": "기",
            "source": str(DISTRICT_ANALYSIS_FILE),
            "note": "publicly_accessible=True 집계",
        },
        {
            "item": "이용 제한 충전기 수",
            "value": count_true(charging_2024_df, "is_limited"),
            "unit": "기",
            "source": str(CHARGING_2024_DISTRICT_FILE),
            "note": "is_limited=True",
        },
    ]
    quality_df = pd.DataFrame(rows)
    quality_df.to_csv(FINAL_DATA_QUALITY_FILE, index=False, encoding="utf-8-sig")
    return quality_df


def validate_final_assets(figure_inventory_df: pd.DataFrame, map_inventory_df: pd.DataFrame) -> None:
    """최종 그래프와 지도 파일 존재 여부를 검증한다."""
    for path in figure_inventory_df["path"].map(Path):
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"최종 그래프 PNG가 없거나 크기가 0입니다: {path}")

    invalid_maps = map_inventory_df.loc[
        ~map_inventory_df["exists"] | map_inventory_df["file_size_bytes"].eq(0)
    ]
    if not invalid_maps.empty:
        raise FileNotFoundError(
            "최종 지도 HTML 파일 검증 실패:\n"
            f"{invalid_maps.to_string(index=False)}"
        )


def top_rows_text(df: pd.DataFrame, columns: list[str], n: int = 5) -> str:
    """Markdown용 표 문자열을 만든다."""
    table_df = df[columns].head(n).copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []

    for _, row in table_df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows])


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """선택 의존성 없이 작은 DataFrame을 Markdown 표로 변환한다."""
    columns = df.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []

    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")

    return "\n".join([header, separator, *rows])


def write_final_notes(
    final_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    map_inventory_df: pd.DataFrame,
    dbscan_summary_df: pd.DataFrame,
) -> None:
    """최종 분석 노트 Markdown을 작성한다."""
    total_ev = int(final_df["total_ev_count"].sum())
    total_stations = int(final_df["station_count"].sum())
    total_chargers = int(final_df["charger_count"].sum())
    total_public = int(final_df["public_charger_count"].sum())
    total_fast = int(final_df["fast_charger_count"].sum())
    registration_date = str(final_df["registration_base_date"].iloc[0])
    boundary_date = str(final_df["boundary_base_date"].iloc[0])
    charging_date = str(final_df["charging_data_base_date"].iloc[0])
    public_low = final_df.sort_values("public_chargers_per_100_ev").head(3)
    fast_low = final_df.sort_values("fast_chargers_per_100_ev").head(3)
    charger_top = final_df.sort_values("charger_count", ascending=False).head(3)
    ev_top = final_df.sort_values("total_ev_count", ascending=False).head(3)
    quality_map = quality_df.set_index("item")["value"].to_dict()

    notes = f"""# 최종 분석 노트

## 1. 분석에 사용한 데이터

- 충전소 데이터: `{CHARGING_2024_DISTRICT_FILE}` 및 충전소 단위 집계 `{STATION_LEVEL_FILE}`
- 군·구 수요·공급 병합 결과: `{DISTRICT_ANALYSIS_FILE}`
- 최종 분석 단위: 2024년 행정구역 기준 인천 10개 군·구
- 전기차 등록 기준일: `{registration_date}`. 이 기준일은 병합 결과 메타데이터와 `ev_registration_2026_02_19_clean.csv` 파일명에서 확인했다.
- 행정구역 경계 기준일: `{boundary_date}`
- 충전소 데이터 기준: `{charging_date}`. 병합 결과 메타데이터와 충전소 상태 갱신 시점이 2026년 7월임을 기준으로 사용했다.

## 2. 전처리 결과

- 삭제 표시 충전기 수: {int(quality_map['삭제 표시 충전기 수']):,}기
- 좌표 이상치 수: {int(quality_map['좌표 이상치 수']):,}기
- 최종 공간 분석 충전기 수: {int(quality_map['최종 공간 분석 충전기 수']):,}기
- 2024년 경계 매칭 성공 수: {int(quality_map['2024년 경계 매칭 성공 수']):,}기
- 경계 미매칭 수: {int(quality_map['경계 미매칭 수']):,}기
- 경계 매칭률: {float(quality_map['경계 매칭률']):.2f}%
- statId 기준 충전소 수: {total_stations:,}개소

## 3. 군·구별 충전 인프라 분포

전체 충전기 수는 {total_chargers:,}기, 충전소 수는 {total_stations:,}개소이다.
전체 충전기 수가 많은 상위 군·구는 다음과 같다.

{top_rows_text(charger_top, ['district', 'charger_count', 'station_count', 'public_charger_count'], 3)}

## 4. 전기차 등록 대수 대비 공급 수준

등록 전기차 총합은 {total_ev:,}대이다. 이 값은 `{registration_date}` 기준이고,
충전기 수는 `{charging_date}` 기준이므로 두 자료의 시점 차이를 전제로 해석해야 한다.
전기차 등록 대수가 많은 상위 군·구는 다음과 같다.

{top_rows_text(ev_top, ['district', 'total_ev_count', 'charger_count', 'chargers_per_100_ev'], 3)}

## 5. 일반 이용 가능 충전기 분석

일반 이용 가능 충전기는 {total_public:,}기이다. 전체 충전기 중 일반 이용 가능 비율은
{total_public / total_chargers * 100:.2f}%이다. 전기차 100대당 일반 이용 가능 충전기 수가 낮은 지역은 다음과 같다.

{top_rows_text(public_low, ['district', 'public_chargers_per_100_ev', 'public_charger_count', 'total_ev_count'], 3)}

## 6. 급속충전기 분석

급속충전기는 {total_fast:,}기이다. 급속·완속 분류는 충전용량 `output >= 50kW` 기준의 프로젝트 분석용 임시 분류이다.
전기차 100대당 급속충전기 수가 낮은 지역은 다음과 같다.

{top_rows_text(fast_low, ['district', 'fast_chargers_per_100_ev', 'fast_charger_count', 'total_ev_count'], 3)}

## 7. 지도에서 확인된 공간적 특징

{dataframe_to_markdown(map_inventory_df[['map_file', 'purpose', 'interpretation_point']])}

충전소 위치 지도와 HeatMap은 도심부의 충전소 밀집, 외곽 및 도서 지역의 낮은 공간 밀도를 함께 보여준다.
군·구 단계구분도는 `public_chargers_per_100_ev`가 낮은 지역을 상대적으로 취약한 공급 후보로 확인하는 데 사용했다.

## 8. DBSCAN 탐색 결과와 미채택 이유

DBSCAN은 충전소 단위 좌표를 EPSG:5179로 변환해 실험했다. 다만 인천은 도심, 외곽, 섬 지역의 공간 밀도 차이가 커서
하나의 eps와 min_samples 조합으로 모든 지역을 안정적으로 설명하기 어렵다. 따라서 DBSCAN은 군집화에 실패한 것이 아니라,
파라미터 민감성과 지역별 밀도 차이 때문에 최종 취약지역 지표로 채택하지 않고 탐색적 보조 분석으로 남겼다.

실험 조합 수: {len(dbscan_summary_df):,}개

## 9. 데이터로 직접 확인된 사실

- 최종 분석표의 군·구 수는 10개이며 중복 군·구는 없다.
- 최종 매칭 충전기 수는 {total_chargers:,}기이다.
- 일반 이용 가능 충전기 수는 {total_public:,}기이다.
- 2024년 경계 매칭률은 {float(quality_map['경계 매칭률']):.2f}%이다.
- 전기차 등록 대수는 {registration_date} 기준 {total_ev:,}대이다.

## 10. 분석을 통해 추론한 내용

- 전기차 등록 대수와 충전기 수가 모두 많은 지역이 반드시 수요 대비 공급이 양호한 지역은 아니다.
- 일반 이용 가능 충전기 기준으로 보면 제한 충전기가 많은 지역의 체감 접근성은 전체 충전기 수보다 낮을 수 있다.
- 도서 및 외곽 지역은 충전소 간 거리가 길어질 수 있으므로 단순 군·구 합계와 별도의 공간 접근성 지표가 필요할 수 있다.

## 11. 추가 데이터가 필요한 내용

- 충전기별 실제 운영 시간, 고장·점검 이력, 요금, 주차 접근 가능 여부
- 공동주택 입주민 전용 여부를 더 정밀하게 구분할 수 있는 현장 데이터
- 2026년 동일 기준의 최신 전기차 등록 데이터와 유동 인구, 통행량, 관광 수요
- 도로망 기반 이동시간 자료

## 12. 분석의 한계

- 전기차 등록 자료와 충전소 자료의 기준 시점이 다르다.
- 급속·완속 분류는 공식 유형 코드가 아니라 충전용량 기준 임시 분류이다.
- 군·구 단위 분석은 행정구역 내부의 세부 접근성 차이를 설명하지 못한다.
- DBSCAN 결과는 탐색적 참고용이며 최종 취약지역 지표로 사용하지 않았다.
"""
    FINAL_NOTES_FILE.write_text(notes, encoding="utf-8")


def print_terminal_summary(
    final_df: pd.DataFrame,
    figure_inventory_df: pd.DataFrame,
    map_inventory_df: pd.DataFrame,
    dbscan_summary_df: pd.DataFrame,
    generated_paths: list[Path],
) -> None:
    """요구된 터미널 요약을 출력한다."""
    print("\n[최종 분석표]")
    print(final_df.to_string(index=False))

    print("\n[일반 이용 가능 공급이 낮은 상위 3개 지역]")
    print(
        final_df.sort_values("public_chargers_per_100_ev")[
            ["district", "public_chargers_per_100_ev", "public_supply_shortage_rank"]
        ]
        .head(3)
        .to_string(index=False)
    )

    print("\n[급속 공급이 낮은 상위 3개 지역]")
    print(
        final_df.sort_values("fast_chargers_per_100_ev")[
            ["district", "fast_chargers_per_100_ev", "fast_supply_shortage_rank"]
        ]
        .head(3)
        .to_string(index=False)
    )

    rank_gap_df = final_df.copy()
    rank_gap_df["charger_vs_public_shortage_rank_gap"] = (
        rank_gap_df["charger_count_rank"] - rank_gap_df["public_supply_shortage_rank"]
    ).abs()
    print("\n[전체 충전기 수 순위와 수요 대비 공급 순위 차이가 큰 지역]")
    print(
        rank_gap_df.sort_values("charger_vs_public_shortage_rank_gap", ascending=False)[
            [
                "district",
                "charger_count_rank",
                "public_supply_shortage_rank",
                "charger_vs_public_shortage_rank_gap",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    print("\n[최종 그래프 목록]")
    print(
        figure_inventory_df.loc[
            figure_inventory_df["selected_for_report"],
            ["figure_file", "purpose", "file_size_bytes"],
        ].to_string(index=False)
    )

    print("\n[최종 지도 목록]")
    print(map_inventory_df[["map_file", "purpose", "file_size_bytes"]].to_string(index=False))

    print("\n[DBSCAN 미채택 요약]")
    print(
        "DBSCAN은 지역별 밀도 차이와 파라미터 민감성 때문에 최종 취약지역 지표로 "
        "사용하지 않고 탐색적 보조 분석으로 남겼습니다."
    )
    print(f"실험 조합 수: {len(dbscan_summary_df):,}")

    print("\n[생성 파일 경로]")
    for path in generated_paths:
        print(path)


def main() -> None:
    ensure_dirs()
    district_df = read_csv(DISTRICT_ANALYSIS_FILE)
    charging_final_df = read_csv(CHARGING_FINAL_FILE)
    charging_2024_df = read_csv(CHARGING_2024_DISTRICT_FILE)
    station_df = read_csv(STATION_LEVEL_FILE)
    dbscan_parameter_df = read_csv(DBSCAN_PARAMETER_FILE)

    final_df = build_final_district_summary(district_df)
    figure_inventory_df = create_final_figures(final_df)
    map_inventory_df = build_map_inventory()
    dbscan_summary_df = build_dbscan_exploratory_summary(dbscan_parameter_df)
    quality_df = build_data_quality_summary(
        charging_final_df,
        charging_2024_df,
        station_df,
        final_df,
    )
    validate_final_assets(figure_inventory_df, map_inventory_df)
    write_final_notes(final_df, quality_df, map_inventory_df, dbscan_summary_df)

    generated_paths = [
        FINAL_DISTRICT_SUMMARY_FILE,
        FINAL_FIGURE_INVENTORY_FILE,
        *FINAL_FIGURES.values(),
        FINAL_MAP_INVENTORY_FILE,
        DBSCAN_EXPLORATORY_SUMMARY_FILE,
        FINAL_DATA_QUALITY_FILE,
        FINAL_NOTES_FILE,
    ]

    print_terminal_summary(
        final_df,
        figure_inventory_df,
        map_inventory_df,
        dbscan_summary_df,
        generated_paths,
    )


if __name__ == "__main__":
    main()
