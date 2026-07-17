from __future__ import annotations

from html import escape
from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster


PROJECT_DIR = Path(__file__).resolve().parent.parent

STATION_INPUT_FILE = (
    PROJECT_DIR / "data" / "processed" / "charging_station_level_spatial.csv"
)
DISTRICT_SUPPLY_DEMAND_FILE = (
    PROJECT_DIR / "outputs" / "tables" / "district_ev_charging_supply_demand_summary.csv"
)
BOUNDARY_SHP_FILE = (
    PROJECT_DIR / "data" / "raw" / "boundary_2024_incheon" / "DA_SIG_202406.shp"
)
BOUNDARY_GEOJSON_FILE = (
    PROJECT_DIR / "data" / "processed" / "charging_stations_incheon_2024_district.geojson"
)

OUTPUT_MAP_DIR = PROJECT_DIR / "outputs" / "maps"
MARKER_CLUSTER_MAP_FILE = OUTPUT_MAP_DIR / "station_marker_cluster_map.html"
CAPACITY_CIRCLE_MAP_FILE = OUTPUT_MAP_DIR / "station_capacity_circle_map.html"
PUBLIC_HEATMAP_FILE = OUTPUT_MAP_DIR / "public_charger_heatmap.html"
DISTRICT_CHOROPLETH_MAP_FILE = OUTPUT_MAP_DIR / "district_supply_choropleth_map.html"

INCHEON_CENTER = [37.4563, 126.7052]
DEFAULT_ZOOM = 10

STATION_REQUIRED_COLUMNS = [
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
]

DISTRICT_REQUIRED_COLUMNS = [
    "district_2024",
    "public_chargers_per_100_ev",
    "registration_base_date",
    "boundary_base_date",
    "charging_data_base_date",
]

BOUNDARY_REQUIRED_COLUMNS = [
    "district_2024",
    "geometry",
]

# 옹진군 도서 지역을 포함하도록 인천 주변을 넓게 잡은 좌표 검증 범위이다.
VALID_LAT_RANGE = (36.5, 38.5)
VALID_LNG_RANGE = (124.0, 127.8)


def ensure_output_dir() -> None:
    """지도 저장 폴더를 생성한다."""
    OUTPUT_MAP_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    """입력 파일 존재 여부를 확인한다."""
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")


def require_columns(df: pd.DataFrame, required_columns: list[str], name: str) -> None:
    """실제 열 이름 기준으로 필수 열을 검증한다."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"{name} 필수 열이 없습니다: {missing_columns}")


def load_station_data() -> pd.DataFrame:
    """충전소 단위 공간 CSV를 읽고 좌표를 검증한다."""
    require_file(STATION_INPUT_FILE)
    station_df = pd.read_csv(STATION_INPUT_FILE, encoding="utf-8-sig", low_memory=False)

    print(f"\n[충전소 단위 입력] {STATION_INPUT_FILE}")
    print(f"행 수: {len(station_df):,}")
    print(f"열 이름: {station_df.columns.tolist()}")

    require_columns(station_df, STATION_REQUIRED_COLUMNS, "충전소 단위 데이터")

    station_df["latitude"] = pd.to_numeric(station_df["latitude"], errors="coerce")
    station_df["longitude"] = pd.to_numeric(station_df["longitude"], errors="coerce")

    missing_coord_mask = station_df["latitude"].isna() | station_df["longitude"].isna()
    if missing_coord_mask.any():
        missing_rows = station_df.loc[
            missing_coord_mask,
            ["station_id", "station_name", "latitude", "longitude"],
        ]
        raise ValueError(
            "충전소 좌표 결측이 있습니다:\n"
            f"{missing_rows.to_string(index=False)}"
        )

    invalid_coord_mask = (
        ~station_df["latitude"].between(*VALID_LAT_RANGE)
        | ~station_df["longitude"].between(*VALID_LNG_RANGE)
    )
    if invalid_coord_mask.any():
        invalid_rows = station_df.loc[
            invalid_coord_mask,
            ["station_id", "station_name", "latitude", "longitude"],
        ]
        raise ValueError(
            "인천 분석 범위를 벗어난 좌표가 있습니다:\n"
            f"{invalid_rows.to_string(index=False)}"
        )

    numeric_columns = [
        "charger_count",
        "fast_charger_count",
        "slow_charger_count",
        "unknown_speed_count",
        "public_charger_count",
        "limited_charger_count",
        "public_charger_ratio_percent",
        "fast_charger_ratio_percent",
    ]
    for column in numeric_columns:
        station_df[column] = pd.to_numeric(station_df[column], errors="coerce")

    if station_df[numeric_columns].isna().any().any():
        missing_counts = station_df[numeric_columns].isna().sum()
        raise ValueError(
            "충전소 수치 열에 결측 또는 변환 불가 값이 있습니다:\n"
            f"{missing_counts[missing_counts > 0].to_string()}"
        )

    if (station_df[numeric_columns] < 0).any().any():
        negative_counts = (station_df[numeric_columns] < 0).sum()
        raise ValueError(
            "충전소 수치 열에 음수가 있습니다:\n"
            f"{negative_counts[negative_counts > 0].to_string()}"
        )

    return station_df


def load_district_supply_demand() -> pd.DataFrame:
    """군·구별 수요·공급 병합 결과를 읽는다."""
    require_file(DISTRICT_SUPPLY_DEMAND_FILE)
    district_df = pd.read_csv(
        DISTRICT_SUPPLY_DEMAND_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    print(f"\n[군·구 수요·공급 입력] {DISTRICT_SUPPLY_DEMAND_FILE}")
    print(f"행 수: {len(district_df):,}")
    print(f"열 이름: {district_df.columns.tolist()}")

    require_columns(district_df, DISTRICT_REQUIRED_COLUMNS, "군·구 수요·공급 데이터")
    district_df["district_2024"] = district_df["district_2024"].astype("string").str.strip()
    district_df["public_chargers_per_100_ev"] = pd.to_numeric(
        district_df["public_chargers_per_100_ev"],
        errors="coerce",
    )

    if district_df["public_chargers_per_100_ev"].isna().any():
        raise ValueError("public_chargers_per_100_ev에 결측 또는 변환 불가 값이 있습니다.")

    if district_df["district_2024"].duplicated().any():
        duplicated = district_df.loc[
            district_df["district_2024"].duplicated(keep=False),
            "district_2024",
        ].tolist()
        raise ValueError(f"군·구 수요·공급 데이터에 중복 군·구가 있습니다: {duplicated}")

    return district_df


def load_boundary() -> gpd.GeoDataFrame:
    """2024년 인천 10개 군·구 경계를 읽어 EPSG:4326으로 변환한다."""
    if BOUNDARY_SHP_FILE.exists():
        boundary_gdf = gpd.read_file(BOUNDARY_SHP_FILE, encoding="cp949")
        print(f"\n[경계 입력] {BOUNDARY_SHP_FILE}")
        print(f"행 수: {len(boundary_gdf):,}")
        print(f"열 이름: {boundary_gdf.columns.tolist()}")

        if "SIGUNGU_NM" not in boundary_gdf.columns:
            raise KeyError("경계 SHP에 SIGUNGU_NM 열이 없습니다.")

        boundary_gdf = boundary_gdf.rename(columns={"SIGUNGU_NM": "district_2024"})
    elif BOUNDARY_GEOJSON_FILE.exists():
        boundary_gdf = gpd.read_file(BOUNDARY_GEOJSON_FILE)
        print(f"\n[경계 입력] {BOUNDARY_GEOJSON_FILE}")
        print(f"행 수: {len(boundary_gdf):,}")
        print(f"열 이름: {boundary_gdf.columns.tolist()}")

        if "district_2024" not in boundary_gdf.columns:
            raise KeyError("경계 GeoJSON에 district_2024 열이 없습니다.")

        boundary_gdf = boundary_gdf[["district_2024", "geometry"]].drop_duplicates(
            subset=["district_2024"]
        )
    else:
        raise FileNotFoundError("2024년 인천 군·구 경계 SHP 또는 GeoJSON을 찾지 못했습니다.")

    boundary_gdf["district_2024"] = boundary_gdf["district_2024"].astype("string").str.strip()
    require_columns(boundary_gdf, BOUNDARY_REQUIRED_COLUMNS, "경계 데이터")

    if boundary_gdf.crs is None:
        raise ValueError("경계 데이터 CRS가 없습니다.")

    boundary_gdf = boundary_gdf.to_crs("EPSG:4326")
    boundary_gdf = boundary_gdf[["district_2024", "geometry"]].copy()

    if len(boundary_gdf) != 10:
        raise ValueError(f"2024년 인천 군·구 경계가 10개가 아닙니다: {len(boundary_gdf):,}")

    if boundary_gdf.geometry.isna().any() or boundary_gdf.geometry.is_empty.any():
        raise ValueError("경계 데이터에 빈 geometry가 있습니다.")

    return boundary_gdf


def create_base_map(title: str, subtitle: str | None = None) -> folium.Map:
    """공통 Folium 기본 지도를 만든다."""
    fmap = folium.Map(
        location=INCHEON_CENTER,
        zoom_start=DEFAULT_ZOOM,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    subtitle_html = f"<div style='font-size:12px;color:#555;'>{escape(subtitle)}</div>" if subtitle else ""
    title_html = f"""
    <div style="
        position: fixed;
        top: 12px;
        left: 50px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.92);
        padding: 8px 12px;
        border: 1px solid #999;
        border-radius: 4px;
        font-family: Arial, sans-serif;
    ">
        <div style="font-size:16px;font-weight:700;">{escape(title)}</div>
        {subtitle_html}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))

    return fmap


def add_boundary_layer(fmap: folium.Map, boundary_gdf: gpd.GeoDataFrame) -> None:
    """행정구역 경계 레이어를 추가한다."""
    folium.GeoJson(
        boundary_gdf,
        name="2024년 인천 군·구 경계",
        style_function=lambda _: {
            "fillColor": "#ffffff",
            "color": "#333333",
            "weight": 1.2,
            "fillOpacity": 0.03,
        },
        tooltip=folium.GeoJsonTooltip(fields=["district_2024"], aliases=["군·구"]),
    ).add_to(fmap)


def build_station_popup(row: pd.Series) -> folium.Popup:
    """충전소 마커 팝업을 간결하게 생성한다."""
    html = f"""
    <div style="font-family:Arial, sans-serif; font-size:12px; line-height:1.45;">
      <b>{escape(str(row.station_name))}</b><br>
      주소: {escape(str(row.representative_address))}<br>
      군·구: {escape(str(row.district_2024))}<br>
      전체: {int(row.charger_count):,}기 |
      급속: {int(row.fast_charger_count):,} |
      완속: {int(row.slow_charger_count):,}<br>
      일반 이용 가능: {int(row.public_charger_count):,} |
      제한: {int(row.limited_charger_count):,}
    </div>
    """

    return folium.Popup(html, max_width=320)


def create_marker_cluster_map(
    station_df: pd.DataFrame,
    boundary_gdf: gpd.GeoDataFrame,
) -> Path:
    """충전소 1곳당 Marker 1개를 MarkerCluster로 표시한다."""
    fmap = create_base_map("인천 전기차 충전소 위치 MarkerCluster")
    add_boundary_layer(fmap, boundary_gdf)

    cluster = MarkerCluster(name="충전소 위치").add_to(fmap)

    for row in station_df.itertuples(index=False):
        tooltip = f"{row.station_name} ({int(row.charger_count):,}기)"
        folium.Marker(
            location=[row.latitude, row.longitude],
            tooltip=tooltip,
            popup=build_station_popup(row),
        ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(MARKER_CLUSTER_MAP_FILE)

    return MARKER_CLUSTER_MAP_FILE


def get_fast_ratio_color(fast_ratio: float) -> str:
    """급속충전기 비율 구간별 색상을 반환한다."""
    if fast_ratio < 5:
        return "#2c7bb6"
    if fast_ratio < 10:
        return "#abd9e9"
    if fast_ratio < 20:
        return "#ffffbf"
    if fast_ratio < 40:
        return "#fdae61"
    return "#d7191c"


def add_fast_ratio_legend(fmap: folium.Map) -> None:
    """급속충전기 비율 범례를 추가한다."""
    legend_html = """
    <div style="
        position: fixed;
        bottom: 35px;
        left: 50px;
        z-index: 9999;
        background: white;
        padding: 10px 12px;
        border: 1px solid #999;
        border-radius: 4px;
        font-family: Arial, sans-serif;
        font-size: 12px;
    ">
      <b>급속충전기 비율</b><br>
      <span style="color:#2c7bb6;">●</span> 5% 미만<br>
      <span style="color:#abd9e9;">●</span> 5-10%<br>
      <span style="color:#b8a800;">●</span> 10-20%<br>
      <span style="color:#fdae61;">●</span> 20-40%<br>
      <span style="color:#d7191c;">●</span> 40% 이상
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))


def create_capacity_circle_map(
    station_df: pd.DataFrame,
    boundary_gdf: gpd.GeoDataFrame,
) -> Path:
    """충전기 수를 sqrt 스케일 반지름으로 표현한다."""
    fmap = create_base_map("인천 충전소별 충전 규모와 급속충전 비율")
    add_boundary_layer(fmap, boundary_gdf)

    circle_layer = folium.FeatureGroup(name="충전소 규모 CircleMarker").add_to(fmap)

    for row in station_df.itertuples(index=False):
        radius = min(18, 3 + np.sqrt(float(row.charger_count)) * 1.5)
        color = get_fast_ratio_color(float(row.fast_charger_ratio_percent))
        tooltip = (
            f"{row.station_name} | 전체 {int(row.charger_count):,}기 | "
            f"급속 {float(row.fast_charger_ratio_percent):.1f}%"
        )
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            weight=1,
            tooltip=tooltip,
            popup=build_station_popup(row),
        ).add_to(circle_layer)

    add_fast_ratio_legend(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(CAPACITY_CIRCLE_MAP_FILE)

    return CAPACITY_CIRCLE_MAP_FILE


def create_public_heatmap(
    station_df: pd.DataFrame,
    boundary_gdf: gpd.GeoDataFrame,
) -> Path:
    """일반 이용 가능 충전기 수를 가중치로 한 HeatMap을 생성한다."""
    fmap = create_base_map(
        "인천 일반 이용 가능 충전기 HeatMap",
        "가중치: public_charger_count",
    )
    add_boundary_layer(fmap, boundary_gdf)

    heat_df = station_df.loc[station_df["public_charger_count"] > 0].copy()

    # 실제 접근성에 가까운 분포를 보기 위해 전체 충전기 수가 아니라
    # 시민이 일반적으로 이용 가능한 충전기 수(public_charger_count)를 가중치로 사용한다.
    heat_data = heat_df[["latitude", "longitude", "public_charger_count"]].values.tolist()

    HeatMap(
        heat_data,
        name="일반 이용 가능 충전기 밀도",
        radius=14,
        blur=18,
        min_opacity=0.25,
        max_zoom=13,
    ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(PUBLIC_HEATMAP_FILE)

    return PUBLIC_HEATMAP_FILE


def build_choropleth_boundary(
    boundary_gdf: gpd.GeoDataFrame,
    district_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """경계와 수요·공급 지표를 결합한다."""
    district_columns = [
        "district_2024",
        "public_chargers_per_100_ev",
        "public_charger_count",
        "total_ev_count",
        "registration_base_date",
        "boundary_base_date",
        "charging_data_base_date",
    ]
    choropleth_gdf = boundary_gdf.merge(
        district_df[district_columns],
        on="district_2024",
        how="left",
        validate="one_to_one",
    )

    if choropleth_gdf["public_chargers_per_100_ev"].isna().any():
        missing = choropleth_gdf.loc[
            choropleth_gdf["public_chargers_per_100_ev"].isna(),
            "district_2024",
        ].tolist()
        raise ValueError(f"단계구분도 지표가 없는 군·구가 있습니다: {missing}")

    return choropleth_gdf


def create_step_colormap(values: pd.Series) -> cm.StepColormap:
    """낮은 값이 취약하게 보이도록 red-to-green 단계 색상을 만든다."""
    quantiles = np.quantile(values, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bins = sorted(set(round(float(value), 2) for value in quantiles))

    if len(bins) < 3:
        bins = [
            float(values.min()),
            float(values.mean()),
            float(values.max()),
        ]

    return cm.StepColormap(
        colors=["#b2182b", "#ef8a62", "#fddbc7", "#d9f0d3", "#1a9850"],
        index=bins,
        vmin=float(values.min()),
        vmax=float(values.max()),
        caption=(
            "2026년 2월 19일 등록 전기차 100대당 2026년 7월 "
            "일반 이용 가능 충전기 수 - 낮을수록 취약"
        ),
    )


def create_district_choropleth_map(
    boundary_gdf: gpd.GeoDataFrame,
    district_df: pd.DataFrame,
) -> Path:
    """군·구별 public_chargers_per_100_ev 단계구분도를 생성한다."""
    choropleth_gdf = build_choropleth_boundary(boundary_gdf, district_df)
    values = choropleth_gdf["public_chargers_per_100_ev"]
    colormap = create_step_colormap(values)

    base_dates = district_df.iloc[0]
    subtitle = (
        f"등록 {base_dates['registration_base_date']} | "
        f"경계 {base_dates['boundary_base_date']} | "
        f"충전소 {base_dates['charging_data_base_date']} | "
        "값이 낮을수록 일반 이용 가능 인프라 취약"
    )
    fmap = create_base_map(
        "군·구별 일반 이용 가능 충전기 공급 수준",
        subtitle,
    )

    def style_function(feature: dict) -> dict:
        value = feature["properties"]["public_chargers_per_100_ev"]
        return {
            "fillColor": colormap(value),
            "color": "#333333",
            "weight": 1.2,
            "fillOpacity": 0.72,
        }

    folium.GeoJson(
        choropleth_gdf,
        name="일반 이용 가능 충전기 공급 취약도",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "district_2024",
                "public_chargers_per_100_ev",
                "public_charger_count",
                "total_ev_count",
            ],
            aliases=[
                "군·구",
                "일반 이용 가능 충전기 / 전기차 100대",
                "일반 이용 가능 충전기 수",
                "등록 전기차 수",
            ],
            localize=True,
        ),
    ).add_to(fmap)

    add_boundary_layer(fmap, boundary_gdf)
    colormap.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(DISTRICT_CHOROPLETH_MAP_FILE)

    return DISTRICT_CHOROPLETH_MAP_FILE


def print_output_paths(paths: list[Path]) -> None:
    """생성된 HTML 경로와 파일 크기를 출력한다."""
    print("\n[생성 HTML]")
    for path in paths:
        size_kb = path.stat().st_size / 1024
        print(f"{path} ({size_kb:,.1f} KB)")


def main() -> None:
    ensure_output_dir()

    station_df = load_station_data()
    district_df = load_district_supply_demand()
    boundary_gdf = load_boundary()

    output_paths = [
        create_marker_cluster_map(station_df, boundary_gdf),
        create_capacity_circle_map(station_df, boundary_gdf),
        create_public_heatmap(station_df, boundary_gdf),
        create_district_choropleth_map(boundary_gdf, district_df),
    ]

    print_output_paths(output_paths)


if __name__ == "__main__":
    main()
