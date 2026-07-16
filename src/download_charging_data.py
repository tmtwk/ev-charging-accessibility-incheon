from pathlib import Path
import os
import time
from xml.etree import ElementTree

import pandas as pd
import requests
from dotenv import load_dotenv


# --------------------------------------------------
# 1. 프로젝트 경로 설정
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV_PATH = RAW_DATA_DIR / "charging_stations_incheon_raw.csv"
OUTPUT_XML_PATH = RAW_DATA_DIR / "charging_stations_incheon_first_response.xml"


# --------------------------------------------------
# 2. 환경변수에서 인증키 불러오기
# --------------------------------------------------

load_dotenv(PROJECT_DIR / ".env")

SERVICE_KEY = os.getenv("DATA_GO_KR_SERVICE_KEY")

if not SERVICE_KEY:
    raise ValueError(
        ".env 파일에서 DATA_GO_KR_SERVICE_KEY를 찾을 수 없습니다."
    )


# --------------------------------------------------
# 3. API 기본 설정
# --------------------------------------------------

ENDPOINT = (
    "https://apis.data.go.kr/"
    "B552584/EvCharger/getChargerInfo"
)

INCHON_ZCODE = "28"

# 처음에는 1,000행씩 요청
ROWS_PER_PAGE = 1000

REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 5


# --------------------------------------------------
# 4. XML item을 딕셔너리로 변환
# --------------------------------------------------

def parse_items(xml_content: bytes) -> tuple[list[dict], int]:
    """
    XML 응답에서 item 목록과 전체 데이터 수를 추출한다.
    """

    root = ElementTree.fromstring(xml_content)

    result_code = root.findtext(".//resultCode")
    result_message = root.findtext(".//resultMsg")

    if result_code != "00":
        raise RuntimeError(
            f"API 요청 실패: {result_code} / {result_message}"
        )

    total_count_text = root.findtext(".//totalCount", default="0")
    total_count = int(total_count_text)

    rows = []

    for item in root.findall(".//item"):
        row = {}

        for child in item:
            row[child.tag] = child.text

        rows.append(row)

    return rows, total_count


# --------------------------------------------------
# 5. 특정 페이지 요청
# --------------------------------------------------

def request_page(page_number: int) -> tuple[list[dict], int, bytes]:
    """
    인천광역시 충전기 정보의 특정 페이지를 요청한다.
    """

    params = {
        "serviceKey": SERVICE_KEY,
        "pageNo": page_number,
        "numOfRows": ROWS_PER_PAGE,
        "zcode": INCHON_ZCODE,
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                ENDPOINT,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            rows, total_count = parse_items(response.content)

            return rows, total_count, response.content

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ) as error:
            last_error = error

            if attempt == MAX_RETRIES:
                break

            print(
                f"{page_number}페이지 요청 실패 "
                f"({attempt}/{MAX_RETRIES}): {error}"
            )
            print(f"{RETRY_WAIT_SECONDS}초 후 재시도합니다.")
            time.sleep(RETRY_WAIT_SECONDS)

    raise RuntimeError(
        f"{page_number}페이지 요청을 {MAX_RETRIES}회 시도했지만 실패했습니다."
    ) from last_error


# --------------------------------------------------
# 6. 전체 페이지 수집
# --------------------------------------------------

def download_all_pages() -> pd.DataFrame:
    """
    인천광역시 전체 충전기 데이터를 페이지별로 수집한다.
    """

    print("1페이지 요청 중...")

    first_rows, total_count, first_xml = request_page(1)

    # 첫 응답 원본 XML 저장
    OUTPUT_XML_PATH.write_bytes(first_xml)

    print(f"전체 충전기 데이터 수: {total_count:,}행")
    print(f"1페이지 수집 행 수: {len(first_rows):,}행")

    total_pages = (
        total_count + ROWS_PER_PAGE - 1
    ) // ROWS_PER_PAGE

    print(f"전체 페이지 수: {total_pages:,}페이지")

    all_rows = first_rows

    for page_number in range(2, total_pages + 1):
        print(
            f"{page_number}/{total_pages}페이지 수집 중..."
        )

        rows, _, _ = request_page(page_number)
        all_rows.extend(rows)

        # 서버에 지나치게 빠른 요청을 보내지 않도록 잠시 대기
        time.sleep(0.2)

    dataframe = pd.DataFrame(all_rows)

    return dataframe


# --------------------------------------------------
# 7. 실행 및 저장
# --------------------------------------------------

def main() -> None:
    charging_df = download_all_pages()

    charging_df.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n수집 완료")
    print(f"저장 경로: {OUTPUT_CSV_PATH}")
    print(f"행 수: {charging_df.shape[0]:,}")
    print(f"열 수: {charging_df.shape[1]:,}")

    print("\n열 이름:")
    print(charging_df.columns.tolist())

    print("\n앞부분 5행:")
    print(charging_df.head().to_string())


if __name__ == "__main__":
    main()
