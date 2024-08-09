import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
import pickle
import os
import time
from geopy.geocoders import Nominatim
from io import StringIO
import feedparser
from datetime import datetime
import pytz
import re
import streamlit as st
from google.cloud import translate_v2 as translate  # pip install google-cloud-translate==2.0.1
from google.oauth2 import service_account

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)

GEOLOC_CACHE_FILE = 'geolocation_cache.pkl'
FETCH_INTERVAL = 10  # 60초마다 뉴스 업데이트

# 스레드 풀 실행자 초기화
# executor = concurrent.futures.ThreadPoolExecutor()


# Set verbose if needed
# globals.set_debug(True)


# # # # # # # # # # # #
# 세션 캐시 처리
# # # # # # # # # # # #


if "geolocations" not in st.session_state:
    if os.path.exists(GEOLOC_CACHE_FILE):
        # 캐시 파일이 있으면 불러온다.
        with open(GEOLOC_CACHE_FILE, 'rb') as f:
            st.session_state.geolocations = pickle.load(f)
            logging.info('캐시 파일 로딩 완료')
            logging.info(st.session_state.geolocations)
    else:
        # 캐시 파일이 없으면
        st.session_state.geolocations = dict()  # 지역별 위경도 dict를 모아둔 dict
        logging.info('캐시 파일 없음.')

if "news_list" not in st.session_state:
    st.session_state.news_list = []


# # # # # # # # # #
# 구글 뉴스 가져오기 #
# # # # # # # # # #


def get_google_outage_news(keyword_):
    query = keyword_
    if and_keyword:
        query += ' ' + and_keyword[0]

    url = f"https://news.google.com/rss/search?q={query}+when:{search_hour}h"
    url += f'&hl=en-US&gl=US&ceid=US:en'
    url = url.replace(' ', '%20')

    title_list = []
    source_list = []
    pubtime_list = []
    link_list = []

    try:
        res = requests.get(url)  # , verify=False)
        logging.info('원본 링크: ' + url)

        if res.status_code == 200:
            datas = feedparser.parse(res.text).entries
            for data in datas:
                title = data.title
                logging.info('구글뉴스제목(원본): ' + title)

                minus_index = title.rindex(' - ')
                title = title[:minus_index].strip()

                # 기사 제목에 검색 키워드가 없으면 넘긴다.
                if keyword_.lower() not in title.lower():
                    continue

                title_list.append(title)
                source_list.append(data.source.title)
                link_list.append(data.link)

                pubtime = datetime.strptime(data.published, "%a, %d %b %Y %H:%M:%S %Z")
                # GMT+9 (Asia/Seoul)으로 변경
                gmt_plus_9 = pytz.FixedOffset(540)  # 9 hours * 60 minutes = 540 minutes
                pubtime = pubtime.replace(tzinfo=pytz.utc).astimezone(gmt_plus_9)

                pubtime_str = pubtime.strftime('%Y-%m-%d %H:%M:%S')
                pubtime_list.append(pubtime_str)

        else:
            logging.error("Google 뉴스 수집 실패! Error Code: " + str(res.status_code))
            logging.error(str(res))
            return None

    except Exception as e:
        logging.error(e)
        logging.error("Google 뉴스 RSS 피드 조회 오류 발생!")
        return None

    # 결과를 dict 형태로 저장
    result = {'제목': title_list, '언론사': source_list, '발행시간': pubtime_list, '링크': link_list}

    df = pd.DataFrame(result)
    return df


def display_news_df(ndf, keyword_):
    # st.divider()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if ndf is None or len(ndf) == 0:
        st.write(f'✅ 검색된 뉴스 없습니다. ({current_time})')
        return

    # st.write('뉴스 검색 결과')

    disp_cnt = 0
    for i, row in ndf.iterrows():
        # 이미 출력했던 뉴스라면 스킵한다.
        if row['제목'] in st.session_state.news_list:
            logging.info('뉴스 스킵!!! - ' + row['제목'])
            continue

        # 출력한 뉴스 리스트에 추가한다.
        st.session_state.news_list.append(row['제목'])
        disp_cnt += 1

        # title = row['제목'].replace(keyword_, f':yellow-background[{keyword_}]')
        # logging.info('keyword: ' + keyword_)
        # logging.info('before: ' + row['제목'])
        title = re.sub(keyword_, f':blue-background[{keyword_}]', row['제목'], flags=re.IGNORECASE)
        if and_keyword:
            title = re.sub(and_keyword[0], f':blue-background[{and_keyword[0]}]', title, flags=re.IGNORECASE)
        # logging.info('after : ' + title)

        # 제목 번역
        korean_title = translate_eng_to_kor(row['제목'])

        with st.container(border=True):
            st.markdown(f'**{title}**')
            st.caption(f'{korean_title}')
            st.markdown(f'- {row["언론사"]}, {row["발행시간"]} <a href="{row["링크"]}" target="_blank">📝</a>',
                        unsafe_allow_html=True)
        # st.write(' - 언론사: ' + row['언론사'] + '  - 발행시각: ' + row['발행시간'])

    if disp_cnt > 0:
        st.write(f'✅ 뉴스 표시 완료 ({current_time})')
    else:
        st.write(f'✅ 신규 뉴스 없습니다. ({current_time})')


def fetch_news(keyword_, infinite_loop=False):
    with st.spinner('뉴스 검색중...'):
        news_df_ = get_google_outage_news(keyword_)
        # st.write(news_df_)
        display_news_df(news_df_, keyword_)

    while infinite_loop:
        time.sleep(search_interval_min * 60)
        with st.spinner('뉴스 검색중...'):
            news_df_ = get_google_outage_news(keyword_)
            # st.write(news_df_)
            display_news_df(news_df_, keyword_)


def translate_eng_to_kor(text):
    trans_key_path = './gcp/coastal-bloom-310611-ad014e3bd1be.json'
    credential_trans = service_account.Credentials.from_service_account_file(trans_key_path)
    translate_client = translate.Client(credentials=credential_trans)

    result = translate_client.translate(text, target_language='ko')
    # print(names)
    return result['translatedText'].replace('&amp;', '&')


# # # # # # # # # # # # # # #
# down 차트, 지도 링크 받아오기
# # # # # # # # # # # # # # #


def get_service_chart_mapdf(service_name):
    # url = "https://istheservicedown.com/problems/disney-plus"
    url = "https://istheservicedown.com/problems/" + service_name

    req_ = requests.Session()
    response_ = req_.get(url,
                         headers={'User-Agent': 'Popular browser\'s user-agent', })

    soup_ = BeautifulSoup(response_.content, 'html.parser')

    # 상태
    status_ = soup_.find('p').text

    # 차트 주소
    chart_url_ = soup_.find(id="chart-img")['src']

    # map 주소
    sub_url = soup_.find(title='Live Outage Map')['href']
    if sub_url:
        map_url_ = 'https://istheservicedown.com' + sub_url

        response_ = req_.get(map_url_,
                             headers={'User-Agent': 'Popular browser\'s user-agent', })

        soup_ = BeautifulSoup(response_.content, 'html.parser')
        table_html = soup_.find('table', class_="table table-striped table-condensed")  # , id_='status-table')
        map_df_ = pd.read_html(StringIO(str(table_html)))[0]
    else:
        map_df_ = None

    return status_, chart_url_, map_df_


# # # # # # # # # #
# 위경도 받아오기
# # # # # # # # # #


def save_cache(loc, lat, lon):
    st.session_state.geolocations[loc] = {'lat': lat, 'lon': lon}
    # 새로운 위경도 정보를 캐시 파일에 저장
    with open(GEOLOC_CACHE_FILE, 'wb') as f_:
        pickle.dump(st.session_state.geolocations, f_)
        logging.info('캐시 파일 업데이트 완료')


def load_cache(loc):
    return st.session_state.geolocations.get(loc)


def get_geo_location(map_df_):
    geolocator = Nominatim(user_agent="jason")
    map_df_['lat'] = None
    map_df_['lon'] = None
    map_df_['color'] = '#ff000077'  # 빨강, 살짝 투명.

    for i, row in map_df_.iterrows():
        # 세션 캐시를 먼저 살펴본다.
        cache = load_cache(row['Location'])
        if cache:
            logging.info('cache hit! - ' + row['Location'])
            map_df_.loc[i, 'lat'] = cache['lat']
            map_df_.loc[i, 'lon'] = cache['lon']
            continue

        # 세션 캐시에 없으면 위경도 api를 써서 불러온다.
        geo = geolocator.geocode(row['Location'])

        if geo:
            map_df_.loc[i, 'lat'] = geo.latitude
            map_df_.loc[i, 'lon'] = geo.longitude
            save_cache(row['Location'], geo.latitude, geo.longitude)
        else:
            # retry
            geo = geolocator.geocode(row['Location'].split(',')[0])

            if geo:
                map_df_.loc[i, 'lat'] = geo.latitude
                map_df_.loc[i, 'lon'] = geo.longitude
                save_cache(row['Location'], geo.latitude, geo.longitude)
            else:
                # retry까지 실패할 경우.
                logging.error('Geo ERROR!!! :' + str(geo))

        time.sleep(0.2)
    return map_df_


def get_multiple(values_sr):
    max_report = values_sr.max()
    multiple_ = int(500000 / max_report)
    logging.info(f'{max_report=} {multiple_=}')
    return multiple_


# # # # # # # # # #
# 회사 목록 받아오기
# # # # # # # # # #


req = requests.Session()
response = req.get('https://istheservicedown.com/companies',
                   headers={'User-Agent': 'Popular browser\'s user-agent'})

soup = BeautifulSoup(response.content, 'html.parser')

companies_html_list = soup.find_all('a', class_='b-lazy-bg')
companies_list = []

for company in companies_html_list:
    code = company['href'].split('/')[-1]
    name = company.h3.text
    logging.debug(code + ' (' + name + ')')
    companies_list.append((code, name))

logging.info('Total companies count:' + str(len(companies_list)))

# # # # # # # # # #
# 웹 페이지 구성
# # # # # # # # # #


st.set_page_config(layout="wide")
# st.title('뉴스 검색 봇')


# 사이드바 구성
service_code_name = st.sidebar.selectbox(
    "검색을 원하는 서비스는?",
    [a + '/' + b for a, b in companies_list],
    index=None,
    placeholder="서비스 이름 선택...",
)

another_service = st.sidebar.text_input("또는 서비스명 입력", )

search_hour = st.sidebar.number_input('최근 몇시간의 뉴스를 검색할까요?', value=1, format='%d')

and_keyword = st.sidebar.multiselect("뉴스 검색 추가 키워드", options=['outage', 'blackout', 'failure'], default=['outage'])

search_interval_min = st.sidebar.number_input('새로고침 주기는 몇 분?', value=1, format='%d')

if os.environ.get("OPENAI_API_KEY"):
    st.sidebar.text_input('OpenAI API Key', value='OS 환경변수에 저장된 Key 사용', disabled=True)
else:
    os.environ["OPENAI_API_KEY"] = st.sidebar.text_input('OpenAI API Key',
                                                         placeholder='Input your ChatGPT API key here.')

st.sidebar.divider()
st.sidebar.write('❓ 참고사이트: https://istheservicedown.com/')


# 서비스 선택시 처리
if service_code_name and not another_service:
    # 본문 화면 구성
    selected_code = service_code_name.split('/')[0]
    selected_name = service_code_name.split('/')[1]

    col1, col2 = st.columns(2)

    # 빈 공간을 생성하여 나중에 내용을 업데이트할 준비
    col1_placeholder = col1.empty()
    col2_placeholder = col2.empty()

    # 이 아래로는 수시로 업데이트 함.
    while True:

        with st.spinner('서비스 상태 조회중...'):
            status, chart_url, map_df = get_service_chart_mapdf(selected_code)

            # 상태
            if 'No problem' in status:
                color = 'green'
            elif status == 'Some problems detected':
                color = 'orange'
                st.toast(f'**{selected_name}** 서비스 문제 발생!', icon="🚨")
            else:  # 'Problems detected':
                color = 'red'
                st.toast(f'**{selected_name}** 서비스 중대 문제 발생!', icon="🚨")

        with col1_placeholder.container():
            # st.title(selected_name)
            st.subheader(f'**{selected_name}**  👉 :{color}[{status}]')
            # st.markdown('**This is :blue-background[test].** abcd')

            st.session_state.news_list = []  # 뉴스 세션 클리어
            st.write('🔎 News List')
            fetch_news(selected_name)

        with col2_placeholder.container():
            # st.divider()
            st.write('📈 Live Report Chart (last 24 hours)')

            # HTML iframe 태그를 사용하여 웹사이트 임베드
            chart_iframe_html = f"""
            <iframe src={chart_url} width="520" height="260" frameborder="0"></iframe>
            """
            st.markdown(chart_iframe_html, unsafe_allow_html=True)

            # st.divider()
            with st.spinner('서비스 맵 구성중...'):
                map_df = get_geo_location(map_df)

                st.write('🌏 Live Outage Map')

                # 지도 그리기
                drawing_df = map_df.dropna()
                multiple = get_multiple(drawing_df['Reports'])

                drawing_df['Reports'] = drawing_df['Reports'] * multiple
                st.map(drawing_df,
                       latitude='lat',
                       longitude='lon',
                       size='Reports',
                       color='color')

                with st.expander('상세 보기'):
                    st.write('Locations in the past 15 days')
                    st.write(map_df[['Location', 'Reports']])

        time.sleep(search_interval_min * 60)
        st.experimental_rerun()  # 페이지를 새로 고쳐서 업데이트 적용


if another_service and not service_code_name:
    st.session_state.news_list = []  # 뉴스 세션 클리어
    st.title(another_service)
    fetch_news(another_service, infinite_loop=True)


if service_code_name and another_service:
    st.error('하나의 서비스만 골라주세요!', icon="🚨")
    st.write(service_code_name, '  VS.  ', another_service)


# 메인 페이지 구성
# chat_placeholder = st.empty()

