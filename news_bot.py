import logging
import requests
from bs4 import BeautifulSoup

import streamlit as st


# 로깅 설정
logging.basicConfig(level=logging.DEBUG)

# 스레드 풀 실행자 초기화
# executor = concurrent.futures.ThreadPoolExecutor()


# Set verbose if needed
# globals.set_debug(True)


# # # # # # # # # # # #
# 테스트 코드. #
# # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # #
# ppt 파일 전체를 png 이미지로 변환하는 코드 #
# # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # #
# RAG를 위한 체인 #
# # # # # # # # # # # #


# # # # # # # # # # # #
# 이미지 해석 체인 #
# # # # # # # # # # # #


# # # # # # # # # #
# 이미지 체인 생성 #
# # # # # # # # # #


# # # # # # # # # # # # # # #
# down 차트, 지도 링크 받아오기
# # # # # # # # # # # # # # #


def get_service_chart_map(service_name):
    # url = "https://istheservicedown.com/problems/disney-plus"
    url = "https://istheservicedown.com/problems/" + service_name

    req_ = requests.Session()
    response_ = req_.get(url,
                         headers={'User-Agent': 'Popular browser\'s user-agent', })

    soup_ = BeautifulSoup(response_.content, 'html.parser')

    # 상태
    status = soup_.find('p').text

    # 차트 주소
    chart_url = soup_.find(id="chart-img")['src']

    # map 주소
    sub_url = soup_.find(title='Live Outage Map')['href']
    if sub_url:
        map_url = 'https://istheservicedown.com' + sub_url
    else:
        map_url = None

    return status, chart_url, map_url


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


# message key 등 세션 정보 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:  # api key
    st.session_state.api_key = None


# 사이드바 구성
option = st.sidebar.selectbox(
    "검색을 원하는 서비스는?",
    [a + '/' + b for a, b in companies_list],
    index=None,
    placeholder="서비스 이름 선택...",
)

if option:
    selected_code = option.split('/')[0]
    selected_name = option.split('/')[1]

    st.title(selected_name)
    # st.write("선택 서비스 코드: ", selected_code)

    status, chart_url, map_url = get_service_chart_map(selected_code)

    # 상태
    st.header(status)

    # HTML iframe 태그를 사용하여 웹사이트 임베드
    chart_iframe_html = f"""
    <iframe src={chart_url} width="800" height="600" frameborder="0"></iframe>
    """

    st.markdown(chart_iframe_html, unsafe_allow_html=True)

    map_iframe_html = f"""
    <iframe src={map_url} width="800" height="600" frameborder="0"></iframe>
    """

    st.markdown(map_iframe_html, unsafe_allow_html=True)

# 메인 페이지 구성

chat_placeholder = st.empty()

