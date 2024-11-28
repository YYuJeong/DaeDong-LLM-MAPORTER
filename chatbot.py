
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, DataFrameLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
import tempfile
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
import pandas as pd
# .env 파일 로드
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# API 키를 환경변수로 관리하기 위한 설정 파일
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

# 폴더 경로 설정
folder_path = "./data"  # 분석할 파일이 저장된 폴더 경로
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

# PDF 문서 로드 함수
def load_pdf_with_metadata(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load_and_split(text_splitter)
    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)
        doc.metadata["page"] = doc.metadata.get("page", "Unknown")
    return documents

# 엑셀 문서 로드 함수
def load_excel_with_metadata(file_path):
    documents = []
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        loader = DataFrameLoader(df, page_content_column=df.columns[0])
        sheet_docs = loader.load_and_split(text_splitter)
        for doc in sheet_docs:
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata["sheet_name"] = sheet_name
            doc.metadata["cell_range"] = f"A1:{df.columns[-1]}{len(df)}"  # 추가 셀 범위 정보
        documents.extend(sheet_docs)
    return documents


# CSV 문서 로드 함수
def load_csv_with_metadata(file_path):
    documents = []
    df = pd.read_csv(file_path)
    loader = DataFrameLoader(df, page_content_column=df.columns[0])
    csv_docs = loader.load_and_split(text_splitter)
    for doc in csv_docs:
        doc.metadata["source"] = os.path.basename(file_path)
        doc.metadata["cell_range"] = f"A1:{df.columns[-1]}{len(df)}"  # 추가 셀 범위 정보
    documents.extend(csv_docs)
    return documents

# 폴더 내 모든 문서를 로드

def load_documents_from_folder(folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".pdf"):
            documents.extend(load_pdf_with_metadata(file_path))
        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            documents.extend(load_excel_with_metadata(file_path))
        elif file_name.endswith(".csv"):
            documents.extend(load_csv_with_metadata(file_path))
    return documents



# 에이전트와 대화하는 함수
def chat_with_agent(user_input, agent_executor):
    result = agent_executor({"input": user_input})
    response = result['output']  # 명시적으로 출력 키를 처리
    return response

# 세션 기록 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state.session_history:
        st.session_state.session_history[session_ids] = ChatMessageHistory()
    return st.session_state.session_history[session_ids]

# 대화 내용 출력하는 함수
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg['role']).write(msg['content'])


# 모든 문서 로드
all_docs = load_documents_from_folder(folder_path)


# FAISS 인덱스 설정 및 생성
vector = FAISS.from_documents(all_docs, OpenAIEmbeddings())
retriever = vector.as_retriever()

# 도구 정의
retriever_tool = create_retriever_tool(
    retriever,
    name="csv_search",
    description="Use this tool to search information from the csv document"
)
# 경쟁사 정보
COMPETITOR_DATA = [
    {
        "title": "농자재업체 ‘상주쌀’ 소비촉진 캠페인 동참",
        "date": "24-11-11",
        "summary": [
            "국내 농자재업체들이 쌀 소비촉진 캠페인에 동참.",
            "TYM, 동방아그로, 대동공업이 캠페인 협약식 참여.",
            "직원 식당에서 급식용 쌀로 캠페인 확산 기여."
        ],
        "source": "https://www.nongmin.com/article/20241118500311"
    },
    {
        "title": "TYM, 세계 4대 농기계 전시회 ‘EIMA’ 참가",
        "date": "24-11-08",
        "summary": [
            "이탈리아 볼로냐에서 열린 국제 농업기계 박람회 ‘EIMA 2024’ 참가.",
            "신제품 트랙터 ‘T115’ 및 ‘T130’ 전시.",
            "국내 농기계 업체 중 유일하게 참가하여 글로벌 기술력 홍보."
        ],
        "source": "https://www.nongmin.com/article/20241118500274"
    },
    {
        "title": "“한국 농기계 이렇습니다”…TYM, 북미 우수 딜러 국내 초청 행사 열어",
        "date": "24-11-06",
        "summary": [
            "북미 우수 딜러를 국내로 초청하여 농기계 시승 및 품평회 진행.",
            "익산, 옥천 공장에서 글로벌 비전 및 성장 전략 공유.",
            "신제품 체험으로 글로벌 딜러와 상생 협력 강화."
        ],
        "source": "https://www.nongmin.com/article/20241118500251"
    }
]

# 시장 정보
MARKET_DATA = [
    {
        "title": "부정여론 확산…가락시장 12월 휴장 철회",
        "date": "24-11-18",
        "summary": [
            "서울시농수산식품공사가 가락시장 주 5일제 동절기 시범휴업 계획 일부 철회.",
            "제주 등 겨울채소 주산지 농민들의 피해를 우려하여 결정.",
            "출하 농민과 시장 유통 구조에 대한 개선 요구 지속."
        ],
        "source": "https://www.nongmin.com/article/20241118500226"
    },
    {
        "title": "[맛있는 이야기] 투박함 속 영양 가득…어머니 떠오르는 푸근한 맛 ‘호박’",
        "date": "24-11-18",
        "summary": [
            "호박이 다양한 요리에 활용되며 환절기 보양 음식으로 주목받음.",
            "일부 지역에서는 김치나 찌개 재료로 활용되며 영양 면에서도 우수.",
            "다이어트 및 소화에 도움을 주는 건강식품으로 평가."
        ],
        "source": "https://www.nongmin.com/article/20241118500494"
    },
    {
        "title": "[판매농협이 간다] 고품질 포도 ‘유통 일번지’…맛과 신뢰가 비결",
        "date": "24-11-18",
        "summary": [
            "경북 서상주농협이 고품질 샤인머스캣과 캠벨얼리 포도 유통을 주도.",
            "연간 1만 톤 이상의 취급량으로 약 710억 원의 매출 기록.",
            "영농 교육 및 농업인과의 협력을 통해 품질 유지 및 온라인 거래 확장."
        ],
        "source": "https://www.nongmin.com/article/20241118500603"
    }
]

# 기술 동향
TECH_DATA = [
    {
        "title": "농협은행, 빅데이터·AI 기반 기업대출 심사시스템 도입",
        "date": "2024-11-18",
        "summary": [
            "빅데이터와 AI를 활용해 신용평가 정확도 및 심사 효율성 제고.",
            "금융 접근성을 확대하고 중소기업 대출 활성화 기대."
        ],
        "source": "https://www.nongmin.com/article/20241118500374"
    },
    {
        "title": "정희용 의원, 인공지능산업 체계적 육성 위한 법안 발의",
        "date": "2024-11-18",
        "summary": [
            "AI 연구개발(R&D) 투자와 규제 완화를 통한 혁신 촉진.",
            "AI 인재 육성 방안을 포함하여 경제적 파급력 증대 전망."
        ],
        "source": "https://www.nongmin.com/article/20241118500336"
    },
    {
        "title": "농협사료 경남지사, 가축전염병 차단 위한 농가 방역지원 강화 나서",
        "date": "2024-11-18",
        "summary": [
            "가축전염병 확산 방지를 위해 방역 지원 활동 강화.",
            "AI 기술을 활용한 방역 데이터 분석으로 예측력 향상 도모."
        ],
        "source": "https://www.nongmin.com/article/20241118500381"
    }
]

# Streamlit 메인 코드
def main():
    # 페이지 설정
    st.set_page_config(page_title="마포터", layout="wide", page_icon="🤖")

    st.image('chatbot_image.png', width=600)
    st.markdown('---')
    st.title("안녕하세요! 대동 마포터 입니다")  # 시작 타이틀

    # 세션 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "session_history" not in st.session_state:
        st.session_state["session_history"] = {}

    tools = [retriever_tool]

    # LLM 설정
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Prompt 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You Should answer the user's questions in a friendly and kind manner. And should answer in Korean.
                Your name is `마포터`.
                You are a 15-year veteran market information analyst specializing in agriculture, agricultural machinery, future agriculture, and smart mobility in our company.
                Our company is a farming machinery firm called '대동'. and Representative domestic competitors include 'TYM' and 'LS엠트론'.
                Our company revenue structure is based on two main channels: domestic sales in South Korea and exports to regions such as Southeast Asia, the United States, and Europe. The revenue comes from parts, products, and services.
                Analyze the 'Title', 'Subheading', 'Content', and 'date' columns of the dataframe (df) to classify each article into one of the following categories: [정치/사회], [경쟁사 정보], [시장 정보] and [기술 동향].
                You have news articles from the last two weeks related to keywords such as [정치/사회], [경쟁사 정보], [시장 정보] and [기술 동향].
                When the chatbot begins, you should introduce yourself and ask the user for a keywords to search.
                If the keyword provided by the user does not match the pre-defined keyword format, you determine the user's intent and confirm if it matches a request related to the keywords [정치/사회], [경쟁사 정보], [시장 정보] or [기술 동향].
                Please answer questions following the format [FORMAT] below.
                `요약` should include a condensed version of the article's content. 
                Include the title in the `기사 제목` section.   

                #FORMAT
                [기사 제목] 
                * 일자 :  

                * 요약
                -
                -
                -

                출처 :              
                    
                """
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # 에이전트 생성
    agent = create_tool_calling_agent(llm, tools, prompt)

    # AgentExecutor 정의
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 사용자 입력 처리
    user_input = st.chat_input('질문이 무엇인가요?')

    if user_input:
        session_id = "default_session"
        session_history = get_session_history(session_id)

        # 조건문 수정
        if "경쟁사 정보" in user_input:
            # 경쟁사 정보 반환
            data_source = COMPETITOR_DATA
        elif "시장 정보" in user_input:
            # 시장 정보 반환
            data_source = MARKET_DATA
        elif "기술 동향" in user_input:
            # 기술 동향 반환
            data_source = TECH_DATA
        else:
            data_source = None

        if data_source is not None:
            response = "\n".join(
                [
                    f"[기사 제목] {item['title']}\n일자: {item['date']}\n* 요약\n" + "\n".join(f"- {summary}" for summary in item['summary']) + f"\n출처: {item['source']}\n"
                    for item in data_source
                ]
            )
        else:
            # 에이전트 실행
            if session_history.messages:
                previous_messages = [{"role": msg['role'], "content": msg['content']} for msg in session_history.messages]
                response = chat_with_agent(user_input + "\n\nPrevious Messages: " + str(previous_messages), agent_executor)
            else:
                response = chat_with_agent(user_input, agent_executor)

        # 메시지를 세션에 추가
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["messages"].append({"role": "assistant", "content": response})

        # 세션 기록에 메시지를 추가
        session_history.add_message({"role": "user", "content": user_input})
        session_history.add_message({"role": "assistant", "content": response})


    # 대화 내용 출력
    print_messages()

if __name__ == "__main__":
    main()
