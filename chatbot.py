
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv
import pandas as pd
from langchain.docstore.document import Document
from langchain.tools import Tool  # 추가 임포트


# .env 파일 로드
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
# .env 파일 로드

def load_csv_data(csv_path):
    # CSV 파일에서 기사 데이터를 로드
    df = pd.read_csv(csv_path)
    
    # 결측값 처리 (NaN 값을 빈 문자열로 대체)
    df = df.fillna("")
    
    # 기사 데이터를 Document 형태로 변환
    documents = [
        Document(
            page_content=row['Content'],
            metadata={"title": row['Title'], "date": row['date'], "url": row['URL']}
        )
        for _, row in df.iterrows()
        if row['Content'].strip()  # Content가 비어 있지 않은 경우만 처리
    ]
    
    # 텍스트 분할기 설정
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    split_docs = text_splitter.split_documents(documents)

    # FAISS 인덱스 생성
    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever()

    # 도구로 변환
    tool = Tool(
        name="news_search",
        func=retriever.get_relevant_documents,
        description="Search for relevant agricultural news articles.",
    )

    return tool

def load_pdf_files(pdf_paths):
    all_documents = []

    for pdf_path in pdf_paths:
        # PyPDFLoader를 사용하여 파일 로드
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)

    # 텍스트 분할기 설정
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    split_docs = text_splitter.split_documents(all_documents)

    # FAISS 인덱스 설정 및 생성
    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever()

    # 도구 정의
    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="Use this tool to search information from the PDF document."
    )
    return retriever_tool

# 에이전트와 대화하는 함수
def chat_with_agent(user_input, agent_executor):
    result = agent_executor({"input": user_input})
    response = result['output']  # 명시적으로 출력 키를 처리
    return response

# 대화 내용 출력하는 함수
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg['role']).write(msg['content'])

# Streamlit 메인 코드
def main():
    # 페이지 설정
    st.set_page_config(page_title="농업 뉴스 Q&A", layout="wide", page_icon="🌾")

    st.image('Maporter_image.png', width=600)
    st.markdown('---')
    st.title("안녕하세요! '대동 마포터' 입니다")  # 시작 타이틀

    # 세션 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 특정 PDF 경로 지정
    # CSV 데이터 로드
    csv_path = './data/news_filtered.csv' 
    if csv_path:
        pdf_search = load_csv_data(csv_path)
        tools = [pdf_search]

        # LLM 설정
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)

        # 프롬프트 설정
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "Be sure to answer in Korean. You are a helpful assistant. "
                 "Make sure to use the `pdf_search` tool for searching information from the PDF document. "
                 "Please always include emojis in your responses with a friendly tone. "
                 "Your name is `대동 마포터`. Please introduce yourself at the beginning of the conversation."
                 ''' 
                 You are a very friendly chatbot called `대동 마포터`. \n\n
                 At the beginning of the chatbot conversation, always display the message: "I am a friendly assistant providing various insights related to agriculture **`based on the latest collected news.`**"\n\n
                 "You are a helpful assistant providing insights based on agricultural news articles collected. "
                 "Always respond in a professional and friendly tone with structured and informative answers."),
                Please enter the area you're curious about in relation to agriculture: `농업과 관련한 [정치/사회], [경쟁사 정보], [시장 정보] [기술 동향] 분야 중 궁금한 분야를 입력해주세요. `
                 [경쟁사 정보]: Our company `대동`, develops tractors, and our competitors include `LS엠트론`, `TYM`, and `존 디어`.
                 When searching for news articles about `[경쟁사 정보]`, try to find articles specifically related to "LS Mtron" or "TYM." If no such articles are available, simply respond that there are none.
                [시장 정보]: For articles containing general information about the agricultural market.
                [기술 동향]: For articles related to the latest or new technologies in the agricultural field, or articles focused on technology.
                 
                 When summarizing a news article in your response, always use the following #format:
                 When providing answers following the #format, make sure to structure them with clean spacing by leaving one line between each sentence.
                When writing [뉴스 내용] in the #format, always summarize the full content of the article using 3~5 bullet points. Ensure the points are clear, concise, and at least one full line long.
                For topics related to [정치/사회], [경쟁사 정보], [시장 정보] [기술 동향] in agriculture, please find and summarize five relevant news articles using the format above.
                
            
                #Format

                **[기사 제목]** Title of the news article **n\n\n
                **[기사 날짜]** Date of the news article **\n\n
                **[뉴스 내용]** First, provide a 2–3 sentence summary of the overall news content. Then, summarize the main points again in 3~5 bullet points format. **\n\n\n
                **[뉴스 출처]** URL link to the original news article **\n\n\n
                
                
                Make sure to follow the format of [기사 제목], [기사 날짜], [뉴스 내용], and [뉴스 출처, just like the ## Example. Ensure there is a line break after each entry.
                ## Example
                [기사 제목] 정길수 전남도의원 “강대찬 벼 미질 논란 후속 조치 미흡” \n\n
                [기사 날짜] 24-11-08 \n\n
                [뉴스 내용] \n
                 * 기술원에서 보완 및 추진 사항에 대한 보고가 누락되었다는 지적이 있습니다. \n
                 * 생태농업이 산업적 농업의 지속가능성을 보완하기 위한 기술적 선택지로 간주되고 있습니다. \n\n
                [뉴스 출처] url \n\n
                 '''),
                ("placeholder", "{chat_history}"),
                ("human", "{input} \n\n Be sure to include emoji in your responses."),
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
            response = chat_with_agent(user_input, agent_executor)

            # 메시지를 세션에 추가
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": response})

        # 대화 내용 출력
        print_messages()

if __name__ == "__main__":
    main()
