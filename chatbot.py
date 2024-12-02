
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
import pandas as pd

# 환경 변수 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_csv_data(csv_path):
    index_path = 'faiss_index'
    if os.path.exists(index_path):
        # 기존 FAISS 인덱스 로드
        vector = FAISS.load_local(index_path, OpenAIEmbeddings())
    else:
        # CSV 데이터 로드 및 처리
        df = pd.read_csv(csv_path)
        df = df.fillna("")
        documents = [
            Document(
                page_content=row['Content'],
                metadata={"title": row['Title'], "date": row['date'], "url": row['URL']}
            )
            for _, row in df.iterrows()
            if row['Content'].strip()
        ]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        split_docs = text_splitter.split_documents(documents)
        vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
        # FAISS 인덱스 저장
        vector.save_local(index_path)
    retriever = vector.as_retriever()
    tool = Tool(
        name="news_search",
        func=retriever.get_relevant_documents,
        description="Search for relevant agricultural news articles.",
    )
    return tool

def chat_with_agent(user_input, agent_executor):
    # 에이전트 실행
    result = agent_executor({"input": user_input})
    response = result['output']
    return response

def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg['role']).write(msg['content'])

def main():
    st.set_page_config(page_title="농업 뉴스 Q&A", layout="wide", page_icon="🌾")
    st.image('Maporter_image.png', width=600)
    st.markdown('---')
    st.title("안녕하세요! '대동 마포터' 입니다")
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    # 기존 대화 내용 출력
    print_messages()

    user_input = st.chat_input('질문이 무엇인가요?')

    if 'agent_executor' not in st.session_state:
        with st.spinner('에이전트를 초기화하고 있습니다. 잠시만 기다려주세요...'):

            csv_path = './data/news_filtered.csv' 
            pdf_search = load_csv_data(csv_path)
            tools = [pdf_search]
            llm = ChatOpenAI(model_name="gpt-4-mini", temperature=0)
            
            # 메모리 설정
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system",
                    "Be sure to answer in Korean. You are a helpful assistant. "
                    "Make sure to use the `pdf_search` tool for searching information from the PDF document. "
                    "Please always include emojis in your responses with a friendly tone. "
                    "Your name is `대동 마포터`. Please introduce yourself at the beginning of the conversation."
                    
                    ''' 
                    
                    When searching for articles containing the word "대동," 
                    make sure to look for information specifically about "대동, a company that develops agricultural machinery and agricultural technology." 
                    Note that "대동" and "대구" are different words.

                    You are a very friendly chatbot called `대동 마포터`. \n\n
                    At the beginning of the chatbot conversation, always display the message: "I am a friendly assistant providing various insights related to agriculture **`based on the latest collected news.`**"\n\n
                    "You are a helpful assistant providing insights based on agricultural news articles collected. "
                    "Always respond in a professional and friendly tone with structured and informative answers."),
                    Please enter the area you're curious about in relation to agriculture: `농업과 관련한 [정치/사회], [경쟁사 정보], [시장 정보] [기술 동향] 분야 중 궁금한 분야를 입력해주세요. `
                    '경쟁사 정보': Our company `대동`, develops tractors, and our competitors include `LS엠트론`, `TYM`, and `존 디어`.
                    When searching for news articles about `'경쟁사 정보'`, try to find articles specifically related to "LS Mtron" or "TYM." If no such articles are available, simply respond that there are none.
                    '시장 정보': For articles containing general information about the agricultural market.
                    '기술 동향': For articles related to the latest or new technologies in the agricultural field, or articles focused on technology.
                    Our company, Daedong, is an agricultural technology and manufacturing company that develops agricultural machinery such as tractors, rice transplanters, and combines. 
                    If a user requests an article about `대동`, ensure that it is related to `대동, which develops tractors, rice transplanters, combines, and other agricultural machinery.`
                    Do not show articles simply because they contain the word `대동`, `대동 주식회사`, `주식회사 대동`, `(주)대동`

                    
                    When summarizing a news article in your response, always use the following #format:
                    When providing answers following the #format, make sure to structure them with clean spacing by leaving one line between each sentence.
                    When writing [기사 내용] in the #format, always summarize the full content of the article using **3–5 bullet points**. and Make sure to write it in multiple lines, not just one line. 
                    For topics related to [정치/사회], [경쟁사 정보], [시장 정보] [기술 동향] in agriculture, please find and summarize five relevant news articles using the format above.
                    
                    Print the [기사 제목] and use **`\n`** to ensure a line break.
                    Print the [기사 날짜] and use **`\n`** to ensure a line break.
                    Print the [기사 내용] and use **`\n`** to ensure a line break.

                    #Format

                    **[기사 제목]** Title of the news article **n\n\n
                    **[기사 날짜]** Date of the news article **\n\n
                    **[기사 내용]** Make sure to write it in **multiple lines**, not just one line.  \n\n
                        1. First, write a **2–3 sentence** summary of the article. \n\n
                        2. Then, create a separate section summarizing the key points in **3–5 bullet points**. Each bullet should highlight a single key takeaway. **\n\n\n
                    **[출처]** URL link to the original news article **\n\n\n
                    
                    
                    Make sure to follow the format of [기사 제목], [기사 날짜], [기사 내용], and [출처], just like the ## Example. Ensure there is a line break after each entry.
                    ## Example (Strictly follow this)
                    **[기사 제목]** Example Title  \n\n
                    **[기사 날짜]** Example Date  \n\n
                    **[기사 내용]**  \n\n
                    Summary: This is a brief summary of the news content. It should contain 2–3 sentences.  \n

                    Key Points:  
                    - • First key point of the news.   \n
                    - • Second key point of the news.  \n
                    - • Third key point of the news.  \n\n

                    **[출처]** URL \n\n
                    '''),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input} \n\n Be sure to include emoji in your responses."),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )

            agent = create_tool_calling_agent(llm, tools, prompt, memory=memory)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)
            st.session_state['agent_executor'] = agent_executor
    else:
        agent_executor = st.session_state['agent_executor']
        
    if user_input:
        with st.spinner('답변을 생성하고 있습니다. 잠시만 기다려주세요...'):
            response = chat_with_agent(user_input, agent_executor)
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["messages"].append({"role": "assistant", "content": response})
    print_messages()

if __name__ == "__main__":
    main()
