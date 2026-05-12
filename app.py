
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# --- 페이지 설정 ---
st.set_page_config(page_title="삼성 메모리카드 서포트", page_icon="💾")
st.title("📂 메모리카드 매뉴얼 Q&A 챗봇")
st.markdown("매뉴얼 내용을 바탕으로 답변해 드립니다.")

# --- RAG 체인 초기화 (캐싱 처리) ---
@st.cache_resource
def initialize_rag():
    # 1. PDF 로드
    pdf_path = "data/Samsung_Card_Manual_Korean_1.3.pdf"
    if not os.path.exists(pdf_path):
        st.error(f"파일을 찾을 수 없습니다: {pdf_path}")
        return None

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # 2. 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    # 3. 벡터DB 생성
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # 4. 프롬프트 및 모델 설정
    prompt = ChatPromptTemplate.from_template("""
    너는 삼성전자 메모리카드 매뉴얼에 대한 전문 어시스턴트이다.
    다음의 참고 문서를 바탕으로 질문에 정확하게 답하라.

    [참고문서]
    {context}

    [질문]
    {question}

    한글로 간결하고 정확하게 답변하라.
    """)

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # 5. 체인 구성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# 체인 생성 (최초 1회 실행)
rag_chain = initialize_rag()

# --- 채팅 인터페이스 구현 ---

# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 내용 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if user_input := st.chat_input("궁금한 점을 물어보세요."):
    # 1. 사용자 메시지 기록 및 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. 답변 생성 및 표시
    with st.chat_message("assistant"):
        if rag_chain:
            with st.spinner("답변을 생성하고 있습니다..."):
                response = rag_chain.invoke(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("데이터 로딩 문제로 답변을 드릴 수 없습니다.")
