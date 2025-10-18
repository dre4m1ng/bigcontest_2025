# src/main_app.py의 최종 완성본

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graph.builder import graph
from utils.logger import log_to_csv

st.set_page_config(page_title="소상공인 상담 챗봇 🤖", layout="wide")
st.title("소상공인 AI 상담 챗봇 🤖")
st.markdown("데이터 분석, 웹 검색, 정책자금 조회 등 다양한 업무를 도와드립니다.")

# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="안녕하세요! AI 챗봇입니다. 무엇을 도와드릴까요?")]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- 채팅 기록 표시 ---
for msg in st.session_state.messages:
    # Agent가 생성한 HumanMessage는 화면에 표시하지 않습니다.
    if isinstance(msg, AIMessage) or (isinstance(msg, HumanMessage) and msg.name == 'user'):
        with st.chat_message(msg.type):
            st.markdown(msg.content)

# --- 사용자 입력 처리 ---
if prompt := st.chat_input("질문을 입력하세요..."):
    # (핵심 수정!) 사용자의 입력을 HumanMessage(name='user')로 명확하게 생성합니다.
    user_message = HumanMessage(content=prompt, name='user')
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        agent_used = "N/A"
        status_message = ""

        # (핵심 수정!) 그래프에 전달할 입력값에도 name='user'가 포함된 메시지를 사용합니다.
        inputs = {"messages": [user_message]}
        config = {
            "configurable": {"thread_id": st.session_state.thread_id},
            "recursion_limit": 5
        }

        for chunk in graph.stream(inputs, config=config):
            if "web_searcher" in chunk:
                agent_used = "web_searcher"
                status_message = "🔍 웹에서 최신 정보를 검색하고 있습니다..."
            elif "data_analyzer" in chunk:
                agent_used = "data_analyzer"
                status_message = "📊 데이터를 분석하고 있습니다. 잠시만 기다려주세요..."
            elif "api_caller" in chunk:
                agent_used = "api_caller"
                status_message = "📞 정책자금 정보를 조회하고 있습니다..."

            if "generate" in chunk:
                generated_messages = chunk["generate"].get("messages", [])
                if generated_messages and isinstance(generated_messages[-1], AIMessage):
                    full_response = generated_messages[-1].content

            if full_response:
                message_placeholder.markdown(full_response + "▌")
            elif status_message:
                message_placeholder.markdown(status_message)

        message_placeholder.markdown(full_response)

    if full_response:
        ai_message = AIMessage(content=full_response)
        st.session_state.messages.append(ai_message) # 최종 답변만 세션에 추가
        log_to_csv(user_input=prompt, ai_output=full_response, agent_used=agent_used)