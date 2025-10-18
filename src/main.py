import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# src 폴더를 sys.path에 추가하여 모듈을 찾을 수 있도록 함
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph.builder import graph

# --- Streamlit 페이지 설정 ---
st.set_page_config(page_title="소상공인 상담 챗봇 🤖", layout="wide")
st.title("소상공인 AI 상담 챗봇 🤖")
st.markdown("데이터 분석, 웹 검색, 정책자금 조회 등 다양한 업무를 도와드립니다.")

# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="안녕하세요! 소상공인 여러분의 사업 파트너, AI 챗봇입니다. 무엇을 도와드릴까요?")]

# --- 채팅 기록 표시 ---
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# --- 사용자 입력 처리 ---
if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 추가 및 표시
    st.session_state.messages.append(HumanMessage(content=prompt, name="user"))
    with st.chat_message("user"):
        st.markdown(prompt)

    # LangGraph 실행 및 결과 스트리밍
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 그래프 실행을 위한 초기 상태
        initial_state = {"messages": [HumanMessage(content=prompt, name="user")]}
        
        # stream()을 사용하여 실시간으로 중간 결과 확인
        for chunk in graph.stream(initial_state, stream_mode="values"):
            # Supervisor의 최종 답변만 추출
            if "messages" in chunk and isinstance(chunk["messages"][-1], AIMessage):
                final_message = chunk["messages"][-1].content
                full_response += final_message
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)

    # 최종 AI 응답을 세션 상태에 추가
    st.session_state.messages.append(AIMessage(content=full_response))