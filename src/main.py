# src/main_app.py

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
    user_message = HumanMessage(content=prompt, name='user')
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        inputs = {"messages": [user_message]}
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        # Planner가 세운 계획을 표시할 변수
        plan_steps_str = ""
        # 실행된 단계를 표시할 변수
        executed_steps_str = ""

        for chunk in graph.stream(inputs, config=config):
            # planner가 계획을 세우면 화면에 표시
            if "planner" in chunk:
                plan = chunk["planner"].get("plan", [])
                plan_steps_str = "\n".join([f"⏳ {step}" for step in plan])
                message_placeholder.markdown(f"**수립된 작업 계획:**\n{plan_steps_str}")

            # executor가 단계를 실행하면 화면 업데이트
            if "executor" in chunk:
                past_steps = chunk["executor"].get("past_steps", [])
                executed_steps_str = "\n".join([f"✅ {step[0]}" for step in past_steps])
                remaining_plan = "\n".join([f"⏳ {step}" for step in chunk["executor"].get("plan", [])])
                message_placeholder.markdown(f"**작업 수행 현황:**\n{executed_steps_str}\n{remaining_plan}")
            
            # synthesizer가 최종 답변을 생성하면 표시
            if "synthesizer" in chunk:
                final_message = chunk["synthesizer"]["messages"][-1]
                if isinstance(final_message, AIMessage):
                    full_response = final_message.content
                    message_placeholder.markdown(full_response)
        
        message_placeholder.markdown(full_response)

    if full_response:
        ai_message = AIMessage(content=full_response)
        st.session_state.messages.append(ai_message)
        # 로깅은 이제 계획 단계, 근거 등을 포함하도록 확장할 수 있습니다.
        # 지금은 간단하게 마지막 도구 이름만 기록합니다.
        log_to_csv(user_input=prompt, ai_output=full_response, agent_used="Plan-and-Execute")