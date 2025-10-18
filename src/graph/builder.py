import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# .env 파일에서 API 키 로드
load_dotenv()

# 모듈화된 Agent 함수들을 가져옵니다.
from src.agents.web_search_agent import web_search_agent
from src.agents.data_analysis_agent import data_analysis_agent
from src.agents.api_call_agent import api_call_agent
from src.graph.state import AgentState

# Supervisor 역할을 할 LLM
supervisor_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# 각 Agent의 역할과 Supervisor의 선택지를 정의하는 시스템 프롬프트
system_prompt = """
당신은 소상공인 상담 챗봇의 총괄 관리자(Supervisor)입니다.
사용자의 질문을 분석하여 다음 중 어떤 전문가에게 작업을 맡길지 결정해야 합니다.

1.  **web_searcher**: 최신 뉴스, 트렌드, 일반적인 정보 검색이 필요할 때 선택합니다.
2.  **data_analyzer**: '매출', '판매량', '수익' 등 제공된 데이터에 대한 분석이나 통계가 필요할 때 선택합니다.
3.  **api_caller**: '정책자금', '대출', '지원금' 등 구체적인 금융 상품 정보가 필요할 때 선택합니다.
4.  **FINISH**: 사용자의 질문에 직접 답변할 수 있거나, 모든 작업이 완료되었을 때 선택합니다.

사용자의 질문: "{query}"
당신의 결정 (web_searcher, data_analyzer, api_caller, FINISH 중 하나만 선택):
"""

def supervisor_node(state):
    """
    어떤 Agent를 호출할지 결정하는 노드
    """
    print("--- 🧑‍⚖️ Supervisor 실행 ---")
    last_message = state["messages"][-1]
    
    # 만약 마지막 메시지가 Agent의 결과물이라면, 바로 FINISH
    if isinstance(last_message, HumanMessage) and last_message.name != "user":
        print("--- Agent 작업 완료, Supervisor가 최종 답변 준비 ---")
        return {"next": "FINISH"}

    # 사용자의 질문을 기반으로 다음 단계를 결정
    prompt = system_prompt.format(query=last_message.content)
    response = supervisor_llm.invoke(prompt)
    decision = response.content.strip()
    
    print(f"--- Supervisor 결정: {decision} ---")
    return {"next": decision}

# 그래프 생성
workflow = StateGraph(AgentState)

# 노드(Agent) 추가
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("web_searcher", web_search_agent)
workflow.add_node("data_analyzer", data_analysis_agent)
workflow.add_node("api_caller", api_call_agent)

# 엣지(연결) 설정
workflow.set_entry_point("supervisor")

# Supervisor의 결정에 따라 분기하는 조건부 엣지
workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "web_searcher": "web_searcher",
        "data_analyzer": "data_analyzer",
        "api_caller": "api_caller",
        "FINISH": END,
    },
)

# 각 Agent 작업이 끝나면 다시 Supervisor에게 보고
workflow.add_edge("web_searcher", "supervisor")
workflow.add_edge("data_analyzer", "supervisor")
workflow.add_edge("api_caller", "supervisor")

# 그래프 컴파일
graph = workflow.compile()
print("✅ LangGraph가 성공적으로 컴파일되었습니다!")