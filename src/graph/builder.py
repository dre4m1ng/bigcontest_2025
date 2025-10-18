import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# .env 파일에서 API 키 로드
load_dotenv()

# 모듈화된 Agent 함수들을 가져옵니다.
from agents.web_search_agent import web_search_agent
from agents.data_analysis_agent import data_analysis_agent
from agents.api_call_agent import api_call_agent
from graph.state import AgentState

# LLM 초기화
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# --- 1. Router 역할을 할 LLM 프롬프트 ---
router_prompt = """
당신은 입력된 텍스트에서 키워드를 찾아 작업을 분류하는 매우 단순한 로봇 분류기입니다.
당신의 유일한 임무는 아래 [IF-ELSE 조건문]을 순서대로 확인하고, 가장 먼저 일치하는 단 하나의 작업을 선택하는 것입니다. 절대 문장의 의미를 해석하거나 추론하지 마십시오.

**[IF-ELSE 조건문]**

1.  **IF** 입력 텍스트에 다음 키워드 중 **하나라도 포함**되어 있는가?
    - '분석'
    - '파일'
    - '데이터'
    - 'CSV'
    - '데이터_레이아웃'
    - 'big_data_set'  (뒤에 숫자가 붙어도 포함)
    -> **THEN** 당신의 결정은 **`data_analyzer`** 이다. (여기서 즉시 중단)

2.  **ELSE IF** 입력 텍스트에 다음 키워드 중 하나라도 포함되어 있는가?
    - '뉴스'
    - '트렌드'
    - '검색'
    -> **THEN** 당신의 결정은 **`web_searcher`** 이다. (여기서 즉시 중단)

3.  **ELSE IF** 입력 텍스트에 다음 키워드 중 하나라도 포함되어 있는가?
    - '정책자금'
    - '대출'
    - '지원금'
    -> **THEN** 당신의 결정은 **`api_caller`** 이다. (여기서 즉시 중단)

4.  **ELSE** (위 1, 2, 3번 조건에 단 하나도 해당하지 않는 모든 경우)
    -> **THEN** 당신의 결정은 **`generate`** 이다.

**[입력 텍스트]**
"{query}"

**[당신의 결정 (위 조건문에 따라 결정된 단 하나의 작업)]**
"""

# --- 2. 최종 답변 생성 역할을 할 LLM 프롬프트 ---
generation_prompt = """당신은 소상공인을 위한 전문 AI 상담가입니다.
지금까지의 대화 내용과 Agent가 찾아온 정보를 종합하여, 사용자의 질문에 대한 최종 답변을 친절하고 명확하게 생성해주세요.

[대화 내용 및 정보]
{messages}

[최종 답변]
"""

def router_node(state):
    """
    메시지 출처를 확인하여, Agent의 보고는 즉시 generate로 보내고,
    사용자의 신규 요청만 LLM에게 판단을 맡깁니다.
    """
    print("--- 🧑‍⚖️ Router(Supervisor) 실행 ---")
    
    messages = state["messages"]
    last_message = messages[-1]

    # 1. (핵심!) 메시지 출처가 Agent인지 먼저 확인합니다.
    # HumanMessage이지만, 이름이 'user'가 아니라면 Agent가 만든 메시지입니다.
    if isinstance(last_message, HumanMessage) and last_message.name != "user":
        print(f"--- ✅ '{last_message.name}' Agent 작업 완료. 최종 답변 생성으로 직행합니다. ---")
        # LLM에게 물어볼 필요 없이, 즉시 'generate' 노드로 가는 지름길을 택합니다.
        return {"next": "generate"}

    # 2. 위 조건에 해당하지 않는 경우 (즉, 사용자의 최초 질문인 경우)에만 LLM을 호출하여 판단합니다.
    valid_destinations = ["data_analyzer", "web_searcher", "api_caller", "generate"]

    prompt = router_prompt.format(query=last_message.content)
    raw_decision = llm.invoke(prompt).content.strip()
    print(f"--- [DEBUG] LLM 원본 출력: '{raw_decision}' ---")

    cleaned_decision = ""
    for node_name in valid_destinations:
        if node_name in raw_decision:
            cleaned_decision = node_name
            break
    
    if not cleaned_decision:
        print(f"--- [경고] '{raw_decision}'에서 유효한 노드를 찾을 수 없습니다. 'generate'로 기본 설정합니다. ---")
        cleaned_decision = "generate"
    
    print(f"--- Router의 최종 정제된 결정: '{cleaned_decision}' ---")
    return {"next": cleaned_decision}

def generation_node(state):
    print("--- 💬 최종 답변 생성 ---")
    messages = state["messages"]
    
    message_str = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    prompt = generation_prompt.format(messages=message_str)
    final_response = llm.invoke(prompt).content.strip()
    
    return {"messages": [AIMessage(content=final_response)]}

# --- 3. 그래프 구성 ---
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("web_searcher", web_search_agent)
workflow.add_node("data_analyzer", data_analysis_agent)
workflow.add_node("api_caller", api_call_agent)
workflow.add_node("generate", generation_node) # 답변 생성 노드

workflow.set_entry_point("router") # 시작 노드 설정

workflow.add_conditional_edges(
    "router",
    lambda x: x["next"],
    {
        "web_searcher": "web_searcher",
        "data_analyzer": "data_analyzer",
        "api_caller": "api_caller",
        "generate": "generate",
    },
)

workflow.add_edge("generate", END)
workflow.add_edge("web_searcher", "router")
workflow.add_edge("data_analyzer", "router")
workflow.add_edge("api_caller", "router")

graph = workflow.compile(checkpointer=MemorySaver())
print("✅ LangGraph가 최종 구조로 성공적으로 컴파일되었습니다!")