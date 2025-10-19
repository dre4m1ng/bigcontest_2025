# src/graph/builder.py

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# 새로운 상태와 우리가 만든 모든 '도구(Tool)'들을 가져옵니다.
from graph.state import AgentState
from tools.data_analysis_tool import data_analysis_tool
from tools.web_search_tool import web_search_tool
from tools.api_call_tool import api_caller_tool

# LLM 초기화
load_dotenv() # .env 파일에서 API 환경 변수 로드
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# --- 1. 도구(Tool) 등록 ---
# 모든 도구를 이름과 함께 딕셔너리로 묶어 Executor가 쉽게 찾아 쓸 수 있도록 합니다.
tools = {
    "data_analyzer": data_analysis_tool,
    "web_searcher": web_search_tool,
    "api_caller": api_caller_tool,
}

# --- 2. 챗봇의 새로운 두뇌: 노드(Node) 정의 ---

def planner_node(state: AgentState):
    """사용자의 질문을 기반으로 '가장 효율적인' 실행 계획을 수립합니다."""
    print("--- 🤔 계획 수립(Planner) 시작 ---")
    
    prompt = f"""당신은 사용자의 질문을 해결하기 위한 '가장 효율적인' 실행 계획을 수립하는 전문 플래너입니다.
당신의 최우선 목표는 각 도구(tool)가 단 한 번의 호출로 작업을 완료할 수 있도록, **가능한 가장 짧고 간결한 계획**을 세우는 것입니다.

**[매우 중요한 규칙]**
- 사용자의 질문이 하나의 도구로 해결될 수 있다면, 계획은 **반드시 단 하나의 단계**여야 합니다.
- 절대 하나의 작업을 여러 개의 자잘한 단계로 나누지 마세요.

**[예시]**
- **나쁜 계획 (절대 이렇게 하지 마세요):**
  1. [Tool: data_analyzer] 파일을 읽어줘.
  2. [Tool: data_analyzer] '상권_코드_명' 컬럼을 찾아줘.
  3. [Tool: data_analyzer] 그룹별로 개수를 세줘.
- **좋은 계획 (반드시 이렇게 하세요):**
  1. [Tool: data_analyzer] 'big_data_set1_f.csv' 파일에서 '상권_코드_명' 별로 데이터 개수를 계산해서 상위 5개만 알려줘.

**사용 가능한 도구:**
- **data_analyzer**: CSV, Excel 파일의 내용을 분석합니다.
- **web_searcher**: 최신 트렌드, 뉴스 등을 웹에서 검색합니다.
- **api_caller**: '정책자금', '대출' 등 금융 상품 정보를 조회합니다.

**사용자 질문:** "{state['messages'][-1].content}"

**가장 효율적인 실행 계획 (위 규칙과 예시를 반드시 참고):**
"""
    
    response = llm.invoke(prompt)
    plan = [step.strip() for step in response.content.split('\n') if step.strip()]
    
    print(f"--- 📝 수립된 계획 ---\n" + "\n".join(plan))
    return {"plan": plan, "past_steps": []}

def executor_node(state: AgentState):
    """계획의 다음 단계를 실행하는 '실행 전문가'"""
    print("--- ⚙️ 계획 실행(Executor) 시작 ---")
    
    step = state["plan"][0]
    
    try:
        tool_name = step.split("[Tool: ")[1].split("]")[0]
        query = step.split("]")[1].strip()
    except IndexError:
        return {"past_steps": state.get("past_steps", []) + [(step, "오류: 계획 형식이 잘못되었습니다.")]}

    print(f"---  [실행] 도구: {tool_name} // 질문: {query} ---")
    
    if tool_name in tools:
        tool = tools[tool_name]
        try:
            result = tool.invoke(query)
            past_step = (step, str(result))
        except Exception as e:
            # (핵심 수정!) 잡힌 오류의 내용을 터미널에 자세히 출력합니다.
            print(f"--- 🚨 EXECUTOR가 도구 실행 중 오류 감지: {e} ---") 
            past_step = (step, f"도구 실행 중 오류 발생: {e}")
            
        return {
            "plan": state["plan"][1:],
            "past_steps": state.get("past_steps", []) + [past_step]
        }
    else:
        return {"past_steps": state.get("past_steps", []) + [(step, "오류: 알 수 없는 도구입니다.")]}

def synthesizer_node(state: AgentState):
    """수집된 모든 근거를 종합하여 최종 답변을 생성하는 '종합 전문가'"""
    print("--- ✍️ 결과 종합(Synthesizer) 시작 ---")

    # past_steps에 저장된 모든 근거 자료를 하나의 텍스트로 합칩니다.
    evidence = "\n\n".join(
        [f"**실행 계획:** {step}\n**수집된 근거:**\n{result}" for step, result in state.get("past_steps", [])]
    )
    
    prompt = f"""당신은 수집된 근거 자료만을 사용하여 사용자의 초기 질문에 대한 최종 답변을 생성하는 전문 분석가입니다.
절대로 당신의 기존 지식을 사용해서는 안 됩니다. 답변은 반드시 한국어로 작성해야 합니다.

**[사용자의 초기 질문]**
{state['messages'][0].content}

**[수집된 근거 자료]**
{evidence}

**[최종 답변]**
위 근거 자료를 바탕으로, 각 내용의 출처(예: [근거: big_data_set1.csv 분석 결과], [근거: 웹 검색 결과])를 명시하여 최종 답변을 생성해주세요.
"""

    response = llm.invoke(prompt)
    return {"messages": [AIMessage(content=response.content)]}

# --- 3. 새로운 그래프 구성 ---

workflow = StateGraph(AgentState)

# 새로운 전문가 노드들을 그래프에 추가
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("synthesizer", synthesizer_node)

# 시작점은 항상 'planner'
workflow.set_entry_point("planner")

# 각 노드를 연결
workflow.add_edge("planner", "executor")
workflow.add_edge("synthesizer", END)

# Executor는 조건부로 연결: 실행할 계획이 남았는지 확인
def should_continue(state: AgentState):
    if state.get("plan"):
        return "executor" # 아직 실행할 계획이 남았으면 executor로 다시 이동 (루프)
    else:
        return "synthesizer" # 계획이 모두 끝났으면 synthesizer로 이동

workflow.add_conditional_edges("executor", should_continue)

# 그래프 최종 컴파일
graph = workflow.compile(checkpointer=MemorySaver())
print("=========================================//")
print("✅ '계획-실행-종합' 모델로 LangGraph가 성공적으로 컴파일되었습니다!")
# print(">> 그래프 노드:", graph.nodes)
print(">> 그래프 도구 목록:", list(tools.keys()))
print("=========================================")
print("시작 >>")