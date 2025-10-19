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
from tools.marketing_idea_tool import marketing_idea_generator_tool

# LLM 초기화
load_dotenv() # .env 파일에서 API 환경 변수 로드
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# --- 1. 도구(Tool) 등록 ---
# 모든 도구를 이름과 함께 딕셔너리로 묶어 Executor가 쉽게 찾아 쓸 수 있도록 합니다.
tools = {
    "data_analyzer": data_analysis_tool,
    "web_searcher": web_search_tool,
    "api_caller": api_caller_tool,
    "marketing_idea_generator": marketing_idea_generator_tool,
}

# --- 2. 챗봇의 새로운 두뇌: 노드(Node) 정의 ---

def planner_node(state: AgentState):
    """사용자의 비즈니스 문제를 해결하기 위한 '실행 가능한 계획'만을 생성합니다."""
    print("--- 🤔 최고 전략 책임자(Planner) 활동 시작 ---")
    
    prompt = f"""당신은 사용자의 요청을 해결하기 위한, 실행 가능한 단계별 계획을 생성하는 AI 플래너입니다.

**[매우 중요한 출력 규칙]**
- 당신의 최종 출력물은 **오직 번호가 매겨진 실행 계획 목록**이어야 합니다.
- 각 줄은 **반드시 `[Tool: 도구이름]` 형식으로 시작**해야 합니다.
- **절대로** 서론, 문제 정의, 가설, 요약 등 다른 어떤 설명도 포함해서는 안 됩니다. 오직 실행 계획 목록만 출력하세요.

**[예시]**
- **요청**: "재방문율 30% 이하 매장의 재방문율을 높일 마케팅 아이디어를 줘."
- **올바른 출력 (반드시 이 형식으로!):**
  1. [Tool: data_analyzer] 재방문 고객과 첫 방문 고객의 특성(방문 요일, 시간대, 주문 메뉴, 결제 금액)을 비교 분석하여 차이점을 찾아줘.
  2. [Tool: web_searcher] '카페 재방문율 높이는 최신 마케팅 전략'을 검색해서 성공 사례 3가지를 요약해줘.
  3. [Tool: marketing_idea_generator] 위 1, 2번의 분석 결과와 성공 사례를 바탕으로, 해당 매장을 위한 구체적인 마케팅 아이디어 3가지를 근거와 함께 제시해줘.

**사용 가능한 도구:**
- **data_analyzer**: 데이터 파일(고객 특성, 매출 등)을 심층 분석합니다.
- **web_searcher**: 최신 트렌드, 경쟁사 정보 등을 검색합니다.
- **marketing_idea_generator**: 분석된 데이터와 트렌드를 바탕으로 마케팅 아이디어를 생성합니다.

**사용자 요청:** "{state['messages'][-1].content}"

**위 규칙과 예시에 따라, 오직 실행 계획 목록만 생성해주세요:**
"""
    
    response = llm.invoke(prompt)
    
    # (핵심 수정!) LLM의 출력물에서 '[Tool:'로 시작하는 줄만 '계획'으로 인정합니다.
    plan = [
        step.strip() for step in response.content.split('\n') 
        if step.strip() and '[Tool:' in step
    ]
    
    print(f"--- 📝 수립된 최종 계획 ---\n" + "\n".join(plan))
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
    """모든 근거를 종합하여 전문 컨설팅 리포트 형식의 최종 답변을 생성합니다."""
    print("--- ✍️ 시니어 컨설턴트(Synthesizer) 최종 보고서 작성 ---")

    evidence = "\n\n".join(
        [f"**실행 계획:** {step}\n**수집된 근거:**\n{result}" for step, result in state.get("past_steps", [])]
    )
    
    prompt = f"""당신은 수집된 모든 근거 자료를 종합하여, 소상공인을 위한 최종 비즈니스 컨설팅 보고서를 작성하는 시니어 컨설턴트입니다.
답변은 아래의 **[보고서 형식]**을 반드시 따라야 하며, 모든 내용은 제공된 **[수집된 근거 자료]**에 기반해야 합니다.

**[사용자의 초기 질문]**
{state['messages'][0].content}

**[수집된 근거 자료]**
{evidence}

**[보고서 형식]**
### 📝 문제점 진단
(데이터 분석 결과를 바탕으로 현재 상황과 가장 큰 문제점을 요약합니다.)

### 💡 해결 방안 제안
(마케팅 아이디어 생성 결과를 바탕으로, 구체적인 해결책과 실행 방안을 제시합니다.)

### 📈 기대 효과
(제시한 해결 방안을 실행했을 때 예상되는 긍정적인 결과를 설명합니다.)

### 📚 핵심 근거 자료
(답변의 각 부분이 어떤 근거 자료(예: [근거: big_data_set1.csv 분석 결과], [근거: 마케팅 아이디어 생성 결과])에 기반했는지 명확히 명시합니다.)

**위 형식에 맞춰 최종 컨설팅 보고서를 작성해주세요:**
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