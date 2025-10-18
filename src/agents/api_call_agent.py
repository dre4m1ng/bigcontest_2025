import json
from langchain_core.messages import HumanMessage

def get_policy_fund_info(topic: str) -> str:
    """
    소상공인 관련 정책 자금 정보를 반환하는 가상의 API 함수
    """
    print(f"--- 📞 가상 API 호출 (주제: {topic}) ---")
    if "청년" in topic:
        return json.dumps({
            "product_name": "청년 소상공인 특별자금",
            "interest_rate": "2.5%",
            "limit": "1억원 이내",
            "conditions": "만 39세 이하 청년 창업가"
        }, ensure_ascii=False)
    else:
        return json.dumps({
            "product_name": "일반 소상공인 성장자금",
            "interest_rate": "3.0%~",
            "limit": "5억원 이내",
            "conditions": "업력 1년 이상 소상공인"
        }, ensure_ascii=False)

def api_call_agent(state):
    """
    외부 API를 호출하는 Agent
    """
    query = state["messages"][-1].content
    result = get_policy_fund_info(query)
    
    return {"messages": [HumanMessage(content=result, name="api_call_agent")]}