import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage

def web_search_agent(state):
    """
    웹 검색을 수행하는 Agent
    """
    print("--- 🔍 웹 검색 Agent 실행 ---")
    query = state["messages"][-1].content
    
    # Tavily 검색 도구 초기화
    search_tool = TavilySearchResults(max_results=2)
    search_result = search_tool.invoke(query)
    
    # 검색 결과를 HumanMessage로 만들어 반환
    return {"messages": [HumanMessage(content=str(search_result), name="web_search_agent")]}