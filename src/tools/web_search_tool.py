import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

def _search_web(query: str) -> str:
    """
    웹 검색을 수행하고 결과를 문자열로 반환하는 도구입니다.
    """
    print("--- 🔍 웹 검색 도구 실행 ---")
    try:
        search_tool = TavilySearchResults(max_results=3)
        search_result = search_tool.invoke(query)
        return str(search_result)
    except Exception as e:
        return f"웹 검색 중 오류 발생: {e}"

# LangChain 시스템과 호환되도록 RunnableLambda로 감싸줍니다.
web_search_tool = RunnableLambda(_search_web)