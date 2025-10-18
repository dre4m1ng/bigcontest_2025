import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

def data_analysis_agent(state):
    """
    Pandas DataFrame을 사용하여 데이터를 분석하는 Agent
    """
    print("--- 📊 데이터 분석 Agent 실행 ---")
    query = state["messages"][-1].content
    
    # 데이터 로드
    df = pd.read_csv("./data/sales_data.csv")
    
    # 데이터 분석을 위한 LLM Agent 생성
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    pandas_agent = create_pandas_dataframe_agent(llm, df, agent_executor_kwargs={"handle_parsing_errors": True}, verbose=False)
    
    try:
        response = pandas_agent.invoke(query)
        result = response.get("output", "결과를 찾을 수 없습니다.")
    except Exception as e:
        result = f"데이터 분석 중 오류가 발생했습니다: {e}"

    return {"messages": [HumanMessage(content=str(result), name="data_analysis_agent")]}