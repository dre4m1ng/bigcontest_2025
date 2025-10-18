# src/agents/data_analysis_agent.py

import os
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

def data_analysis_agent(state):
    """
    다양한 인코딩을 시도하여 'data' 폴더의 모든 CSV 파일을 동적으로 로드하고,
    Pandas DataFrame을 사용하여 데이터를 분석하는 Agent
    """
    print("--- 📊 데이터 분석 Agent 실행 ---")
    query = state["messages"][-1].content
    data_dir = "./data/"

    try:
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            return {"messages": [HumanMessage(content="분석할 CSV 파일이 'data' 폴더에 없습니다.", name="data_analysis_agent")]}
        
        print(f"--- 📂 로드된 CSV 파일: {', '.join(csv_files)} ---")

        dataframes = []
        # (핵심 수정!) 각 파일을 여러 인코딩으로 로드 시도
        for f in csv_files:
            file_path = os.path.join(data_dir, f)
            try:
                # 1. 가장 표준인 UTF-8로 먼저 시도
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    # 2. 실패하면, 한국어 환경에서 흔한 cp949로 시도
                    print(f"--- ⚠️ '{f}' 파일 UTF-8 로딩 실패. cp949로 재시도... ---")
                    df = pd.read_csv(file_path, encoding='cp949')
                except Exception as e:
                    # 3. cp949도 실패하면 에러 메시지 반환
                    error_message = f"'{f}' 파일 로딩 중 오류 발생: 인코딩 문제일 수 있습니다. (오류: {e})"
                    return {"messages": [HumanMessage(content=error_message, name="data_analysis_agent")]}
            dataframes.append(df)

    except FileNotFoundError:
        return {"messages": [HumanMessage(content="'data' 폴더를 찾을 수 없습니다.", name="data_analysis_agent")]}
    except Exception as e:
        return {"messages": [HumanMessage(content=f"파일 목록 조회 중 오류 발생: {e}", name="data_analysis_agent")]}
    
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    pandas_agent = create_pandas_dataframe_agent(
        llm, 
        dataframes,
        agent_executor_kwargs={"handle_parsing_errors": True}, 
        verbose=True, # 상세한 분석 과정을 보기 위해 True로 변경
        allow_dangerous_code=True # 위험한 코드 실행에 동의하는 서명 추가
    )
    
    try:
        response = pandas_agent.invoke(query)
        result = response.get("output", "결과를 찾을 수 없습니다.")
    except Exception as e:
        result = f"데이터 분석 중 오류가 발생했습니다: {e}"

    return {"messages": [HumanMessage(content=str(result), name="data_analysis_agent")]}