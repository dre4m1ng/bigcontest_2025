# src/tools/data_analysis_tool.py (최종 완성본)

import os, pandas as pd, traceback
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

def _analyze_data(query: str) -> str:
    """실제 데이터 분석 로직을 처리하는 내부 함수"""
    data_dir = "./data/"
    try:
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files: return "분석할 CSV 파일이 'data' 폴더에 없습니다."
        
        dataframes = []
        for f in csv_files:
            file_path = os.path.join(data_dir, f)
            try: df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError: df = pd.read_csv(file_path, encoding='cp949')
            dataframes.append(df)
    except Exception as e:
        return f"파일 로딩 중 오류 발생: {e}"

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    
    # Agent에게 명확한 사전 작업 지시사항(prefix)을 추가.
    agent_prefix = """
당신은 여러 개의 Pandas DataFrame을 다루는 데이터 분석 전문가입니다.
당신에게는 실제 데이터 파일들과 함께, 컬럼 설명을 담은 '데이터_레이아웃' 파일이 주어질 수 있습니다.
**[분석 파일 설명]**
big_data_set1_f.csv -> '2025_빅콘테스트_데이터_레이아웃_20250902_데이터셋1.csv' 에 컬럼_한글명 및 설명
big_data_set2_f.csv -> '2025_빅콘테스트_데이터_레이아웃_20250902_데이터셋2.csv' 에 컬럼_한글명 및 설명
big_data_set3_f.csv -> '2025_빅콘테스트_데이터_레이아웃_20250902_데이터셋3.csv' 에 컬럼_한글명 및 설명

**[매우 중요한 행동 강령]**
1.  사용자가 한글로 된 컬럼명(예: '상권_코드_명')을 언급하면, **가장 먼저 '데이터_레이아웃'이라는 문구가 들어간 파일을 참조**하여 그에 해당하는 실제 영어 컬럼명(예: 'ADSTRD_CD_NM')을 찾아야 합니다.
2.  실제 Python 코드를 실행할 때는 반드시 '데이터_레이아웃' 파일에서 찾은 **영어 컬럼명을 사용**해야 합니다.
3.  사용자의 질문에 최대한 정확하게 답변할 수 있는 Python 코드를 생성하고 실행하세요.
"""

    try:
        pandas_agent = create_pandas_dataframe_agent(
            llm, 
            dataframes,
            prefix=agent_prefix, # 지시사항을 prefix 인자로 전달
            agent_executor_kwargs={"handle_parsing_errors": True}, 
            verbose=True,
            allow_dangerous_code=True
        )
        
        response = pandas_agent.invoke({"input": query})
        return response.get("output", "결과를 찾을 수 없습니다.")
    except Exception as e:
        detailed_error = traceback.format_exc()
        print(f"--- 💥 DATA ANALYZER TOOL 내부 오류: {detailed_error} ---")
        return f"데이터 분석 Agent 실행 중 오류 발생: {e}"

# 외부에서 사용할 최종 '도구'
data_analysis_tool = RunnableLambda(_analyze_data)