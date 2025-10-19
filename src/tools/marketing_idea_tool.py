# src/tools/marketing_idea_tool.py

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7) # 창의성을 위해 온도를 약간 높임

def _generate_marketing_idea(topic: str) -> str:
    """
    주어진 주제(데이터 분석 결과, 트렌드)를 바탕으로 마케팅 아이디어를 생성합니다.
    """
    print("--- 💡 마케팅 아이디어 생성 도구 실행 ---")
    
    prompt = f"""당신은 데이터 기반 마케팅 아이디어 전문가입니다.
아래에 제공된 '분석 결과 및 트렌드'를 바탕으로, 소상공인 매장을 위한 **구체적이고 실행 가능한 마케팅 아이디어 3가지**를 각각의 근거와 함께 제안해주세요.

**[분석 결과 및 트렌드]**
{topic}

**[마케팅 아이디어 제안 (구체적인 실행 방안과 근거 포함)]**
1. **아이디어**: ...
   - **근거**: ...
2. ...
"""
    
    response = llm.invoke(prompt)
    return response.content

marketing_idea_generator_tool = RunnableLambda(_generate_marketing_idea)