from typing import List, TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Agent들이 공유하는 상태
    - messages: 대화 기록
    - next: 다음에 호출할 Agent의 이름
    """
    messages: List[BaseMessage]
    next: str