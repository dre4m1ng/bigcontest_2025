from typing import List, Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Agent들이 공유하는 상태
    - messages: 대화 기록 (덮어쓰는 대신 계속 추가되도록 Annotated 사용)
    - next: 다음에 호출할 Agent의 이름
    """
    # messages의 타입 힌트를 수정하여 메시지가 누적되도록 합니다.
    messages: Annotated[list, add_messages]
    next: str