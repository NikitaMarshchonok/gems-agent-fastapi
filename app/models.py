from typing import List, Literal, Optional
from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]

class Message(BaseModel):
    role: Role
    content: str

class Gem(BaseModel):
    id: str
    name: str
    system_prompt: str = Field(default="You are a helpful assistant.")
    tools: List[str] = Field(default_factory=list)
    temperature: float = 0.2
    model: Optional[str] = None  # override default model if set

class GemCreate(BaseModel):
    name: str
    system_prompt: str = "You are a helpful assistant."
    tools: List[str] = Field(default_factory=list)
    temperature: float = 0.2
    model: Optional[str] = None

class GemUpdate(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: Optional[List[str]] = None
    temperature: Optional[float] = None
    model: Optional[str] = None

class ChatRequest(BaseModel):
    gem_id: str
    messages: List[Message]
    tools_mode: Literal["off", "auto"] = "auto"

class ChatResponse(BaseModel):
    content: str
    used_tool: Optional[str] = None
    tool_input: Optional[str] = None
