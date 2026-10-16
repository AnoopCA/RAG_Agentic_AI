from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

class AgentState(TypedDict):
    task: str

class SupervisorDecision(BaseModel):
    decision: Literal["delegate", "finish"] = Field(description="delegate = worker must continue working. finish = task has been verified successfully.")
    instruction: str = Field(description="Detailed instruction for the worker.")

def supervisor(state: AgentState):
    prompt = f"""You are the supervisor of an autonomous software development team."""
    decision = model.with_structured_output(SupervisorDecision).invoke([SystemMessage(content=prompt)])
    return {"supervisor_decision": decision.decision, "worker_task": decision.instruction}

def worker(state: AgentState):
    new_attempt = state["attempts"] + 1
    prompt = f"""You are the software developer in an autonomous agent system."""
    response = model.invoke([SystemMessage(content=prompt)])
    return {"code": response.content, "attempts": new_attempt, "verification_status": "NOT_VERIFIED"}

def verifier(state: AgentState):
    verification_prompt = f"""You are a strict software verifier."""
    result = model.invoke([SystemMessage(content=verification_prompt)]).content
    return {"verification_result": result, "verification_status": "PASS" if result.strip().upper().startswith("PASS") else "FAIL"}

def supervisor_router(state: AgentState) -> Literal["worker", "end"]:
    return "end" if state["supervisor_decision"] == "finish" else "worker"

builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor)
builder.add_node("worker", worker)
builder.add_node("verifier", verifier)
builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", supervisor_router, {"worker": "worker", "end": END})
builder.add_edge("worker", "verifier")
builder.add_edge("verifier", "supervisor")
graph = builder.compile()

initial_state = {"task": "Create a Python function called calculate_premium."}

result = graph.invoke(initial_state)
print(result["code"])
print(result["verification_result"])