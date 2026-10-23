from typing import Literal
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

class AgentState(TypedDict):
    task: str
    history: list[str]
    next_agent: str
    result: str
    iterations: int

def manager(state: AgentState) -> Command[Literal["researcher", "coder", "analyst", END]]:
    response = model.invoke([SystemMessage(content="""You are a multi-agent manager. Choose the next action based on the task and history.
               Return exactly one of:
               RESEARCHER - gather factual information
               CODER - write or improve Python code
               ANALYST - evaluate results and identify problems
               DONE - if the task is complete
               If the task needs multiple steps, select one agent at a time."""),
               HumanMessage(content=(f"Task: {state['task']}\nHistory: {state['history']}\nIterations: {state['iterations']}"))])
    decision = response.content.strip().upper()
    if state["iterations"] >= 8 or decision.startswith("DONE"):
        return Command(update={"result": "\n".join(state["history"])}, goto=END)
    routes = {"RESEARCHER": "researcher", "CODER": "coder", "ANALYST": "analyst",}
    destination = next((node for keyword, node in routes.items() if decision.startswith(keyword)), "analyst",)
    return Command(update={"next_agent": destination, "iterations": state["iterations"] + 1,}, goto=destination,)

def make_agent(role: str):
    def agent(state: AgentState) -> dict:
        response = model.invoke([SystemMessage(content=f"""You are the {role} agent in a multi-agent system.
                   Perform your assigned role on the user's task. Use the existing history as context. Return your findings concisely."""),
            HumanMessage(content=(f"Task: {state['task']}\n"f"Previous work: {state['history']}"))])
        return {"history": [f"{role.upper()}: {response.content}"]}
    return agent

builder = StateGraph(AgentState)
builder.add_node("manager", manager)
builder.add_node("researcher", make_agent("researcher"))
builder.add_node("coder", make_agent("coder"))
builder.add_node("analyst", make_agent("analyst"))
builder.add_edge(START, "manager")

for agent_name in ("researcher", "coder", "analyst"):
    builder.add_edge(agent_name, "manager")

graph = builder.compile()
result = graph.invoke({"task": "Build a Python function to identify prime numbers and test it.", "history": [], "next_agent": "", "result": "", "iterations": 0,})
print(result["result"])