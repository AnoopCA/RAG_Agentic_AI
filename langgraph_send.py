
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


class OverallState(TypedDict):
    task: str
    subtasks: list[str]
    results: Annotated[list[str], operator.add]
    final_answer: str


class WorkerState(TypedDict):
    task: str
    subtask: str


def planner(state: OverallState) -> dict:
    response = model.invoke([
        SystemMessage(content="""
Break the task into independent subtasks.
Return only a numbered list, with one subtask per line.
Create as many subtasks as needed, usually 2 to 5.
"""),
        HumanMessage(content=state["task"]),
    ])

    subtasks = [
        line.split(".", 1)[-1].strip()
        for line in response.content.splitlines()
        if line.strip() and any(c.isalpha() for c in line)
    ]

    return {"subtasks": subtasks}


def dispatch(state: OverallState):
    return [
        Send(
            "worker",
            {
                "task": state["task"],
                "subtask": subtask,
            },
        )
        for subtask in state["subtasks"]
    ]


def worker(state: WorkerState) -> dict:
    response = model.invoke([
        SystemMessage(content="""
You are a specialized worker.
Complete only the assigned subtask.
Provide useful findings that can be combined with other workers' findings.
"""),
        HumanMessage(content=(
            f"Overall task: {state['task']}\n"
            f"Your subtask: {state['subtask']}"
        )),
    ])

    return {
        "results": [
            f"Subtask: {state['subtask']}\n"
            f"Result: {response.content}"
        ]
    }


def synthesizer(state: OverallState) -> dict:
    response = model.invoke([
        SystemMessage(content="""
Combine all worker results into one coherent answer.
Resolve inconsistencies and avoid repeating information.
"""),
        HumanMessage(content=(
            f"Original task: {state['task']}\n"
            f"Worker results:\n" +
            "\n\n".join(state["results"])
        )),
    ])

    return {"final_answer": response.content}


builder = StateGraph(OverallState)

builder.add_node("planner", planner)
builder.add_node("worker", worker)
builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "planner")

builder.add_conditional_edges(
    "planner",
    dispatch,
    ["worker"],
)

builder.add_edge("worker", "synthesizer")
builder.add_edge("synthesizer", END)

graph = builder.compile()

result = graph.invoke({
    "task": (
        "Explain the advantages and disadvantages of microservices "
        "compared with a monolithic architecture."
    ),
    "subtasks": [],
    "results": [],
    "final_answer": "",
})

print(result["final_answer"])
