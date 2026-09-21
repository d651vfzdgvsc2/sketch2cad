"""LangGraph 多Agent 编排：识别 -> 提取 -> 校验 -> Leader(收敛/回环)。"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.extractor import extract
from agents.leader import leader
from agents.recognizer import recognize
from agents.verifier import verify


class AgentState(TypedDict, total=False):
    image: str
    params: dict
    used_params: dict
    round: int
    log: list[Any]
    width: float
    height: float
    ocr: list
    vlm: dict
    ir_json: str
    counts: dict
    score: dict
    feedback: dict
    obj: float
    render_png: str
    dxf: str
    done: bool
    # 历史最优（保证多Agent结果永不差于基线）
    best_obj: float
    best_ssim: float
    best_chamfer: float
    best_ir_json: str
    best_score: dict
    best_render_png: str
    best_round: int


def _route(state: AgentState) -> str:
    return "end" if state.get("done") else "retry"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("recognize", recognize)
    g.add_node("extract", extract)
    g.add_node("verify", verify)
    g.add_node("leader", leader)

    g.add_edge(START, "recognize")
    g.add_edge("recognize", "extract")
    g.add_edge("extract", "verify")
    g.add_edge("verify", "leader")
    g.add_conditional_edges("leader", _route, {"retry": "extract", "end": END})
    return g.compile()


def run_agent(image: str, params: dict | None = None) -> AgentState:
    app = build_graph()
    return app.invoke({"image": image, "params": params or {}, "round": 0, "log": []})
