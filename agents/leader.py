"""Leader Agent：根据校验报告决定「收敛」还是「调整参数重做」。"""
from __future__ import annotations

import json

from core.config import get_int
from tools.llm import chat_json
from vectorize.vectorize import DEFAULTS

PASS_OBJ = 0.90

LEADER_PROMPT = """你是多Agent协同出图系统的 Leader。下面是一轮"矢量化->渲染->比对"的校验报告。
请判断如何调整 OpenCV 参数，让下一轮结果更接近原图。只输出 JSON，不要解释。

当前参数：
{params}

校验报告：
{feedback}

可调参数（只允许改这些，且必须落在范围内）：
- circle_param2: 圆检测严格度，越大圆越少（20~90）
- min_line_len: 最短线段（15~80）
- merge_angle_tol: 合并角度容差（1~10）
- merge_dist_tol: 合并距离容差（2~15）
如果已经足够好，就原样返回。

输出格式：{{"params": {{...}}, "reason": "一句话"}}"""


def _rule_based(state: dict) -> dict:
    cur = {**DEFAULTS, **(state.get("params") or {})}
    fb = state.get("feedback", {})
    cp, cv = fb.get("circles_pred", 0), fb.get("circles_vlm", 0)
    if cv:
        if cp > cv:
            cur["circle_param2"] = min(cur["circle_param2"] + 8, 90)
        elif cp < cv:
            cur["circle_param2"] = max(cur["circle_param2"] - 8, 20)
    return cur


def _clamp(p: dict) -> dict:
    out = {**DEFAULTS, **p}
    out["circle_param2"] = min(max(float(out["circle_param2"]), 20), 90)
    out["min_line_len"] = min(max(float(out["min_line_len"]), 15), 80)
    out["merge_angle_tol"] = min(max(float(out["merge_angle_tol"]), 1), 10)
    out["merge_dist_tol"] = min(max(float(out["merge_dist_tol"]), 2), 15)
    return out


def _converged(state: dict) -> bool:
    issues = (state.get("feedback", {}) or {}).get("issues", [])
    return state.get("obj", 0.0) >= PASS_OBJ and not issues


def leader(state: dict) -> dict:
    max_rounds = get_int("MAX_REVISION_ROUNDS", 3)
    rnd = state.get("round", 0)
    log = list(state.get("log", []))

    if _converged(state) or rnd >= max_rounds:
        log.append({"round": rnd, "action": "STOP",
                    "reason": "已收敛" if _converged(state) else "达到最大轮数",
                    "iou": state.get("score", {}).get("iou")})
        return {"done": True, "log": log}

    cur = {**DEFAULTS, **(state.get("params") or {})}
    new_params, reason = None, ""
    try:
        reply = chat_json([
            {"role": "user", "content": LEADER_PROMPT.format(
                params=json.dumps(cur, ensure_ascii=False),
                feedback=json.dumps(state.get("feedback", {}), ensure_ascii=False))},
        ], max_tokens=400)
        new_params = _clamp(reply.get("params", {}))
        reason = reply.get("reason", "LLM 决策")
    except Exception as e:  # noqa: BLE001
        new_params = _rule_based(state)
        reason = f"规则兜底（LLM失败: {type(e).__name__}）"

    if new_params == cur:
        log.append({"round": rnd, "action": "STOP",
                    "reason": "参数已是最优，无进一步调整空间",
                    "iou": state.get("score", {}).get("iou")})
        return {"done": True, "log": log}

    log.append({"round": rnd, "action": "RETRY",
                "reason": reason, "iou": state.get("score", {}).get("iou"),
                "params": new_params})
    return {"params": new_params, "round": rnd + 1, "done": False, "log": log}
