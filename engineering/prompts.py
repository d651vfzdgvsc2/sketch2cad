"""Versioned prompts for evidence-grounded template selection, not guessed CAD."""
PROMPT_VERSION = "engineering-template-v1"

TEMPLATE_PROMPT = """你是工程图结构识别助手。程序已在原图像素坐标中测量并验证模板候选。
全图宽 {W}、高 {H}；原点左上，x向右、y向下。此阶段不换算毫米、不翻转y。
模板库：{catalog}
本批候选（稳定ID、参数、拟合残差和像素支持）：{candidates}
OCR候选（可能识别错误）：{texts}

任务：逐一判断本批候选的语义角色，并指出组合关系或歧义。
- 保留每一个候选ID；角色只能为 outline/hole/slot/annotation/frame/unknown。
- 长圆槽的两端是相切半圆；圆角矩形必须保留圆角；不将它们简化为尖角矩形。
- 候选为测量结果，不能凭“机械零件通常如此”改坐标、增加孔或删除细节。
- 孔阵列/同心圆/平行/相切关系必须引用本批存在的ID和图片证据，不跨视图拼接。
- 区分尺寸文字、螺纹M、直径、半径、角度和数量；标注与图像比例冲突时报告。
- 不认识或有遮挡的候选标为unknown，加入needs_review，不能猜。
- 不生成Python或DXF，不输出新坐标；不依赖图元数量限制来简化图纸。

仅输出JSON对象，结构如下：
{{"classifications":[{{"id":"候选ID","role":"outline","evidence":"具体依据"}}],
"relations":[{{"type":"concentric_circles","ids":["ID1","ID2"],"evidence":"依据"}}],
"needs_review":[{{"id":"候选ID","reason":"原因"}}]}}
模型判断只是建议；坐标、删除、合并必须继续接受程序的像素验证。
"""


def validate_semantics(payload, known_ids):
    if not isinstance(payload, dict):
        raise ValueError("Semantic response must be an object")
    known = set(known_ids)
    seen = set()
    roles = {"outline", "hole", "slot", "annotation", "frame", "unknown"}
    for item in payload.get("classifications", []):
        ident = item.get("id")
        if ident not in known or ident in seen or item.get("role") not in roles:
            raise ValueError("Invalid, duplicate or unknown semantic entity")
        seen.add(ident)
    if seen != known:
        raise ValueError("Semantic response omitted candidates")
    relation_types = {"hole_array", "concentric_circles", "parallel", "tangent", "symmetric"}
    for item in payload.get("relations", []):
        ids = item.get("ids", [])
        if item.get("type") not in relation_types or len(set(ids)) < 2 or not set(ids) <= known:
            raise ValueError("Invalid semantic relation")
    for item in payload.get("needs_review", []):
        if item.get("id") not in known:
            raise ValueError("Unknown review candidate")
    return payload
