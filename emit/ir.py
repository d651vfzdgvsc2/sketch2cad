"""中间表示 (IR)：AI 与确定性代码之间的契约。

坐标约定：一律使用「图像像素坐标」——原点在左上角，x 向右，y 向下。
写出 DXF 时再统一翻转 y 轴，保证全流程只有一处坐标变换。
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Point = tuple[float, float]
EntityType = Literal["line", "polyline", "circle", "arc", "ellipse", "text"]


class LayerSpec(BaseModel):
    name: str
    color: int = 7


class Entity(BaseModel):
    type: EntityType
    layer: str = "outline"

    # line
    start: Optional[Point] = None
    end: Optional[Point] = None

    # polyline
    points: Optional[list[Point]] = None
    closed: bool = False

    # circle / arc
    center: Optional[Point] = None
    radius: Optional[float] = None
    start_angle: Optional[float] = None  # 图像坐标系角度（度，x 正向为 0，y 向下为正）
    end_angle: Optional[float] = None

    # ellipse
    major: Optional[float] = None  # 半长轴
    minor: Optional[float] = None  # 半短轴

    # text
    content: Optional[str] = None
    pos: Optional[Point] = None
    height: Optional[float] = None
    rotation: float = 0.0

    @model_validator(mode="after")
    def _check(self) -> "Entity":
        if self.type == "line" and (self.start is None or self.end is None):
            raise ValueError("line 需要 start/end")
        if self.type == "polyline" and (self.points is None or len(self.points) < 2):
            raise ValueError("polyline 至少需要 2 个点")
        if self.type in ("circle", "arc") and (self.center is None or self.radius is None):
            raise ValueError(f"{self.type} 需要 center/radius")
        if self.type == "ellipse" and (self.center is None or self.major is None or self.minor is None):
            raise ValueError("ellipse 需要 center/major/minor")
        if self.type == "text" and (self.pos is None or self.content is None):
            raise ValueError("text 需要 pos/content")
        return self


class DrawingIR(BaseModel):
    width: float
    height: float
    layers: list[LayerSpec] = Field(default_factory=lambda: [LayerSpec(name="outline")])
    entities: list[Entity] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entities:
            out[e.type] = out.get(e.type, 0) + 1
        return out

    def to_json(self) -> str:
        return self.model_dump_json(indent=1)

    @classmethod
    def from_json(cls, text: str) -> "DrawingIR":
        return cls.model_validate_json(text)
