# tools/registry.py — discovers and registers all CXR tools

from tools.base import BaseTool, ToolSpec
from tools.general_perception.lung_region_detector import LungRegionDetector
from tools.general_perception.opacity_segmenter import OpacitySegmenter
from tools.general_perception.cardiomegaly_quantifier import CardiomegalyQuantifier
from tools.spatial_analysis.costophrenic_angle_analyzer import CostophrenicAngleAnalyzer
from tools.spatial_analysis.airspace_density_mapper import AirspaceDensityMapper
from tools.spatial_analysis.trachea_mediastinum_analyzer import TracheaMediastinumAnalyzer
from tools.geometry.pleural_line_detector import PleuralLineDetector
from tools.geometry.rib_bone_analyzer import RibBoneAnalyzer
from tools.geometry.tube_line_localizer import TubeLineLocalizer
from tools.auxiliary.image_quality_assessor import ImageQualityAssessor
from tools.auxiliary.differential_ranker import DifferentialRanker
from tools.auxiliary.terminate import Terminate


# ── Registry ──────────────────────────────────────────────────────────────────

_ALL_TOOLS: list[BaseTool] = [
    ImageQualityAssessor(),
    LungRegionDetector(),
    OpacitySegmenter(),
    CardiomegalyQuantifier(),
    CostophrenicAngleAnalyzer(),
    AirspaceDensityMapper(),
    TracheaMediastinumAnalyzer(),
    PleuralLineDetector(),
    RibBoneAnalyzer(),
    TubeLineLocalizer(),
    DifferentialRanker(),
    Terminate(),
]

_REGISTRY: dict[str, BaseTool] = {t.id: t for t in _ALL_TOOLS}


def get_tool(tool_id: str) -> BaseTool:
    if tool_id not in _REGISTRY:
        raise KeyError(f"Tool '{tool_id}' not found. Available: {list_tool_ids()}")
    return _REGISTRY[tool_id]


def list_tools() -> list[BaseTool]:
    return list(_REGISTRY.values())


def list_tool_ids() -> list[str]:
    return list(_REGISTRY.keys())


def list_specs() -> list[ToolSpec]:
    return [t.spec for t in _REGISTRY.values()]


def toolbox_prompt_block() -> str:
    """Returns the full toolbox description formatted for the planner prompt."""
    return "\n\n".join(t.spec.to_prompt_block() for t in _ALL_TOOLS)


def register_tool(tool: BaseTool) -> None:
    """Register a new tool at runtime (e.g. from a plugin)."""
    _REGISTRY[tool.id] = tool
