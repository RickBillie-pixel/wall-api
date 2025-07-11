"""
Wall Detection API - Knowledge Base Rule 5.1 - WORKING VERSION
Simple but effective implementation that handles large datasets
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import logging
import math
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wall_api")

# Knowledge Base Constants (Rule 5.1)
MIN_WALL_THICKNESS_M = 0.07  # 7cm
MAX_WALL_THICKNESS_M = 0.35  # 35cm
INTERIOR_WALL_MIN = 0.07     # 7cm
INTERIOR_WALL_MAX = 0.25     # 25cm  
EXTERIOR_WALL_MIN = 0.25     # 25cm
EXTERIOR_WALL_MAX = 0.35     # 35cm
LOAD_BEARING_MIN = 0.15      # 15cm
PARALLEL_TOLERANCE = 0.1     # radians

app = FastAPI(
    title="Wall Detection API - Rule 5.1",
    description="Wall detection according to Knowledge Base Rule 5.1",
    version="2025-07",
)

class DrawingItem(BaseModel):
    type: str
    p1: Optional[Dict[str, float]] = None
    p2: Optional[Dict[str, float]] = None
    p3: Optional[Dict[str, float]] = None
    rect: Optional[Dict[str, float]] = None
    length: Optional[float] = None
    color: Union[List[float], int, float, str, None] = Field(default=[0, 0, 0])
    width: Optional[float] = 1.0
    area: Optional[float] = None
    fill: Union[List[Any], Any] = Field(default_factory=list)
    
    @validator('color', pre=True)
    def normalize_color(cls, v):
        if isinstance(v, (int, float)):
            return [float(v), float(v), float(v)]
        elif isinstance(v, list):
            return v
        return [0, 0, 0]

class Drawings(BaseModel):
    lines: List[DrawingItem]
    rectangles: List[DrawingItem]
    curves: List[DrawingItem]

class TextItem(BaseModel):
    text: str
    position: Dict[str, float]
    font_size: float
    font_name: str
    color: Union[List[float], int, float, str, None] = Field(default=[0, 0, 0])
    bbox: Dict[str, float]
    
    @validator('color', pre=True)
    def normalize_color(cls, v):
        if isinstance(v, (int, float)):
            return [float(v), float(v), float(v)]
        elif isinstance(v, list):
            return v
        return [0, 0, 0]

class PageData(BaseModel):
    page_number: int
    page_size: Dict[str, float]
    drawings: Drawings
    texts: List[TextItem]
    is_vector: bool = True
    processing_time_ms: Optional[int] = None

class WallDetectionRequest(BaseModel):
    pages: List[PageData]
    scale_m_per_pixel: float = 1.0

class WallDetectionResponse(BaseModel):
    pages: List[Dict[str, Any]]

# Utility functions
def distance(p1: dict, p2: dict) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)

def line_angle(line: dict) -> float:
    """Calculate line angle in radians"""
    dx = line["p2"]["x"] - line["p1"]["x"]
    dy = line["p2"]["y"] - line["p1"]["y"]
    return math.atan2(dy, dx)

def lines_parallel(line1: dict, line2: dict, tolerance: float = PARALLEL_TOLERANCE) -> bool:
    """Check if two lines are parallel"""
    angle1 = line_angle(line1)
    angle2 = line_angle(line2)
    
    # Normalize angles to [0, π]
    angle1 = abs(angle1) % math.pi
    angle2 = abs(angle2) % math.pi
    
    angle_diff = min(abs(angle1 - angle2), abs(angle1 - angle2 + math.pi), abs(angle1 - angle2 - math.pi))
    return angle_diff < tolerance

def perpendicular_distance(line1: dict, line2: dict) -> float:
    """Calculate perpendicular distance between parallel lines"""
    # Get midpoints
    mid1_x = (line1["p1"]["x"] + line1["p2"]["x"]) / 2
    mid1_y = (line1["p1"]["y"] + line1["p2"]["y"]) / 2
    mid2_x = (line2["p1"]["x"] + line2["p2"]["x"]) / 2
    mid2_y = (line2["p1"]["y"] + line2["p2"]["y"]) / 2
    
    # Direction vector of line1 (normalized)
    dx = line1["p2"]["x"] - line1["p1"]["x"]
    dy = line1["p2"]["y"] - line1["p1"]["y"]
    length = math.sqrt(dx*dx + dy*dy)
    
    if length == 0:
        return math.sqrt((mid2_x - mid1_x)**2 + (mid2_y - mid1_y)**2)
    
    # Normalized direction vector
    dir_x = dx / length
    dir_y = dy / length
    
    # Vector from mid1 to mid2
    vec_x = mid2_x - mid1_x
    vec_y = mid2_y - mid1_y
    
    # Perpendicular distance = |cross product|
    return abs(vec_x * (-dir_y) + vec_y * dir_x)

def classify_wall_by_thickness(thickness_m: float) -> Dict[str, Any]:
    """Class
