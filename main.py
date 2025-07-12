from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from shapely.geometry import Polygon, Point, LineString
import logging
import math
from typing import List, Dict, Any, Optional, Union
import re
from datetime import datetime
import time
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger("wall_api")

# Knowledge Base Constants (Rule 5.1)
MIN_WALL_LENGTH_M = 0.50  # meters
MIN_WALL_THICKNESS_M = 0.07  # meters
MAX_WALL_THICKNESS_M = 0.35  # meters
INTERIOR_WALL_MIN = 0.07
INTERIOR_WALL_MAX = 0.25
EXTERIOR_WALL_MIN = 0.25
EXTERIOR_WALL_MAX = 0.35
LOAD_BEARING_MIN = 0.15
PARALLEL_TOLERANCE = 0.1  # radians
CONNECTIVITY_DISTANCE = 5.0  # pixels for wall connectivity
MIN_POLYGON_AREA_M2 = 1.0  # m² for closed polygons

app = FastAPI(
    title="Wall Detection API - Knowledge Base Rule 5.1",
    description="Professional wall detection according to Knowledge Base Rule 5.1",
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
    return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)

def is_parallel(line1: dict, line2: dict, tolerance: float = PARALLEL_TOLERANCE) -> bool:
    if line1.get("type") != "line" or line2.get("type") != "line":
        return False
    if not all(key in line1 for key in ["p1", "p2"]) or not all(key in line2 for key in ["p1", "p2"]):
        return False
    dx1 = line1["p2"]["x"] - line1["p1"]["x"]
    dy1 = line1["p2"]["y"] - line1["p1"]["y"]
    dx2 = line2["p2"]["x"] - line2["p1"]["x"]
    dy2 = line2["p2"]["y"] - line2["p1"]["y"]
    len1 = math.sqrt(dx1**2 + dy1**2)
    len2 = math.sqrt(dx2**2 + dy2**2)
    if len1 == 0 or len2 == 0:
        return False
    dot_product = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
    return abs(abs(dot_product) - 1) < tolerance

def calculate_perpendicular_distance(line1: dict, line2: dict) -> float:
    mid1 = {"x": (line1["p1"]["x"] + line1["p2"]["x"]) / 2, "y": (line1["p1"]["y"] + line1["p2"]["y"]) / 2}
    mid2 = {"x": (line2["p1"]["x"] + line2["p2"]["x"]) / 2, "y": (line2["p1"]["y"] + line2["p2"]["y"]) / 2}
    dx = line1["p2"]["x"] - line1["p1"]["x"]
    dy = line1["p2"]["y"] - line1["p1"]["y"]
    length = math.sqrt(dx**2 + dy**2)
    if length == 0:
        return distance(mid1, mid2)
    nx = dx / length
    ny = dy / length
    vx = mid2["x"] - mid1["x"]
    vy = mid2["y"] - mid1["y"]
    return abs(-ny * vx + nx * vy)

def detect_wall_orientation(line: dict) -> str:
    dx = abs(line["p2"]["x"] - line["p1"]["x"])
    dy = abs(line["p2"]["y"] - line["p1"]["y"])
    return "horizontal" if dx > dy else "vertical"

def classify_wall_type_by_thickness(thickness_m: float) -> Dict[str, Any]:
    if EXTERIOR_WALL_MIN <= thickness_m <= EXTERIOR_WALL_MAX:
        return {"wall_type": "exterior", "is_load_bearing": thickness_m >= LOAD_BEARING_MIN, "confidence": 0.9}
    elif INTERIOR_WALL_MIN <= thickness_m <= INTERIOR_WALL_MAX:
        return {"wall_type": "interior", "is_load_bearing": thickness_m >= LOAD_BEARING_MIN, "confidence": 0.8}
    else:
        return {"wall_type": "unknown", "is_load_bearing": False, "confidence": 0.3}

def get_wall_label_codes(wall_type: str, orientation: str) -> Dict[str, str]:
    if wall_type == "interior" and orientation == "horizontal":
        return {"label_code": "MW01", "label_nl": "Binnenmuur_horizontaal", "label_en": "Interior_wall_horizontal"}
    elif wall_type == "interior" and orientation == "vertical":
        return {"label_code": "MW02", "label_nl": "Binnenmuur_verticaal", "label_en": "Interior_wall_vertical"}
    elif wall_type == "exterior" and orientation == "horizontal":
        return {"label_code": "MW03", "label_nl": "Buitenmuur_horizontaal", "label_en": "Exterior_wall_horizontal"}
    elif wall_type == "exterior" and orientation == "vertical":
        return {"label_code": "MW04", "label_nl": "Buitenmuur_verticaal", "label_en": "Exterior_wall_vertical"}
    else:
        return {"label_code": "MW00", "label_nl": "Onbekende_muur", "label_en": "Unknown_wall"}

def create_wall_polygon(line1: dict, line2: dict) -> List[Dict[str, float]]:
    return [
        {"x": line1["p1"]["x"], "y": line1["p1"]["y"]},
        {"x": line1["p2"]["x"], "y": line1["p2"]["y"]},
        {"x": line2["p2"]["x"], "y": line2["p2"]["y"]},
        {"x": line2["p1"]["x"], "y": line2["p1"]["y"]}
    ]

def analyze_wall_text_context(wall_polygon: List[Dict[str, float]], texts: List[Dict[str, Any]]) -> Dict[str, Any]:
    center_x = sum(p["x"] for p in wall_polygon) / len(wall_polygon)
    center_y = sum(p["y"] for p in wall_polygon) / len(wall_polygon)
    wall_center = {"x": center_x, "y": center_y}
    context = {"material_hints": [], "type_hints": [], "structural_hints": [], "materials": []}
    
    for text in texts:
        text_center = {"x": (text["bbox"]["x0"] + text["bbox"]["x1"]) / 2, "y": (text["bbox"]["y0"] + text["bbox"]["y1"]) / 2}
        if distance(wall_center, text_center) < 100:
            text_lower = text["text"].lower()
            if any(material in text_lower for material in ["beton", "metselwerk", "hout", "gips", "ms", "hsb", "kalkzandsteen", "betonsteen"]):
                context["material_hints"].append(text["text"])
                # Parse material layers (e.g., "gips 12,5mm")
                if re.match(r".*\d+\.?\d*mm", text_lower):
                    context["materials"].append(text["text"])
            if any(wall_type in text_lower for wall_type in ["binnen", "buiten", "draag", "schei", "ms", "hsb"]):
                context["type_hints"].append(text["text"])
            if any(struct in text_lower for struct in ["draag", "constructie", "fundament"]):
                context["structural_hints"].append(text["text"])
    
    return context

def is_polygon_closed(walls: List[Dict[str, Any]]) -> bool:
    if not walls:
        return False
    polygon_points = []
    for wall in walls:
        if wall["wall_type"] == "exterior":
            polygon_points.extend(wall["properties"]["polygon"])
    if len(polygon_points) < 3:
        return False
    polygon = Polygon([(p["x"], p["y"]) for p in polygon_points])
    return polygon.is_closed and polygon.area * (0.01765 ** 2) >= MIN_POLYGON_AREA_M2

@app.post("/detect-walls/", response_model=WallDetectionResponse)
async def detect_walls(request: WallDetectionRequest):
    try:
        logger.info(f"Detecting walls for {len(request.pages)} pages with scale {request.scale_m_per_pixel}")
        start_time = time.time()
        results = []
        
        for page_data in request.pages:
            logger.info(f"Analyzing walls on page {page_data.page_number}")
            walls = _detect_walls_rule_5_1(page_data, request.scale_m_per_pixel)
            
            page_stats = {
                "total_walls": len(walls),
                "exterior_walls": sum(1 for w in walls if w.get("wall_type") == "exterior"),
                "interior_walls": sum(1 for w in walls if w.get("wall_type") == "interior"),
                "load_bearing_walls": sum(1 for w in walls if w.get("classification", {}).get("is_load_bearing", False)),
                "total_wall_area_m2": round(sum(w.get("properties", {}).get("area_m2", 0) for w in walls), 2),
                "average_thickness_m": round(sum(w.get("thickness_meters", 0) for w in walls) / max(len(walls), 1), 3)
            }
            
            polygon_closed = is_polygon_closed(walls)
            validation = {
                "rule_5_1_compliance": True,
                "processed_lines": len(page_data.drawings.lines),
                "version": "2025-07",
                "errors": []
            }
            if not polygon_closed and page_stats["exterior_walls"] > 0:
                validation["errors"].append({
                    "error_code": "WALL_TOPOLOGY_001",
                    "message": "Exterior walls do not form a closed polygon",
                    "severity": "error"
                })
            
            results.append({
                "page_number": page_data.page_number,
                "walls": walls,
                "page_statistics": page_stats,
                "validation": validation
            })
        
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Successfully detected walls for {len(results)} pages in {processing_time:.2f} seconds")
        return {"pages": results}
        
    except Exception as e:
        logger.error(f"Error detecting walls: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _detect_walls_rule_5_1(page_data: PageData, scale: float) -> List[Dict[str, Any]]:
    walls = []
    processed_line_pairs = set()
    lines = [line.dict() for line in page_data.drawings.lines]
    texts = [text.dict() for text in page_data.texts]
    
    logger.info(f"Processing {len(lines)} lines for wall detection according to Rule 5.1")
    
    for i, line1 in enumerate(lines):
        if line1.get("type") != "line" or not all(key in line1 for key in ["p1", "p2"]):
            continue
        for j, line2 in enumerate(lines[i+1:], i+1):
            if line2.get("type") != "line" or not all(key in line2 for key in ["p1", "p2"]):
                continue
            pair_key = tuple(sorted([i, j]))
            if pair_key in processed_line_pairs:
                continue
            if not is_parallel(line1, line2):
                continue
            thickness_pixels = calculate_perpendicular_distance(line1, line2)
            thickness_m = thickness_pixels * scale
            if not (MIN_WALL_THICKNESS_M <= thickness_m <= MAX_WALL_THICKNESS_M):
                continue
            length_pixels = distance(line1["p1"], line1["p2"])
            length_m = length_pixels * scale
            if length_m <= MIN_WALL_LENGTH_M:
                continue
            
            orientation = detect_wall_orientation(line1)
            classification = classify_wall_type_by_thickness(thickness_m)
            label_info = get_wall_label_codes(classification["wall_type"], orientation)
            wall_polygon = create_wall_polygon(line1, line2)
            text_context = analyze_wall_text_context(wall_polygon, texts)
            area_m2 = length_m * thickness_m
            wall_id = f"wall_{len(walls)+1:03d}"
            
            connected_walls = []
            for k, other_wall in enumerate(lines):
                if k not in [i, j] and other_wall.get("type") == "line":
                    for p1 in [line1["p1"], line1["p2"], line2["p1"], line2["p2"]]:
                        for p2 in [other_wall["p1"], other_wall["p2"]]:
                            if distance(p1, p2) < CONNECTIVITY_DISTANCE:
                                connected_walls.append(f"wall_{k:03d}")
            
            wall_data = {
                "id": wall_id,
                "type": f"{classification['wall_type']}_wall_{orientation}",
                "label_code": label_info["label_code"],
                "label_nl": label_info["label_nl"],
                "label_en": label_info["label_en"],
                "label_type": "constructie",
                "thickness_meters": round(thickness_m, 3),
                "properties": {
                    "length_meters": round(length_m, 2),
                    "area_m2": round(area_m2, 2),
                    "orientation": orientation,
                    "polygon": wall_polygon,
                    "line1_index": i,
                    "line2_index": j,
                    "center_point": {
                        "x": sum(p["x"] for p in wall_polygon) / 4,
                        "y": sum(p["y"] for p in wall_polygon) / 4
                    }
                },
                "classification": {
                    "wall_type": classification["wall_type"],
                    "is_load_bearing": classification["is_load_bearing"],
                    "material_hints": text_context["material_hints"],
                    "materials": text_context["materials"],
                    "structural_type": "load_bearing" if classification["is_load_bearing"] else "non_load_bearing",
                    "confidence": classification["confidence"]
                },
                "validation": {
                    "status": True,
                    "reason": f"Rule 5.1 compliant: thickness {thickness_m:.3f}m, length {length_m:.2f}m",
                    "rule_5_1_compliance": {
                        "thickness_valid": MIN_WALL_THICKNESS_M <= thickness_m <= MAX_WALL_THICKNESS_M,
                        "length_valid": length_m > MIN_WALL_LENGTH_M,
                        "parallel_lines": True,
                        "thickness_classification": classification["wall_type"]
                    },
                    "norm_bindend": "Eurocode EN 1992"
                },
                "text_context": text_context,
                "connected_walls": list(set(connected_walls)),
                "line1_index": i,
                "line2_index": j,
                "orientation": orientation,
                "wall_type": classification["wall_type"],
                "confidence": classification["confidence"],
                "reason": f"Rule 5.1: {classification['wall_type']} wall, thickness {thickness_m:.3f}m, length {length_m:.2f}m",
                "version": "2025-07"
            }
            walls.append(wall_data)
            processed_line_pairs.add(pair_key)
    
    _validate_wall_topology(walls)
    return walls

def _validate_wall_topology(walls: List[Dict[str, Any]]) -> None:
    exterior_walls = [w for w in walls if w.get("wall_type") == "exterior"]
    interior_walls = [w for w in walls if w.get("wall_type") == "interior"]
    
    if exterior_walls and interior_walls:
        min_exterior_thickness = min(w.get("thickness_meters", 0) for w in exterior_walls)
        max_interior_thickness = max(w.get("thickness_meters", 0) for w in interior_walls)
        if min_exterior_thickness <= max_interior_thickness:
            for wall in walls:
                wall["validation"]["errors"] = wall["validation"].get("errors", []) + [{
                    "error_code": "WALL_TOPOLOGY_002",
                    "message": f"Exterior wall thickness ({min_exterior_thickness:.3f}m) not greater than interior wall thickness ({max_interior_thickness:.3f}m)",
                    "severity": "warning"
                }]
    
    for wall in walls:
        wall["validation"]["topology_valid"] = bool(wall.get("connected_walls"))
        if not wall["validation"]["topology_valid"]:
            wall["validation"]["errors"] = wall["validation"].get("errors", []) + [{
                    "error_code": "WALL_TOPOLOGY_003",
                    "message": "Wall not connected to other walls or components",
                    "severity": "warning"
                }]
        wall["validation"]["rule_5_1_thickness_hierarchy"] = True

@app.get("/")
async def root():
    return {
        "message": "Wall Detection API - Knowledge Base Rule 5.1 Implementation",
        "version": "2025-07",
        "knowledge_base": "KENNISBANK BOUWTEKENING-ANALYSE VECTOR API (Rule 5.1)",
        "rules": {
            "min_wall_length_m": MIN_WALL_LENGTH_M,
            "wall_thickness_range_m": [MIN_WALL_THICKNESS_M, MAX_WALL_THICKNESS_M],
            "interior_wall_range_m": [INTERIOR_WALL_MIN, INTERIOR_WALL_MAX],
            "exterior_wall_range_m": [EXTERIOR_WALL_MIN, EXTERIOR_WALL_MAX],
            "load_bearing_min_m": LOAD_BEARING_MIN
        },
        "endpoints": {
            "/detect-walls/": "Detect walls using Knowledge Base Rule 5.1",
            "/health/": "Health check"
        }
    }

@app.get("/health/")
async def health_check():
    return {
        "status": "healthy",
        "service": "wall_api",
        "version": "2025-07",
        "knowledge_base_compliance": "Rule 5.1",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    from gunicorn.app.base import BaseApplication
    import os

    class StandaloneApplication(BaseApplication):
        def __init__(self, app, options=None):
            self.application = app
            self.options = options or {}
            super().__init__()

        def load_config(self):
            for key, value in self.options.items():
                if key in self.cfg.settings and value is not None:
                    self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    port = int(os.environ.get("PORT", 8000))
    options = {
        "bind": "0.0.0.0:" + str(port),
        "workers": 4,
        "worker_class": "uvicorn.workers.UvicornWorker",
        "timeout": 300,  # Enforce 300-second timeout
        "graceful_timeout": 300  # Allow 300 seconds for graceful shutdown
    }
    StandaloneApplication(app, options).run()
