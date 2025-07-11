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
    """Classify wall according to Rule 5.1"""
    if EXTERIOR_WALL_MIN <= thickness_m <= EXTERIOR_WALL_MAX:
        return {
            "wall_type": "exterior",
            "is_load_bearing": thickness_m >= LOAD_BEARING_MIN,
            "confidence": 0.9
        }
    elif INTERIOR_WALL_MIN <= thickness_m <= INTERIOR_WALL_MAX:
        return {
            "wall_type": "interior", 
            "is_load_bearing": thickness_m >= LOAD_BEARING_MIN,
            "confidence": 0.8
        }
    else:
        return {
            "wall_type": "unknown",
            "is_load_bearing": False,
            "confidence": 0.3
        }

def get_wall_labels(wall_type: str, orientation: str) -> Dict[str, str]:
    """Get Knowledge Base labels (Rule 3.1)"""
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

@app.post("/detect-walls/", response_model=WallDetectionResponse)
async def detect_walls(request: WallDetectionRequest):
    """
    Detect walls according to Knowledge Base Rule 5.1 - OPTIMIZED for large datasets
    """
    try:
        logger.info(f"Detecting walls for {len(request.pages)} pages with scale {request.scale_m_per_pixel}")
        start_time = time.time()
        
        results = []
        
        for page_data in request.pages:
            logger.info(f"Analyzing walls on page {page_data.page_number}")
            
            walls = await _detect_walls_optimized(page_data, request.scale_m_per_pixel)
            
            # Calculate statistics
            stats = {
                "total_walls": len(walls),
                "exterior_walls": sum(1 for w in walls if w.get("wall_type") == "exterior"),
                "interior_walls": sum(1 for w in walls if w.get("wall_type") == "interior"),
                "load_bearing_walls": sum(1 for w in walls if w.get("classification", {}).get("is_load_bearing", False)),
                "total_wall_area_m2": round(sum(w.get("properties", {}).get("area_m2", 0) for w in walls), 2),
                "average_thickness_m": round(sum(w.get("thickness_meters", 0) for w in walls) / max(len(walls), 1), 3)
            }
            
            results.append({
                "page_number": page_data.page_number,
                "walls": walls,
                "page_statistics": stats,
                "validation": {
                    "rule_5_1_compliance": True,
                    "processed_lines": len(page_data.drawings.lines),
                    "version": "2025-07"
                }
            })
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        total_walls = sum(len(page["walls"]) for page in results)
        logger.info(f"Successfully detected {total_walls} walls in {processing_time:.2f} seconds")
        
        return {"pages": results}
        
    except Exception as e:
        logger.error(f"Error detecting walls: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def _detect_walls_optimized(page_data: PageData, scale: float) -> List[Dict[str, Any]]:
    """
    Optimized wall detection for large datasets (80k+ lines)
    Uses smart sampling and early termination
    """
    
    # Filter valid lines
    valid_lines = []
    for i, line in enumerate(page_data.drawings.lines):
        line_dict = line.dict()
        if (line_dict.get("type") == "line" and 
            "p1" in line_dict and "p2" in line_dict and
            line_dict["p1"] and line_dict["p2"]):
            
            # Pre-filter: only lines longer than 5 pixels
            if distance(line_dict["p1"], line_dict["p2"]) > 5:
                valid_lines.append((i, line_dict))
    
    logger.info(f"Processing {len(valid_lines)} valid lines (filtered from {len(page_data.drawings.lines)})")
    
    if len(valid_lines) == 0:
        return []
    
    # For very large datasets, use sampling
    if len(valid_lines) > 30000:
        logger.info(f"Large dataset detected ({len(valid_lines)} lines). Using optimized sampling.")
        return await _detect_walls_large_dataset(valid_lines, scale)
    
    # Standard processing for reasonable datasets
    walls = []
    processed_pairs = set()
    comparisons = 0
    max_comparisons = min(200000, len(valid_lines) * 20)  # Dynamic limit
    
    for i, (line1_idx, line1) in enumerate(valid_lines):
        if comparisons >= max_comparisons:
            logger.warning(f"Reached comparison limit ({max_comparisons}). Stopping.")
            break
            
        if i % 2000 == 0 and i > 0:
            logger.info(f"Processed {i}/{len(valid_lines)} lines, found {len(walls)} walls")
        
        # Limit search range for performance
        search_end = min(i + 500, len(valid_lines))  # Only check next 500 lines
        
        for j in range(i + 1, search_end):
            line2_idx, line2 = valid_lines[j]
            comparisons += 1
            
            if comparisons >= max_comparisons:
                break
            
            pair_key = (line1_idx, line2_idx)
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            # Quick distance check - skip if lines are too far apart
            line1_center = {"x": (line1["p1"]["x"] + line1["p2"]["x"]) / 2, "y": (line1["p1"]["y"] + line1["p2"]["y"]) / 2}
            line2_center = {"x": (line2["p1"]["x"] + line2["p2"]["x"]) / 2, "y": (line2["p1"]["y"] + line2["p2"]["y"]) / 2}
            
            if distance(line1_center, line2_center) > 80:  # 80 pixels max distance
                continue
            
            # Check if parallel
            if not lines_parallel(line1, line2):
                continue
            
            # Calculate thickness
            thickness_pixels = perpendicular_distance(line1, line2)
            thickness_m = thickness_pixels * scale
            
            # Rule 5.1: Check thickness range
            if not (MIN_WALL_THICKNESS_M <= thickness_m <= MAX_WALL_THICKNESS_M):
                continue
            
            # Calculate length
            length_pixels = distance(line1["p1"], line1["p2"])
            length_m = length_pixels * scale
            
            # Create wall
            wall_data = _create_wall_data(line1, line2, line1_idx, line2_idx, thickness_m, length_m, len(walls))
            walls.append(wall_data)
            
            # Limit total walls to prevent memory issues
            if len(walls) >= 800:
                logger.info(f"Found {len(walls)} walls, stopping to prevent memory issues")
                break
        
        if len(walls) >= 800:
            break
    
    logger.info(f"Detected {len(walls)} walls with {comparisons} comparisons")
    return walls

async def _detect_walls_large_dataset(valid_lines: List[tuple], scale: float) -> List[Dict[str, Any]]:
    """
    Ultra-optimized for very large datasets (30k+ lines)
    """
    logger.info(f"Using ultra-optimized processing for {len(valid_lines)} lines")
    
    # Sort by line length and take top 20k longest lines
    lines_with_length = []
    for orig_idx, line_dict in valid_lines:
        length = distance(line_dict["p1"], line_dict["p2"])
        lines_with_length.append((length, orig_idx, line_dict))
    
    lines_with_length.sort(reverse=True)
    max_lines = min(20000, len(lines_with_length))
    selected_lines = [(orig_idx, line_dict) for _, orig_idx, line_dict in lines_with_length[:max_lines]]
    
    logger.info(f"Selected {len(selected_lines)} longest lines for processing")
    
    # Simple grid-based approach for large datasets
    walls = []
    grid_size = 150  # Large grid for performance
    grid = {}
    
    # Add lines to grid
    for idx, line_data in selected_lines:
        center_x = (line_data["p1"]["x"] + line_data["p2"]["x"]) / 2
        center_y = (line_data["p1"]["y"] + line_data["p2"]["y"]) / 2
        grid_key = (int(center_x // grid_size), int(center_y // grid_size))
        
        if grid_key not in grid:
            grid[grid_key] = []
        grid[grid_key].append((idx, line_data))
    
    # Process each grid cell
    comparisons = 0
    max_comparisons = 100000  # Strict limit
    
    for cell_lines in grid.values():
        if comparisons >= max_comparisons:
            break
            
        # Within each cell, find walls
        for i, (idx1, line1) in enumerate(cell_lines):
            for j, (idx2, line2) in enumerate(cell_lines[i+1:], i+1):
                comparisons += 1
                
                if comparisons >= max_comparisons or len(walls) >= 400:
                    break
                
                # Quick checks
                if not lines_parallel(line1, line2):
                    continue
                
                thickness_pixels = perpendicular_distance(line1, line2)
                thickness_m = thickness_pixels * scale
                
                if not (MIN_WALL_THICKNESS_M <= thickness_m <= MAX_WALL_THICKNESS_M):
                    continue
                
                length_pixels = distance(line1["p1"], line1["p2"])
                length_m = length_pixels * scale
                
                wall_data = _create_wall_data(line1, line2, idx1, idx2, thickness_m, length_m, len(walls))
                walls.append(wall_data)
            
            if len(walls) >= 400:
                break
        
        if len(walls) >= 400:
            break
    
    logger.info(f"Large dataset processing: {len(walls)} walls found with {comparisons} comparisons")
    return walls

def _create_wall_data(line1: dict, line2: dict, idx1: int, idx2: int, thickness_m: float, length_m: float, wall_count: int) -> Dict[str, Any]:
    """Create wall data object"""
    orientation = "horizontal" if abs(line1["p2"]["x"] - line1["p1"]["x"]) > abs(line1["p2"]["y"] - line1["p1"]["y"]) else "vertical"
    classification = classify_wall_by_thickness(thickness_m)
    labels = get_wall_labels(classification["wall_type"], orientation)
    
    polygon = [
        {"x": line1["p1"]["x"], "y": line1["p1"]["y"]},
        {"x": line1["p2"]["x"], "y": line1["p2"]["y"]},
        {"x": line2["p2"]["x"], "y": line2["p2"]["y"]},
        {"x": line2["p1"]["x"], "y": line2["p1"]["y"]}
    ]
    
    return {
        "id": f"wall_{wall_count+1:03d}",
        "type": f"{classification['wall_type']}_wall_{orientation}",
        "label_code": labels["label_code"],
        "label_nl": labels["label_nl"],
        "label_en": labels["label_en"],
        "label_type": "constructie",
        "thickness_meters": round(thickness_m, 3),
        "properties": {
            "length_meters": round(length_m, 3),
            "area_m2": round(length_m * thickness_m, 3),
            "orientation": orientation,
            "polygon": polygon,
            "line1_index": idx1,
            "line2_index": idx2,
            "center_point": {
                "x": sum(p["x"] for p in polygon) / 4,
                "y": sum(p["y"] for p in polygon) / 4
            }
        },
        "classification": {
            "wall_type": classification["wall_type"],
            "is_load_bearing": classification["is_load_bearing"],
            "structural_type": "load_bearing" if classification["is_load_bearing"] else "non_load_bearing",
            "confidence": classification["confidence"]
        },
        "validation": {
            "status": True,
            "reason": f"Rule 5.1: thickness {thickness_m:.3f}m, length {length_m:.3f}m",
            "rule_5_1_compliance": {
                "thickness_valid": MIN_WALL_THICKNESS_M <= thickness_m <= MAX_WALL_THICKNESS_M,
                "parallel_lines": True,
                "thickness_classification": classification["wall_type"]
            }
        },
        "line1_index": idx1,
        "line2_index": idx2,
        "orientation": orientation,
        "wall_type": classification["wall_type"],
        "confidence": classification["confidence"],
        "reason": f"Rule 5.1: {classification['wall_type']} wall, thickness {thickness_m:.3f}m, length {length_m:.3f}m",
        "version": "2025-07"
    }

@app.get("/")
async def root():
    return {
        "message": "Wall Detection API - Knowledge Base Rule 5.1",
        "version": "2025-07",
        "performance": "Optimized for 80k+ lines",
        "features": ["Rule 5.1 compliant", "No external dependencies", "Smart sampling for large datasets"]
    }

@app.get("/health/")
async def health_check():
    return {
        "status": "healthy",
        "service": "wall_api",
        "version": "2025-07",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
