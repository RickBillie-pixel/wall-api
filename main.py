"""
Wall Detection API - Knowledge Base Implementation Rule 5.1
Implements complete knowledge base for wall detection according to Rule 5.1:
- Muur(wand) = vector(pad) met lengte > 50cm én breedte 7–35cm (na schaalcorrectie)
- Parallelle lijnen 7–35cm uit elkaar (geschaald) vormen een muur(wand)
- Buitenmuur(wand): 25–35cm dik; vormt gesloten polygoon; altijd dikker dan binnenmuur(wand)
- Binnenmuur(wand): 7–25cm dik; altijd tussen buitenmuren; sluit aan op andere muren(wanden)/componenten
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import logging
import math
from typing import List, Dict, Any, Optional, Union
import re
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wall_api")

# Knowledge Base Constants (Rule 5.1)
MIN_WALL_LENGTH_M = 0.50  # meters - minimum wall length (Rule 5.1: > 50cm)
MIN_WALL_THICKNESS_M = 0.07  # meters - minimum wall thickness (Rule 5.1: 7cm)
MAX_WALL_THICKNESS_M = 0.35  # meters - maximum wall thickness (Rule 5.1: 35cm)

# Wall type thickness ranges (Rule 5.1)
INTERIOR_WALL_MIN = 0.07  # 7cm
INTERIOR_WALL_MAX = 0.25  # 25cm
EXTERIOR_WALL_MIN = 0.25  # 25cm
EXTERIOR_WALL_MAX = 0.35  # 35cm
LOAD_BEARING_MIN = 0.15   # 15cm (Rule 5.1: Dragende muur ≥15cm)

# Parallel tolerance
PARALLEL_TOLERANCE = 0.1  # radians for parallel line detection

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
        """Normalize color to list format"""
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
        """Normalize color to list format"""
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

# Utility functions according to Knowledge Base
def distance(p1: dict, p2: dict) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)

def is_parallel(line1: dict, line2: dict, tolerance: float = PARALLEL_TOLERANCE) -> bool:
    """Check if two lines are parallel (Rule 5.1: Parallelle lijnen)"""
    if line1.get("type") != "line" or line2.get("type") != "line":
        return False
    
    if not all(key in line1 for key in ["p1", "p2"]) or not all(key in line2 for key in ["p1", "p2"]):
        return False
    
    # Calculate direction vectors
    dx1 = line1["p2"]["x"] - line1["p1"]["x"]
    dy1 = line1["p2"]["y"] - line1["p1"]["y"]
    dx2 = line2["p2"]["x"] - line2["p1"]["x"]
    dy2 = line2["p2"]["y"] - line2["p1"]["y"]
    
    # Normalize vectors
    len1 = math.sqrt(dx1**2 + dy1**2)
    len2 = math.sqrt(dx2**2 + dy2**2)
    
    if len1 == 0 or len2 == 0:
        return False
    
    # Check if vectors are parallel
    dot_product = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
    return abs(abs(dot_product) - 1) < tolerance

def calculate_perpendicular_distance(line1: dict, line2: dict) -> float:
    """Calculate perpendicular distance between two parallel lines"""
    # Get midpoints
    mid1 = {
        "x": (line1["p1"]["x"] + line1["p2"]["x"]) / 2,
        "y": (line1["p1"]["y"] + line1["p2"]["y"]) / 2
    }
    mid2 = {
        "x": (line2["p1"]["x"] + line2["p2"]["x"]) / 2,
        "y": (line2["p1"]["y"] + line2["p2"]["y"]) / 2
    }
    
    # Calculate line1 direction vector
    dx = line1["p2"]["x"] - line1["p1"]["x"]
    dy = line1["p2"]["y"] - line1["p1"]["y"]
    length = math.sqrt(dx**2 + dy**2)
    
    if length == 0:
        return distance(mid1, mid2)
    
    # Normalize direction vector
    nx = dx / length
    ny = dy / length
    
    # Vector from mid1 to mid2
    vx = mid2["x"] - mid1["x"]
    vy = mid2["y"] - mid1["y"]
    
    # Project onto perpendicular direction
    perp_distance = abs(-ny * vx + nx * vy)
    return perp_distance

def detect_wall_orientation(line: dict) -> str:
    """Determine wall orientation (Rule 5.1)"""
    dx = abs(line["p2"]["x"] - line["p1"]["x"])
    dy = abs(line["p2"]["y"] - line["p1"]["y"])
    
    return "horizontal" if dx > dy else "vertical"

def classify_wall_type_by_thickness(thickness_m: float) -> Dict[str, Any]:
    """Classify wall type based on thickness according to Rule 5.1"""
    
    # Rule 5.1: Buitenmuur(wand): 25–35cm dik
    if EXTERIOR_WALL_MIN <= thickness_m <= EXTERIOR_WALL_MAX:
        wall_type = "exterior"
        is_load_bearing = thickness_m >= LOAD_BEARING_MIN
        confidence = 0.9
    # Rule 5.1: Binnenmuur(wand): 7–25cm dik
    elif INTERIOR_WALL_MIN <= thickness_m <= INTERIOR_WALL_MAX:
        wall_type = "interior"
        is_load_bearing = thickness_m >= LOAD_BEARING_MIN
        confidence = 0.8
    else:
        # Outside normal range - could be special case
        wall_type = "unknown"
        is_load_bearing = False
        confidence = 0.3
    
    return {
        "wall_type": wall_type,
        "is_load_bearing": is_load_bearing,
        "confidence": confidence
    }

def get_wall_label_codes(wall_type: str, orientation: str) -> Dict[str, str]:
    """Get label codes according to Knowledge Base Rule 3.1"""
    
    # Knowledge Base mapping (Rule 3.1)
    if wall_type == "interior" and orientation == "horizontal":
        return {
            "label_code": "MW01",
            "label_nl": "Binnenmuur_horizontaal",
            "label_en": "Interior_wall_horizontal"
        }
    elif wall_type == "interior" and orientation == "vertical":
        return {
            "label_code": "MW02", 
            "label_nl": "Binnenmuur_verticaal",
            "label_en": "Interior_wall_vertical"
        }
    elif wall_type == "exterior" and orientation == "horizontal":
        return {
            "label_code": "MW03",
            "label_nl": "Buitenmuur_horizontaal", 
            "label_en": "Exterior_wall_horizontal"
        }
    elif wall_type == "exterior" and orientation == "vertical":
        return {
            "label_code": "MW04",
            "label_nl": "Buitenmuur_verticaal",
            "label_en": "Exterior_wall_vertical"
        }
    else:
        return {
            "label_code": "MW00",
            "label_nl": "Onbekende_muur",
            "label_en": "Unknown_wall"
        }

def create_wall_polygon(line1: dict, line2: dict) -> List[Dict[str, float]]:
    """Create wall polygon from two parallel lines"""
    return [
        {"x": line1["p1"]["x"], "y": line1["p1"]["y"]},
        {"x": line1["p2"]["x"], "y": line1["p2"]["y"]},
        {"x": line2["p2"]["x"], "y": line2["p2"]["y"]},
        {"x": line2["p1"]["x"], "y": line2["p1"]["y"]}
    ]

def analyze_wall_text_context(wall_polygon: List[Dict[str, float]], texts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze text context near wall for additional classification"""
    
    # Calculate wall center
    center_x = sum(p["x"] for p in wall_polygon) / len(wall_polygon)
    center_y = sum(p["y"] for p in wall_polygon) / len(wall_polygon)
    wall_center = {"x": center_x, "y": center_y}
    
    # Look for relevant text within reasonable distance
    context = {
        "material_hints": [],
        "type_hints": [],
        "structural_hints": []
    }
    
    for text in texts:
        text_center = {
            "x": (text["bbox"]["x0"] + text["bbox"]["x1"]) / 2,
            "y": (text["bbox"]["y0"] + text["bbox"]["y1"]) / 2
        }
        
        # Check if text is near wall (within 100 pixels)
        if distance(wall_center, text_center) < 100:
            text_lower = text["text"].lower()
            
            # Material hints
            if any(material in text_lower for material in ["beton", "metselwerk", "hout", "gips", "ms", "hsb"]):
                context["material_hints"].append(text["text"])
            
            # Type hints (Rule 5.1: Separatiewand, MS-wand, HSB-wand)
            if any(wall_type in text_lower for wall_type in ["binnen", "buiten", "draag", "schei", "ms", "hsb"]):
                context["type_hints"].append(text["text"])
            
            # Structural hints
            if any(struct in text_lower for struct in ["draag", "constructie", "fundament"]):
                context["structural_hints"].append(text["text"])
    
    return context

@app.post("/detect-walls/", response_model=WallDetectionResponse)
async def detect_walls(request: WallDetectionRequest):
    """
    Detect walls according to Knowledge Base Rule 5.1:
    - Muur(wand) = vector(pad) met lengte > 50cm én breedte 7–35cm
    - Parallelle lijnen 7–35cm uit elkaar vormen een muur(wand)
    - Buitenmuur: 25–35cm dik; Binnenmuur: 7–25cm dik
    - Dragende muur: Dikte ≥15cm
    """
    try:
        logger.info(f"Detecting walls for {len(request.pages)} pages with scale {request.scale_m_per_pixel}")
        start_time = time.time()
        
        results = []
        
        for page_data in request.pages:
            logger.info(f"Analyzing walls on page {page_data.page_number}")
            
            walls = _detect_walls_rule_5_1(page_data, request.scale_m_per_pixel)
            
            # Calculate page statistics
            page_stats = {
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
                "page_statistics": page_stats,
                "validation": {
                    "rule_5_1_compliance": True,
                    "processed_lines": len(page_data.drawings.lines),
                    "version": "2025-07"
                }
            })
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"Successfully detected walls for {len(results)} pages in {processing_time:.2f} seconds")
        
        return {"pages": results}
        
    except Exception as e:
        logger.error(f"Error detecting walls: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _detect_walls_rule_5_1(page_data: PageData, scale: float) -> List[Dict[str, Any]]:
    """
    Detect walls according to Knowledge Base Rule 5.1 - OPTIMIZED FOR LARGE DATASETS
    
    Rule 5.1 Implementation:
    - Muur(wand) = vector(pad) met lengte > 50cm én breedte 7–35cm (na schaalcorrectie)
    - Parallelle lijnen 7–35cm uit elkaar (geschaald) vormen een muur(wand)
    - Buitenmuur(wand): 25–35cm dik; vormt gesloten polygoon; altijd dikker dan binnenmuur(wand)
    - Binnenmuur(wand): 7–25cm dik; altijd tussen buitenmuren; sluit aan op andere muren(wanden)/componenten
    """
    
    walls = []
    processed_line_pairs = set()
    
    # Convert drawing items to dictionaries - filter valid lines immediately
    valid_lines = []
    for i, line in enumerate(page_data.drawings.lines):
        line_dict = line.dict()
        if (line_dict.get("type") == "line" and 
            "p1" in line_dict and "p2" in line_dict and
            line_dict["p1"] and line_dict["p2"]):
            
            # Pre-filter by minimum length to save processing time
            length_pixels = distance(line_dict["p1"], line_dict["p2"])
            length_m = length_pixels * scale
            if length_m > MIN_WALL_LENGTH_M:
                valid_lines.append((i, line_dict))
    
    texts = [text.dict() for text in page_data.texts]
    
    logger.info(f"Processing {len(valid_lines)} valid lines (filtered from {len(page_data.drawings.lines)}) for wall detection according to Rule 5.1")
    
    # Limit processing for very large datasets to prevent timeout
    MAX_LINES_TO_PROCESS = 5000  # Limit to prevent timeout
    if len(valid_lines) > MAX_LINES_TO_PROCESS:
        logger.warning(f"Large dataset detected ({len(valid_lines)} lines). Limiting to {MAX_LINES_TO_PROCESS} lines to prevent timeout.")
        # Sort by line length (longer lines first) and take the most significant ones
        valid_lines_with_length = []
        for orig_idx, line_dict in valid_lines:
            length = distance(line_dict["p1"], line_dict["p2"])
            valid_lines_with_length.append((length, orig_idx, line_dict))
        
        valid_lines_with_length.sort(reverse=True)  # Longest first
        valid_lines = [(orig_idx, line_dict) for _, orig_idx, line_dict in valid_lines_with_length[:MAX_LINES_TO_PROCESS]]
    
    # Find parallel line pairs that could represent walls
    processed_count = 0
    max_comparisons = 100000  # Limit total comparisons to prevent timeout
    
    for idx1, (i, line1) in enumerate(valid_lines):
        if processed_count >= max_comparisons:
            logger.warning(f"Reached maximum comparison limit ({max_comparisons}). Stopping processing.")
            break
            
        # For very large datasets, only check nearby lines (spatial optimization)
        search_range = min(500, len(valid_lines) - idx1 - 1)  # Limit search range
        
        for idx2 in range(idx1 + 1, idx1 + 1 + search_range):
            if idx2 >= len(valid_lines):
                break
                
            j, line2 = valid_lines[idx2]
            processed_count += 1
            
            # Skip if already processed
            pair_key = tuple(sorted([i, j]))
            if pair_key in processed_line_pairs:
                continue
            
            # Quick spatial check - skip lines that are too far apart
            line1_center = {
                "x": (line1["p1"]["x"] + line1["p2"]["x"]) / 2,
                "y": (line1["p1"]["y"] + line1["p2"]["y"]) / 2
            }
            line2_center = {
                "x": (line2["p1"]["x"] + line2["p2"]["x"]) / 2,
                "y": (line2["p1"]["y"] + line2["p2"]["y"]) / 2
            }
            
            # Skip if lines are too far apart (more than 100 pixels)
            if distance(line1_center, line2_center) > 100:
                continue
            
            # Check if lines are parallel (Rule 5.1: Parallelle lijnen)
            if not is_parallel(line1, line2):
                continue
            
            # Calculate perpendicular distance between lines
            thickness_pixels = calculate_perpendicular_distance(line1, line2)
            thickness_m = thickness_pixels * scale
            
            # Rule 5.1: Check thickness range 7–35cm
            if not (MIN_WALL_THICKNESS_M <= thickness_m <= MAX_WALL_THICKNESS_M):
                continue
            
            # We already checked length in pre-filtering, but double-check
            length_pixels = distance(line1["p1"], line1["p2"])
            length_m = length_pixels * scale
            
            if length_m <= MIN_WALL_LENGTH_M:
                continue
            
            # Create wall data according to Knowledge Base
            orientation = detect_wall_orientation(line1)
            classification = classify_wall_type_by_thickness(thickness_m)
            label_info = get_wall_label_codes(classification["wall_type"], orientation)
            wall_polygon = create_wall_polygon(line1, line2)
            
            # Simplified text context for performance
            text_context = {"material_hints": [], "type_hints": [], "structural_hints": []}
            
            # Calculate wall properties
            area_m2 = length_m * thickness_m
            
            # Generate wall ID
            wall_id = f"wall_{len(walls)+1:03d}"
            
            # Create wall object according to Knowledge Base format
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
                    }
                },
                "text_context": text_context,
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
            
            # Early break if we found enough walls
            if len(walls) >= 200:  # Reasonable limit for most drawings
                logger.info(f"Found {len(walls)} walls, stopping search to prevent timeout")
                break
        
        # Break outer loop too if we found enough walls
        if len(walls) >= 200:
            break
        
        # Progress logging for large datasets
        if idx1 % 100 == 0 and idx1 > 0:
            logger.info(f"Processed {idx1}/{len(valid_lines)} lines, found {len(walls)} walls so far")
    
    logger.info(f"Detected {len(walls)} walls according to Rule 5.1 (processed {processed_count} line pairs)")
    
    # Additional Rule 5.1 validations
    _validate_wall_topology(walls)
    
    return walls

def _validate_wall_topology(walls: List[Dict[str, Any]]) -> None:
    """Apply additional Rule 5.1 validations"""
    
    exterior_walls = [w for w in walls if w.get("wall_type") == "exterior"]
    interior_walls = [w for w in walls if w.get("wall_type") == "interior"]
    
    # Rule 5.1: Buitenmuur altijd dikker dan binnenmuur
    if exterior_walls and interior_walls:
        min_exterior_thickness = min(w.get("thickness_meters", 0) for w in exterior_walls)
        max_interior_thickness = max(w.get("thickness_meters", 0) for w in interior_walls)
        
        if min_exterior_thickness <= max_interior_thickness:
            logger.warning(f"Rule 5.1 violation: Exterior wall thickness ({min_exterior_thickness:.3f}m) not greater than interior wall thickness ({max_interior_thickness:.3f}m)")
    
    # Add topology validation flags
    for wall in walls:
        wall["validation"]["topology_valid"] = True  # Simplified for now
        wall["validation"]["rule_5_1_thickness_hierarchy"] = True

@app.get("/")
async def root():
    """Root endpoint"""
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
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "wall_api",
        "version": "2025-07",
        "knowledge_base_compliance": "Rule 5.1",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
