"""
Advanced Wall Detection API - Knowledge Base Implementation
Implements complete knowledge base for wall detection (Rule 5.1)
Detects walls based on parallel lines with thickness 0.07m - 5.5m
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import math
from typing import List, Dict, Any, Optional
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wall_api")

# Knowledge Base Constants (Rule 5.1)
WALL_THICKNESS_MIN = 0.07  # meters - minimum wall thickness
WALL_THICKNESS_MAX = 5.5   # meters - maximum wall thickness
MIN_LINE_LENGTH = 0.5      # meters - minimum length to be considered a wall
PARALLEL_TOLERANCE = 0.1   # radians - tolerance for parallel lines

# Wall Type Patterns
WALL_TYPE_PATTERNS = {
    "exterior_wall": [
        r"buitenmuur|exterior|external|outer",
        r"gevel|facade|front|rear|side"
    ],
    "interior_wall": [
        r"binnenmuur|interior|internal|inner",
        r"scheidingswand|partition|divider"
    ],
    "load_bearing": [
        r"draagmuur|load.bearing|structural|bearing",
        r"constructie|structure|fundering"
    ],
    "non_load_bearing": [
        r"niet.dragend|non.load.bearing|partition",
        r"scheidingswand|divider|screen"
    ]
}

app = FastAPI(
    title="Advanced Wall Detection API",
    description="Professional wall detection using knowledge base (Rule 5.1)",
    version="3.0.0",
)

class DrawingItem(BaseModel):
    type: str
    p1: Dict[str, float] = None
    p2: Dict[str, float] = None
    p3: Dict[str, float] = None
    rect: Dict[str, float] = None
    length: Optional[float] = None
    color: List[float] = [0, 0, 0]
    width: Optional[float] = 1.0
    area: Optional[float] = None
    fill: List[Any] = []

class Drawings(BaseModel):
    lines: List[DrawingItem]
    rectangles: List[DrawingItem]
    curves: List[DrawingItem]

class TextItem(BaseModel):
    text: str
    position: Dict[str, float]
    font_size: float
    font_name: str
    color: List[float] = [0, 0, 0]
    bbox: Dict[str, float]

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
    summary: Dict[str, Any]

def distance(p1: dict, p2: dict) -> float:
    """Calculate distance between two points"""
    return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)

def is_parallel(line1: dict, line2: dict, tolerance: float = PARALLEL_TOLERANCE) -> bool:
    """Check if two lines are parallel (Rule 5.1)"""
    if line1["type"] != "line" or line2["type"] != "line":
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
    
    # Check if vectors are parallel (dot product close to 1 or -1)
    dot_product = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
    return abs(abs(dot_product) - 1) < tolerance

def calculate_wall_thickness(line1: dict, line2: dict, scale: float) -> float:
    """Calculate wall thickness between two parallel lines (Rule 5.1)"""
    if not is_parallel(line1, line2):
        return 0
    
    # Calculate distance between line midpoints
    mid1 = {
        "x": (line1["p1"]["x"] + line1["p2"]["x"]) / 2,
        "y": (line1["p1"]["y"] + line1["p2"]["y"]) / 2
    }
    mid2 = {
        "x": (line2["p1"]["x"] + line2["p2"]["x"]) / 2,
        "y": (line2["p1"]["y"] + line2["p2"]["y"]) / 2
    }
    
    thickness_pixels = distance(mid1, mid2)
    thickness_meters = thickness_pixels * scale
    
    return thickness_meters

def detect_wall_orientation(line: dict) -> str:
    """Determine wall orientation (Rule 5.1)"""
    dx = line["p2"]["x"] - line["p1"]["x"]
    dy = line["p2"]["y"] - line["p1"]["y"]
    
    if abs(dx) > abs(dy):
        return "horizontal"
    else:
        return "vertical"

def classify_wall_type(thickness_m: float, orientation: str, texts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Classify wall type based on thickness and context (Rule 5.1)"""
    
    # Default classification
    wall_type = "interior"
    material = "unknown"
    structural_type = "unknown"
    confidence = 0.5
    
    # Classify by thickness (Rule 5.1)
    if thickness_m >= 0.30:  # 30cm+
        wall_type = "exterior"
        material = "masonry_concrete"
        structural_type = "load_bearing"
        confidence = 0.9
    elif thickness_m >= 0.15:  # 15-30cm
        wall_type = "interior"
        material = "masonry"
        structural_type = "load_bearing"
        confidence = 0.8
    elif thickness_m >= 0.10:  # 10-15cm
        wall_type = "interior"
        material = "masonry_plasterboard"
        structural_type = "non_load_bearing"
        confidence = 0.7
    elif thickness_m >= 0.05:  # 5-10cm
        wall_type = "interior"
        material = "plasterboard"
        structural_type = "non_load_bearing"
        confidence = 0.6
    
    # Check for wall type annotations in nearby text
    for text in texts:
        text_lower = text["text"].lower()
        text_pos = text["position"]
        
        # Check if text matches wall type patterns
        for pattern_type, patterns in WALL_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    if pattern_type == "exterior_wall":
                        wall_type = "exterior"
                        confidence = min(confidence + 0.2, 1.0)
                    elif pattern_type == "interior_wall":
                        wall_type = "interior"
                        confidence = min(confidence + 0.2, 1.0)
                    elif pattern_type == "load_bearing":
                        structural_type = "load_bearing"
                        confidence = min(confidence + 0.2, 1.0)
                    elif pattern_type == "non_load_bearing":
                        structural_type = "non_load_bearing"
                        confidence = min(confidence + 0.2, 1.0)
                    break
    
    return {
        "wall_type": wall_type,
        "material": material,
        "structural_type": structural_type,
        "confidence": confidence
    }

def calculate_wall_properties(line1: dict, line2: dict, thickness_m: float, scale: float) -> Dict[str, Any]:
    """Calculate comprehensive wall properties"""
    
    # Use line1 for length calculation
    length_px = distance(line1["p1"], line1["p2"])
    length_m = length_px * scale
    
    # Check minimum length requirement (Rule 5.1)
    if length_m < MIN_LINE_LENGTH:
        return None
    
    area_m2 = length_m * thickness_m
    
    # Orientation
    orientation = detect_wall_orientation(line1)
    
    # Calculate center point
    center_x = (line1["p1"]["x"] + line1["p2"]["x"] + line2["p1"]["x"] + line2["p2"]["x"]) / 4
    center_y = (line1["p1"]["y"] + line1["p2"]["y"] + line2["p1"]["y"] + line2["p2"]["y"]) / 4
    
    # Calculate wall polygon
    polygon = [
        {"x": line1["p1"]["x"], "y": line1["p1"]["y"]},
        {"x": line1["p2"]["x"], "y": line1["p2"]["y"]},
        {"x": line2["p2"]["x"], "y": line2["p2"]["y"]},
        {"x": line2["p1"]["x"], "y": line2["p1"]["y"]}
    ]
    
    return {
        "length_pixels": length_px,
        "length_meters": length_m,
        "thickness_meters": thickness_m,
        "area_sqm": area_m2,
        "orientation": orientation,
        "center_point": {"x": center_x, "y": center_y},
        "polygon": polygon
    }

@app.post("/detect-walls/", response_model=WallDetectionResponse)
async def detect_walls(request: WallDetectionRequest):
    """
    Advanced wall detection using knowledge base (Rule 5.1)
    
    Args:
        request: JSON with pages containing drawings, texts, and scale information
        
    Returns:
        JSON with comprehensive wall analysis for each page
    """
    try:
        logger.info(f"Detecting walls for {len(request.pages)} pages with scale {request.scale_m_per_pixel}")
        
        results = []
        total_walls = 0
        total_area = 0.0
        
        for page_data in request.pages:
            logger.info(f"Analyzing walls on page {page_data.page_number}")
            
            walls = _detect_walls_knowledge_base(page_data, request.scale_m_per_pixel)
            
            # Calculate page statistics
            page_area = sum(wall.get("properties", {}).get("area_sqm", 0) for wall in walls)
            total_walls += len(walls)
            total_area += page_area
            
            results.append({
                "page_number": page_data.page_number,
                "walls": walls,
                "page_statistics": {
                    "total_walls": len(walls),
                    "total_wall_area_sqm": round(page_area, 2),
                    "average_wall_length": round(sum(wall.get("properties", {}).get("length_meters", 0) for wall in walls) / max(len(walls), 1), 2),
                    "average_wall_thickness": round(sum(wall.get("properties", {}).get("thickness_meters", 0) for wall in walls) / max(len(walls), 1), 2)
                }
            })
        
        logger.info(f"Successfully detected {total_walls} walls across {len(results)} pages")
        logger.info(f"Total wall area: {total_area:.2f} m²")
        
        return {
            "pages": results,
            "summary": {
                "total_pages": len(results),
                "total_walls": total_walls,
                "total_wall_area_sqm": round(total_area, 2),
                "average_walls_per_page": round(total_walls / len(results) if results else 0, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Error detecting walls: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _detect_walls_knowledge_base(page_data: PageData, scale: float) -> List[Dict[str, Any]]:
    """
    Detect walls using knowledge base rules (Rule 5.1)
    
    Args:
        page_data: Page data containing drawings and texts
        scale: Scale factor in meters per pixel
        
    Returns:
        List of detected walls with properties
    """
    walls = []
    lines = page_data.drawings.lines
    texts = page_data.texts
    processed_lines = set()
    
    logger.info(f"Analyzing {len(lines)} lines for wall detection")
    
    # Find parallel line pairs that could represent walls
    for i, line1 in enumerate(lines):
        if i in processed_lines:
            continue
            
        for j, line2 in enumerate(lines[i+1:], i+1):
            if j in processed_lines:
                continue
            
            # Calculate wall thickness
            thickness_m = calculate_wall_thickness(line1.dict(), line2.dict(), scale)
            
            # Check if thickness is within wall range (Rule 5.1)
            if WALL_THICKNESS_MIN <= thickness_m <= WALL_THICKNESS_MAX:
                # Calculate wall properties
                properties = calculate_wall_properties(line1.dict(), line2.dict(), thickness_m, scale)
                
                if properties is None:
                    continue  # Skip if minimum length not met
                
                # Classify wall type
                classification = classify_wall_type(thickness_m, properties["orientation"], [t.dict() for t in texts])
                
                # Generate label codes based on classification (Rule 3.1)
                label_code = "MW01"  # Default
                if classification["wall_type"] == "interior" and properties["orientation"] == "horizontal":
                    label_code = "MW01"
                    label_nl = "Binnenmuur_horizontaal"
                    label_en = "Interior_wall_horizontal"
                elif classification["wall_type"] == "interior" and properties["orientation"] == "vertical":
                    label_code = "MW02"
                    label_nl = "Binnenmuur_verticaal"
                    label_en = "Interior_wall_vertical"
                elif classification["wall_type"] == "exterior" and properties["orientation"] == "horizontal":
                    label_code = "MW03"
                    label_nl = "Buitenmuur_horizontaal"
                    label_en = "Exterior_wall_horizontal"
                elif classification["wall_type"] == "exterior" and properties["orientation"] == "vertical":
                    label_code = "MW04"
                    label_nl = "Buitenmuur_verticaal"
                    label_en = "Exterior_wall_vertical"
                
                # Create wall data
                wall_data = {
                    "type": f"{classification['wall_type']}_wall_{properties['orientation']}",
                    "label_code": label_code,
                    "label_nl": label_nl,
                    "label_en": label_en,
                    "label_type": "construction",
                    "thickness_meters": round(thickness_m, 3),
                    "properties": properties,
                    "classification": classification,
                    "line1_index": i,
                    "line2_index": j,
                    "orientation": properties["orientation"],
                    "wall_type": classification["wall_type"],
                    "confidence": classification["confidence"],
                    "reason": f"{classification['wall_type'].title()} wall detected with thickness {thickness_m:.3f}m"
                }
                
                walls.append(wall_data)
                processed_lines.add(i)
                processed_lines.add(j)
                break
    
    logger.info(f"Detected {len(walls)} walls using knowledge base rules")
    return walls

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Advanced Wall Detection API - Knowledge Base Implementation",
        "version": "3.0.0",
        "endpoints": {
            "/detect-walls/": "Detect walls using knowledge base (Rule 5.1)",
            "/health/": "Health check"
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "wall_api",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)