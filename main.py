"""
Advanced Wall Detection API - Professional Construction Analysis
Detects and analyzes structural walls, partitions, and building elements from construction drawings
Optimized for complex architectural and engineering plans
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import math
from typing import List, Dict, Any, Optional
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wall_api")

app = FastAPI(
    title="Advanced Wall Detection API",
    description="Professional wall detection and structural analysis for construction drawings",
    version="2.0.0",
)

class PageData(BaseModel):
    page_number: int
    drawings: List[Dict[str, Any]]
    texts: List[Dict[str, Any]]
    dimensions: List[Dict[str, Any]]
    room_labels: List[Dict[str, Any]]
    annotations: List[Dict[str, Any]]

class WallDetectionRequest(BaseModel):
    pages: List[PageData]
    scale_m_per_pixel: float = 1.0

def distance(p1: dict, p2: dict) -> float:
    """Calculate distance between two points"""
    return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)

def is_parallel(line1: dict, line2: dict, tolerance: float = 5.0) -> bool:
    """Check if two lines are parallel within tolerance"""
    dx1 = line1['p2']['x'] - line1['p1']['x']
    dy1 = line1['p2']['y'] - line1['p1']['y']
    dx2 = line2['p2']['x'] - line2['p1']['x']
    dy2 = line2['p2']['y'] - line2['p1']['y']
    
    if dx1 == 0 and dx2 == 0:
        return True
    if dx1 == 0 or dx2 == 0:
        return False
    
    angle1 = math.degrees(math.atan2(dy1, dx1))
    angle2 = math.degrees(math.atan2(dy2, dx2))
    
    angle_diff = abs(angle1 - angle2)
    return angle_diff < tolerance or abs(angle_diff - 180) < tolerance

def classify_wall_type(thickness_m: float, length_m: float, annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Classify wall type based on thickness, length, and nearby annotations"""
    
    # Default classification
    wall_type = "unknown"
    material = "unknown"
    structural_type = "unknown"
    confidence = 0.5
    
    # Classify by thickness
    if thickness_m >= 0.30:  # 30cm+
        wall_type = "exterior_wall"
        material = "masonry_concrete"
        structural_type = "load_bearing"
        confidence = 0.9
    elif thickness_m >= 0.15:  # 15-30cm
        wall_type = "interior_wall"
        material = "masonry"
        structural_type = "load_bearing"
        confidence = 0.8
    elif thickness_m >= 0.10:  # 10-15cm
        wall_type = "partition_wall"
        material = "masonry_plasterboard"
        structural_type = "non_load_bearing"
        confidence = 0.7
    elif thickness_m >= 0.05:  # 5-10cm
        wall_type = "light_partition"
        material = "plasterboard"
        structural_type = "non_load_bearing"
        confidence = 0.6
    
    # Check for material annotations nearby
    for annotation in annotations:
        if annotation.get("text_type") == "construction_note":
            text = annotation["text"].lower()
            if any(word in text for word in ["beton", "concrete"]):
                material = "concrete"
                confidence = min(confidence + 0.2, 1.0)
            elif any(word in text for word in ["hout", "wood", "timber"]):
                material = "wood"
                confidence = min(confidence + 0.2, 1.0)
            elif any(word in text for word in ["staal", "steel"]):
                material = "steel"
                confidence = min(confidence + 0.2, 1.0)
    
    return {
        "wall_type": wall_type,
        "material": material,
        "structural_type": structural_type,
        "confidence": confidence
    }

def calculate_wall_properties(p1: dict, p2: dict, thickness_m: float, scale: float) -> Dict[str, Any]:
    """Calculate comprehensive wall properties"""
    
    # Basic properties
    length_px = distance(p1, p2)
    length_m = length_px * scale
    area_m2 = length_m * thickness_m
    
    # Orientation
    dx = p2['x'] - p1['x']
    dy = p2['y'] - p1['y']
    angle_degrees = math.degrees(math.atan2(dy, dx))
    
    # Normalize angle to 0-180 degrees
    if angle_degrees < 0:
        angle_degrees += 180
    
    # Determine orientation
    if angle_degrees < 15 or angle_degrees > 165:
        orientation = "horizontal"
    elif 75 < angle_degrees < 105:
        orientation = "vertical"
    else:
        orientation = "diagonal"
    
    # Calculate center point
    center_x = (p1['x'] + p2['x']) / 2
    center_y = (p1['y'] + p2['y']) / 2
    
    return {
        "length_pixels": length_px,
        "length_meters": length_m,
        "thickness_meters": thickness_m,
        "area_sqm": area_m2,
        "angle_degrees": angle_degrees,
        "orientation": orientation,
        "center_point": {"x": center_x, "y": center_y}
    }

@app.post("/detect-walls/")
async def detect_walls(request: WallDetectionRequest):
    """
    Advanced wall detection for construction drawings
    
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
            
            walls = _detect_advanced_walls(page_data, request.scale_m_per_pixel)
            
            # Calculate page statistics
            page_area = sum(wall.get("properties", {}).get("area_sqm", 0) for wall in walls)
            total_walls += len(walls)
            total_area += page_area
            
            results.append({
                "page_number": page_data.page_number,
                "walls": walls,
                "page_statistics": {
                    "total_walls": len(walls),
                    "total_wall_area_sqm": page_area,
                    "average_wall_length": sum(wall.get("properties", {}).get("length_meters", 0) for wall in walls) / max(len(walls), 1),
                    "average_wall_thickness": sum(wall.get("properties", {}).get("thickness_meters", 0) for wall in walls) / max(len(walls), 1)
                }
            })
        
        logger.info(f"Successfully detected {total_walls} walls across {len(results)} pages")
        logger.info(f"Total wall area: {total_area:.2f} m²")
        
        return {
            "pages": results,
            "summary": {
                "total_pages": len(results),
                "total_walls": total_walls,
                "total_wall_area_sqm": total_area,
                "average_walls_per_page": total_walls / len(results) if results else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error detecting walls: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _detect_advanced_walls(page_data: PageData, scale: float) -> List[Dict[str, Any]]:
    """
    Advanced wall detection using multiple analysis techniques
    
    Args:
        page_data: Page data containing drawings, texts, and annotations
        scale: Scale factor in meters per pixel
        
    Returns:
        List of detected walls with comprehensive properties
    """
    walls = []
    potential_walls = []
    
    # Step 1: Extract potential wall lines from drawings
    for drawing in page_data.drawings:
        if drawing.get("construction_type") in ["wall_structure", "unknown"]:
            for item in drawing["items"]:
                if item["type"] == "line":
                    p1, p2 = item["p1"], item["p2"]
                    
                    # Calculate basic properties
                    thickness_m = drawing["width"] * scale
                    length_px = distance(p1, p2)
                    length_m = length_px * scale
                    
                    # Filter by minimum requirements
                    if length_m < 0.5:  # Minimum 50cm wall length
                        continue
                    if thickness_m < 0.05:  # Minimum 5cm wall thickness
                        continue
                    
                    # Calculate comprehensive properties
                    properties = calculate_wall_properties(p1, p2, thickness_m, scale)
                    
                    # Classify wall type
                    classification = classify_wall_type(thickness_m, length_m, page_data.annotations)
                    
                    wall = {
                        "p1": p1,
                        "p2": p2,
                        "properties": properties,
                        "classification": classification,
                        "detection_method": "line_analysis",
                        "confidence": classification["confidence"]
                    }
                    
                    potential_walls.append(wall)
    
    # Step 2: Group parallel walls (double lines)
    grouped_walls = _group_parallel_walls(potential_walls)
    
    # Step 3: Analyze wall connections and patterns
    for wall in grouped_walls:
        # Check for nearby room labels
        nearby_rooms = _find_nearby_rooms(wall, page_data.room_labels)
        
        # Check for nearby dimensions
        nearby_dimensions = _find_nearby_dimensions(wall, page_data.dimensions)
        
        # Add contextual information
        wall["context"] = {
            "nearby_rooms": nearby_rooms,
            "nearby_dimensions": nearby_dimensions,
            "wall_connections": _find_wall_connections(wall, grouped_walls)
        }
        
        walls.append(wall)
    
    # Step 4: Detect wall openings (doors, windows)
    openings = _detect_wall_openings(walls, page_data.drawings, scale)
    
    # Add openings to walls
    for wall in walls:
        wall["openings"] = [op for op in openings if _is_opening_in_wall(op, wall)]
    
    if not walls:
        logger.warning(f"No walls detected on page {page_data.page_number}")
        return [{
            "type": "unknown", 
            "reason": "No structural walls found", 
            "confidence": 0.0
        }]
    
    logger.info(f"Detected {len(walls)} walls on page {page_data.page_number}")
    return walls

def _group_parallel_walls(walls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group parallel walls that might represent double lines"""
    grouped = []
    processed = set()
    
    for i, wall1 in enumerate(walls):
        if i in processed:
            continue
            
        group = [wall1]
        processed.add(i)
        
        for j, wall2 in enumerate(walls[i+1:], i+1):
            if j in processed:
                continue
                
            if is_parallel(wall1, wall2, tolerance=10.0):
                # Check if walls are close to each other (double line)
                center1 = wall1["properties"]["center_point"]
                center2 = wall2["properties"]["center_point"]
                distance_between = distance(center1, center2)
                
                if distance_between < 20:  # Close enough to be double line
                    group.append(wall2)
                    processed.add(j)
        
        # If we have multiple walls in group, merge them
        if len(group) > 1:
            merged_wall = _merge_wall_group(group)
            grouped.append(merged_wall)
        else:
            grouped.append(wall1)
    
    return grouped

def _merge_wall_group(walls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge a group of parallel walls into a single wall"""
    # Use the wall with highest confidence as base
    base_wall = max(walls, key=lambda w: w["confidence"])
    
    # Calculate average properties
    avg_thickness = sum(w["properties"]["thickness_meters"] for w in walls) / len(walls)
    avg_length = sum(w["properties"]["length_meters"] for w in walls) / len(walls)
    
    # Update properties
    base_wall["properties"]["thickness_meters"] = avg_thickness
    base_wall["properties"]["length_meters"] = avg_length
    base_wall["properties"]["area_sqm"] = avg_length * avg_thickness
    
    # Update classification
    base_wall["classification"] = classify_wall_type(avg_thickness, avg_length, [])
    base_wall["detection_method"] = "double_line_analysis"
    base_wall["confidence"] = min(base_wall["confidence"] + 0.1, 1.0)
    
    return base_wall

def _find_nearby_rooms(wall: Dict[str, Any], room_labels: List[Dict[str, Any]], max_distance: float = 100) -> List[str]:
    """Find room labels near the wall"""
    nearby = []
    wall_center = wall["properties"]["center_point"]
    
    for room in room_labels:
        room_center = {
            "x": (room["bbox"]["x0"] + room["bbox"]["x1"]) / 2,
            "y": (room["bbox"]["y0"] + room["bbox"]["y1"]) / 2
        }
        
        if distance(wall_center, room_center) < max_distance:
            nearby.append(room["text"])
    
    return nearby

def _find_nearby_dimensions(wall: Dict[str, Any], dimensions: List[Dict[str, Any]], max_distance: float = 50) -> List[Dict[str, Any]]:
    """Find dimensions near the wall"""
    nearby = []
    wall_center = wall["properties"]["center_point"]
    
    for dim in dimensions:
        dim_center = {
            "x": (dim["bbox"]["x0"] + dim["bbox"]["x1"]) / 2,
            "y": (dim["bbox"]["y0"] + dim["bbox"]["y1"]) / 2
        }
        
        if distance(wall_center, dim_center) < max_distance:
            nearby.append(dim)
    
    return nearby

def _find_wall_connections(wall: Dict[str, Any], all_walls: List[Dict[str, Any]], max_distance: float = 10) -> List[int]:
    """Find walls that connect to this wall"""
    connections = []
    wall_endpoints = [wall["p1"], wall["p2"]]
    
    for i, other_wall in enumerate(all_walls):
        if other_wall == wall:
            continue
            
        other_endpoints = [other_wall["p1"], other_wall["p2"]]
        
        # Check if any endpoints are close
        for ep1 in wall_endpoints:
            for ep2 in other_endpoints:
                if distance(ep1, ep2) < max_distance:
                    connections.append(i)
                    break
            if i in connections:
                break
    
    return connections

def _detect_wall_openings(walls: List[Dict[str, Any]], drawings: List[Dict[str, Any]], scale: float) -> List[Dict[str, Any]]:
    """Detect doors and windows in walls"""
    openings = []
    
    for drawing in drawings:
        if drawing.get("construction_type") == "opening":
            for item in drawing["items"]:
                if item["type"] == "rect":
                    rect = item["rect"]
                    
                    # Calculate opening properties
                    width_m = rect["width"] * scale
                    height_m = rect["height"] * scale
                    area_m2 = width_m * height_m
                    
                    # Classify opening type
                    if width_m > 0.8 and height_m > 2.0:  # Door
                        opening_type = "door"
                    elif width_m > 0.6 and height_m > 1.0:  # Window
                        opening_type = "window"
                    else:
                        opening_type = "opening"
                    
                    opening = {
                        "type": opening_type,
                        "rect": rect,
                        "properties": {
                            "width_meters": width_m,
                            "height_meters": height_m,
                            "area_sqm": area_m2
                        },
                        "center_point": {
                            "x": (rect["x0"] + rect["x1"]) / 2,
                            "y": (rect["y0"] + rect["y1"]) / 2
                        }
                    }
                    
                    openings.append(opening)
    
    return openings

def _is_opening_in_wall(opening: Dict[str, Any], wall: Dict[str, Any], tolerance: float = 20) -> bool:
    """Check if an opening is located in a wall"""
    wall_center = wall["properties"]["center_point"]
    opening_center = opening["center_point"]
    
    return distance(wall_center, opening_center) < tolerance

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "wall-api", "version": "2.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 