"""
Wall API - Detects structural walls from extracted vector data
Analyzes line thickness and patterns to identify walls
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import math
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wall_api")

app = FastAPI(
    title="Wall Detection API",
    description="Detects structural walls from extracted vector data",
    version="1.0.0",
)

class PageData(BaseModel):
    page_number: int
    drawings: List[Dict[str, Any]]
    texts: List[Dict[str, Any]]

class WallDetectionRequest(BaseModel):
    pages: List[PageData]
    scale_m_per_pixel: float = 1.0

@app.post("/detect-walls/")
async def detect_walls(request: WallDetectionRequest):
    """
    Detect structural walls from extracted vector data
    
    Args:
        request: JSON with pages containing drawings and scale information
        
    Returns:
        JSON with detected walls for each page
    """
    try:
        logger.info(f"Detecting walls for {len(request.pages)} pages with scale {request.scale_m_per_pixel}")
        
        results = []
        
        for page_data in request.pages:
            logger.info(f"Analyzing walls on page {page_data.page_number}")
            
            walls = _detect_structural_walls(page_data, request.scale_m_per_pixel)
            
            results.append({
                "page_number": page_data.page_number,
                "walls": walls
            })
        
        logger.info(f"Successfully detected walls for {len(results)} pages")
        return {"pages": results}
        
    except Exception as e:
        logger.error(f"Error detecting walls: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _detect_structural_walls(page_data: PageData, scale: float) -> List[Dict[str, Any]]:
    """
    Detect structural walls from page data using rule-based approach
    
    Args:
        page_data: Page data containing drawings
        scale: Scale factor in meters per pixel
        
    Returns:
        List of detected walls with properties
    """
    walls = []
    
    for drawing in page_data.drawings:
        for item in drawing["items"]:
            if item["type"] == "line":
                p1, p2 = item["p1"], item["p2"]
                
                # Calculate line length in pixels and meters
                dx, dy = p2["x"] - p1["x"], p2["y"] - p1["y"]
                length_px = math.hypot(dx, dy)
                length_m = length_px * scale
                
                # Skip very short lines
                if length_m < 0.0001:
                    continue
                
                # Calculate wall thickness in meters
                thickness_m = drawing["width"] * scale
                
                # Filter by minimum wall thickness
                if thickness_m < 0.01:
                    continue
                
                wall = {
                    "p1": p1,
                    "p2": p2,
                    "wall_thickness": thickness_m,
                    "wall_length": length_m,
                    "wall_type": "solid",
                    "confidence": 1.0,
                    "reason": "Thick line detected as wall"
                }
                walls.append(wall)
    
    if not walls:
        logger.warning(f"No walls detected on page {page_data.page_number}")
        return [{
            "type": "unknown", 
            "reason": "No thick, parallel lines found", 
            "confidence": 0.0
        }]
    
    logger.info(f"Detected {len(walls)} walls on page {page_data.page_number}")
    return walls

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "wall-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 