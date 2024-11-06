import numpy as np
from typing import Dict

def get_zone_vector() -> Dict[str, np.ndarray]:
    """Create zone/goal representations for Unity environment"""
    zone_vectors = {}
    base_vectors = {
        'green': np.array([1, 0, 0]),  # For GreenPlus
        'red': np.array([0, 1, 0]),    # For RedEx
        'yellow': np.array([0, 0, 1])  # For YellowStar
    }
    
    # Extend to 24-dimensional vectors (8 sets of 3D vectors)
    for zone, base in base_vectors.items():
        zone_vectors[zone] = np.tile(base, 8)
        zone_vectors[zone] = zone_vectors[zone] / np.linalg.norm(zone_vectors[zone])
        
    return zone_vectors
