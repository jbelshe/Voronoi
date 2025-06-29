from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
import numpy as np

@dataclass
class Vertex:
    """Represents a vertex in the Voronoi diagram."""
    x: float
    y: float
    
@dataclass
class Point:
    """Represents an input point that generates the Voronoi cell."""
    x: float
    y: float
    index: int

@dataclass
class Ridge:
    """Represents a ridge (edge) between two Voronoi cells."""
    start_vertex: Vertex
    end_vertex: Vertex
    point_indices: Tuple[int, int]  # Indices of the points that generate this ridge

@dataclass
class Region:
    """Represents a region in the Voronoi diagram."""
    point: int
    vertices: List[int]
    ridges: List[int]


class MyVoronoi:
    """Class representing a Voronoi diagram."""
    
    def __init__(self):
        """Initialize an empty Voronoi diagram."""
        self.vertices: List[Vertex] = []  # List of vertices
        self.ridges: List[Ridge] = []     # List of ridges
        self.points: List[Point] = []     # List of input points
        self.regions: List[Region] = []   # List of regions
        
    def add_point(self, x: float, y: float) -> int:
        """Add a new point to the diagram and return its index."""
        point = Point(x=x, y=y, index=len(self.points))
        self.points.append(point)
        return point.index
        
    def add_vertex(self, x: float, y: float) -> int:
        """Add a new vertex to the diagram and return its index."""
        vertex = Vertex(x=x, y=y)
        self.vertices.append(vertex)
        return len(self.vertices) - 1
        
    def add_ridge(self, start_vertex_idx: int, end_vertex_idx: int, point_indices: Tuple[int, int]) -> int:
        """Add a new ridge between two vertices."""
        ridge = Ridge(
            start_vertex=self.vertices[start_vertex_idx],
            end_vertex=self.vertices[end_vertex_idx],
            point_indices=point_indices
        )
        self.ridges.append(ridge)
        return len(self.ridges) - 1
        
    def get_point(self, index: int) -> Point:
        """Get point by index."""
        return self.points[index]
        
    def get_vertex(self, index: int) -> Vertex:
        """Get vertex by index."""
        return self.vertices[index]
        
    def get_ridge(self, index: int) -> Ridge:
        """Get ridge by index."""
        return self.ridges[index]
        
    def get_ridges_for_point(self, point_index: int) -> List[Ridge]:
        """Get all ridges associated with a given point."""
        return [ridge for ridge in self.ridges if point_index in ridge.point_indices]
