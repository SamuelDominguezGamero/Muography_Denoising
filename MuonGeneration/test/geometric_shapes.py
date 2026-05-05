"""
Geometric shape generator for Geant4 geometry creation.
Generates 2D numpy arrays representing filled or hollow geometric shapes.
Each shape is a binary matrix (1 for filled, 0 for empty).
"""

import numpy as np
from scipy import ndimage
import warnings


def create_circle(radius: int, filled: bool = True, stroke: int = 1) -> np.ndarray:
    """
    Creates a 2D matrix representing a circle.
    
    Args:
        radius: Circle radius in pixels (defines the size)
        filled: If True, creates a filled circle; if False, creates an outline
        stroke: Thickness of the circle outline (for filled=False). Ignored when filled=True.
    
    Returns:
        2D numpy array (matrix) with 1s for the circle and 0s for the background
        Shape is (diameter, diameter) where diameter = 2*radius + 1
    """
    diameter = 2 * radius + 1
    
    # Create coordinate grids centered at (radius, radius)
    y, x = np.ogrid[:diameter, :diameter]
    center = radius
    
    # Distance from center
    distance = np.sqrt((x - center)**2 + (y - center)**2)
    
    if filled:
        # Filled circle: all pixels within radius
        circle_matrix = (distance <= radius).astype(int)
    else:
        # Outline: pixels within stroke distance from the border
        outer = distance <= radius
        inner = distance <= (radius - stroke)
        circle_matrix = (outer & ~inner).astype(int)
    
    return circle_matrix


def create_triangle(size: int, filled: bool = True, stroke: int = 1) -> np.ndarray:
    """
    Creates a 2D matrix representing an equilateral triangle.
    
    Args:
        size: Height of the triangle in pixels
        filled: If True, creates a filled triangle; if False, creates an outline
        stroke: Thickness of the triangle outline (for filled=False)
    
    Returns:
        2D numpy array representing the triangle, centered horizontally
        Shape is (size, size) where the base width ≈ size
    """
    # For an equilateral triangle inscribed in a square
    # Base width ≈ size, height = size
    # Vertices: top center, bottom-left, bottom-right
    
    matrix = np.zeros((size, size), dtype=int)
    
    # Vertices of an equilateral triangle centered in the square
    # Top vertex
    top_y = 0
    top_x = size // 2
    
    # Bottom vertices
    bottom_y = size - 1
    bottom_left_x = 0
    bottom_right_x = size - 1
    
    # Create the triangle by checking which pixels are inside
    for y in range(size):
        for x in range(size):
            # Barycentric coordinates to check if point is inside triangle
            # Using cross product method
            if is_point_in_triangle(x, y, top_x, top_y, bottom_left_x, bottom_y, bottom_right_x, bottom_y):
                matrix[y, x] = 1
    
    if not filled:
        # Create outline by keeping only edge pixels
        matrix = extract_outline(matrix, stroke)
    
    return matrix


def create_rectangle(width: int, height: int, filled: bool = True, stroke: int = 1) -> np.ndarray:
    """
    Creates a 2D matrix representing a rectangle.
    
    Args:
        width: Rectangle width in pixels
        height: Rectangle height in pixels
        filled: If True, creates a filled rectangle; if False, creates an outline
        stroke: Thickness of the rectangle outline (for filled=False)
    
    Returns:
        2D numpy array representing the rectangle
        Shape is (height, width)
    """
    if filled:
        # Filled rectangle: all 1s
        rectangle_matrix = np.ones((height, width), dtype=int)
    else:
        # Outline: 1s only at the borders
        rectangle_matrix = np.zeros((height, width), dtype=int)
        
        # Top and bottom borders
        rectangle_matrix[0:stroke, :] = 1
        rectangle_matrix[height-stroke:height, :] = 1
        
        # Left and right borders
        rectangle_matrix[:, 0:stroke] = 1
        rectangle_matrix[:, width-stroke:width] = 1
    
    return rectangle_matrix


def create_star(size: int, points: int = 5, filled: bool = True, stroke: int = 1) -> np.ndarray:
    """
    Creates a 2D matrix representing a star polygon.
    
    Args:
        size: Size of the bounding square (size x size)
        points: Number of star points (default 5 for pentagram)
        filled: If True, creates a filled star; if False, creates an outline
        stroke: Thickness of the star outline (for filled=False)
    
    Returns:
        2D numpy array representing the star
        Shape is (size, size)
    """
    matrix = np.zeros((size, size), dtype=int)
    center = size / 2.0
    
    # Outer radius and inner radius for the star
    outer_radius = size / 2.0 - 1
    inner_radius = outer_radius * 0.4
    
    # Generate star vertices
    vertices = []
    for i in range(2 * points):
        angle = (i * np.pi / points) - (np.pi / 2)  # Start from top
        
        if i % 2 == 0:  # Outer points
            radius = outer_radius
        else:  # Inner points
            radius = inner_radius
        
        x = center + radius * np.cos(angle)
        y = center + radius * np.sin(angle)
        vertices.append((x, y))
    
    # Fill the star using a simple point-in-polygon check
    for y in range(size):
        for x in range(size):
            if is_point_in_polygon(x, y, vertices):
                matrix[y, x] = 1
    
    if not filled:
        # Create outline
        matrix = extract_outline(matrix, stroke)
    
    return matrix


def create_polygon(size: int, sides: int, filled: bool = True, stroke: int = 1) -> np.ndarray:
    """
    Creates a 2D matrix representing a regular polygon.
    
    Args:
        size: Size of the bounding square (size x size)
        sides: Number of sides (3=triangle, 4=square, 6=hexagon, etc.)
        filled: If True, creates a filled polygon; if False, creates an outline
        stroke: Thickness of the polygon outline (for filled=False)
    
    Returns:
        2D numpy array representing the polygon
        Shape is (size, size)
    """
    matrix = np.zeros((size, size), dtype=int)
    center = size / 2.0
    radius = size / 2.0 - 1
    
    # Generate polygon vertices
    vertices = []
    for i in range(sides):
        angle = (2 * np.pi * i / sides) - (np.pi / 2)  # Start from top
        x = center + radius * np.cos(angle)
        y = center + radius * np.sin(angle)
        vertices.append((x, y))
    
    # Fill the polygon
    for y in range(size):
        for x in range(size):
            if is_point_in_polygon(x, y, vertices):
                matrix[y, x] = 1
    
    if not filled:
        # Create outline
        matrix = extract_outline(matrix, stroke)
    
    return matrix


def create_diamond(size: int, filled: bool = True, stroke: int = 1) -> np.ndarray:
    """
    Creates a 2D matrix representing a diamond (rotated square).
    
    Args:
        size: Size of the bounding square
        filled: If True, creates a filled diamond; if False, creates an outline
        stroke: Thickness of the diamond outline (for filled=False)
    
    Returns:
        2D numpy array representing the diamond
        Shape is (size, size)
    """
    # Diamond is a regular polygon with 4 sides rotated 45 degrees
    matrix = np.zeros((size, size), dtype=int)
    center = size / 2.0
    radius = size / 2.0 - 1
    
    # Generate diamond vertices (square rotated 45 degrees)
    vertices = []
    for i in range(4):
        angle = (np.pi / 2) * i  # 0, π/2, π, 3π/2
        x = center + radius * np.cos(angle)
        y = center + radius * np.sin(angle)
        vertices.append((x, y))
    
    # Fill the diamond
    for y in range(size):
        for x in range(size):
            if is_point_in_polygon(x, y, vertices):
                matrix[y, x] = 1
    
    if not filled:
        # Create outline
        matrix = extract_outline(matrix, stroke)
    
    return matrix


# ============= HELPER FUNCTIONS =============

def is_point_in_triangle(x: float, y: float, 
                         x1: float, y1: float,
                         x2: float, y2: float,
                         x3: float, y3: float) -> bool:
    """
    Check if a point (x, y) is inside a triangle defined by three vertices.
    Uses the barycentric coordinate method.
    """
    def sign(px, py, ax, ay, bx, by):
        return (px - bx) * (ay - by) - (ax - bx) * (py - by)
    
    d1 = sign(x, y, x1, y1, x2, y2)
    d2 = sign(x, y, x2, y2, x3, y3)
    d3 = sign(x, y, x3, y3, x1, y1)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)


def is_point_in_polygon(x: float, y: float, vertices: list) -> bool:
    """
    Check if a point is inside a polygon using the ray casting algorithm.
    
    Args:
        x, y: Point coordinates
        vertices: List of (x, y) tuples representing polygon vertices
    
    Returns:
        True if point is inside, False otherwise
    """
    inside = False
    n = len(vertices)
    
    p1x, p1y = vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def extract_outline(matrix: np.ndarray, stroke: int = 1) -> np.ndarray:
    """
    Extract the outline of a filled shape with specified stroke thickness.
    
    Args:
        matrix: Binary matrix of the filled shape
        stroke: Thickness of the outline
    
    Returns:
        Binary matrix with only the outline (at specified thickness)
    """
    if stroke <= 0:
        stroke = 1
    
    # Dilate to create the outer boundary with stroke
    dilated = ndimage.binary_dilation(matrix, iterations=stroke).astype(int)
    
    # Erode the original to create the inner boundary
    eroded = ndimage.binary_erosion(matrix, iterations=stroke).astype(int)
    
    # Outline is dilated minus eroded
    outline = dilated - eroded
    
    return outline


def get_shape(shape_type: str, size: int = 32, filled: bool = True, stroke: int = 1, **kwargs) -> np.ndarray:
    """
    Unified interface to generate geometric shapes.
    
    Args:
        shape_type: Type of shape ('circle', 'triangle', 'rectangle', 'star', 'polygon', 'diamond')
        size: Size parameter (radius for circle, side length for others)
        filled: Whether the shape is filled or hollow
        stroke: Thickness for hollow shapes
        **kwargs: Additional parameters for specific shapes
            - For rectangle: width, height
            - For polygon: sides
    
    Returns:
        2D numpy array representing the shape
    """
    shape_type = shape_type.lower()
    
    if shape_type == 'circle':
        return create_circle(size, filled=filled, stroke=stroke)
    
    elif shape_type == 'triangle':
        return create_triangle(size, filled=filled, stroke=stroke)
    
    elif shape_type == 'rectangle':
        width = kwargs.get('width', size)
        height = kwargs.get('height', size)
        return create_rectangle(width, height, filled=filled, stroke=stroke)
    
    elif shape_type == 'star':
        points = kwargs.get('points', 5)
        return create_star(size, points=points, filled=filled, stroke=stroke)
    
    elif shape_type == 'polygon':
        sides = kwargs.get('sides', 6)
        return create_polygon(size, sides=sides, filled=filled, stroke=stroke)
    
    elif shape_type == 'diamond':
        return create_diamond(size, filled=filled, stroke=stroke)
    
    else:
        raise ValueError(f"Unknown shape type: '{shape_type}'. "
                        f"Available shapes: circle, triangle, rectangle, star, polygon, diamond")


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt
    
    shapes = {
        'circle_filled': create_circle(16, filled=True),
        'circle_empty': create_circle(16, filled=False, stroke=2),
        'triangle_filled': create_triangle(32, filled=True),
        'triangle_empty': create_triangle(32, filled=False, stroke=1),
        'rectangle_filled': create_rectangle(30, 20, filled=True),
        'rectangle_empty': create_rectangle(30, 20, filled=False, stroke=2),
        'star': create_star(32, points=5, filled=True),
        'diamond': create_diamond(32, filled=True),
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for idx, (name, shape) in enumerate(shapes.items()):
        axes[idx].imshow(shape, cmap='Greys', origin='lower')
        axes[idx].set_title(name)
        axes[idx].set_aspect('equal')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('geometric_shapes_test.png', dpi=100, bbox_inches='tight')
    print("Test visualization saved as 'geometric_shapes_test.png'")
