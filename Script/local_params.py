import numpy as np
from skimage import morphology, measure
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import os

def load_mask_from_txt(file_path):
    """Load binary mask from text file"""
    with open(file_path, 'r') as f:
        mask = np.loadtxt(f)
    
    return np.array(mask, dtype=bool)

def calculate_segment_curvatures(skeleton, segment_length=10, pixel_size_nm=5000/512):
    """Calculate curvatures for segments in skeleton, excluding nodes"""
    endpoints, nodes = find_branch_points(skeleton)
    branch_points = endpoints | nodes
    
    # Label skeleton branches (with nodes removed)
    skeleton_cleaned = skeleton.copy()
    skeleton_cleaned[nodes] = 0
    labeled = measure.label(skeleton_cleaned, connectivity=2)
    
    segments = []
    segment_id = 1
    
    for region in measure.regionprops(labeled):
        coords = region.coords
        if len(coords) < 2:
            continue
            
        ordered_path = order_branch_coordinates(coords)
        
        # Create overlapping segments
        for i in range(0, len(ordered_path) - segment_length + 1, segment_length//2):
            segment = ordered_path[i:i+segment_length]
            if len(segment) < 3:
                continue
                
            # Skip segments containing nodes
            if any(branch_points[y, x] for y, x in segment):
                continue
                
            curvature, smooth_coords = calculate_curvature(segment)
            
            segments.append({
                'id': segment_id,
                'length_pixels': len(segment),
                'length_nm': len(segment) * pixel_size_nm,
                'curvature': curvature,
                'mean_curvature': np.mean(curvature),
                'max_curvature': np.max(curvature),
                'x_coords': [x for y, x in segment],
                'y_coords': [y for y, x in segment],
                'smooth_x': smooth_coords[:, 0],
                'smooth_y': smooth_coords[:, 1]
            })
            segment_id += 1
            
    return segments, np.argwhere(nodes)

def find_branch_points(skeleton):
    """Identify endpoints (1 neighbor) and nodes (≥3 neighbors)"""
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant')
    
    endpoints = (neighbor_count == 1) & skeleton
    nodes = (neighbor_count >= 3) & skeleton
    
    return endpoints, nodes

def order_branch_coordinates(coords):
    """Order coordinates from one end to another"""
    points = [tuple(coord) for coord in coords]
    graph = {p: [] for p in points}
    
    for p in points:
        for q in points:
            if p != q and abs(p[0]-q[0]) <= 1 and abs(p[1]-q[1]) <= 1:
                graph[p].append(q)
    
    endpoints = [p for p in points if len(graph[p]) == 1]
    if not endpoints:
        endpoints = [points[0]]
    
    ordered = [endpoints[0]]
    current = endpoints[0]
    visited = set([current])
    
    while True:
        neighbors = [n for n in graph[current] if n not in visited]
        if not neighbors:
            break
        next_point = neighbors[0]
        ordered.append(next_point)
        visited.add(next_point)
        current = next_point
    
    return ordered

def calculate_curvature(points, smoothing_factor=2.0, pixel_size_nm=5000/512):
    """Calculate curvature using spline fitting"""
    x = np.array([p[1] for p in points])
    y = np.array([p[0] for p in points])
    
    tck, u = splprep([x, y], s=smoothing_factor, per=False)
    u_new = np.linspace(u.min(), u.max(), 3*len(points))
    x_new, y_new = splev(u_new, tck)
    dx, dy = splev(u_new, tck, der=1)
    d2x, d2y = splev(u_new, tck, der=2)
    
    curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5 / pixel_size_nm
    return curvature, np.column_stack((x_new, y_new))

def visualize_results(skeleton, segments, nodes):
    """Visualize raw skeleton pixels with curvature coloring and histogram"""
    plt.figure(figsize=(15, 5))
    
    # Create curvature map (same size as skeleton)
    curvature_map = np.zeros_like(skeleton, dtype=float)
    all_curvatures = []
    
    # Mark pixels with their curvature values
    for seg in segments:
        for x, y, curv in zip(seg['x_coords'], seg['y_coords'], seg['curvature']):
            curvature_map[y, x] = curv
        all_curvatures.extend(seg['curvature'])
    
    # Plot 1: Original skeleton with nodes marked
    plt.subplot(1, 3, 1)
    plt.imshow(skeleton, cmap='gray')
    if len(nodes) > 0:
        plt.plot(nodes[:, 1], nodes[:, 0], 'ro', markersize=4, label='Nodes')
    plt.title('Original Skeleton')
    plt.legend()

    # Plot 2: Raw skeleton with curvature coloring
    plt.subplot(1, 3, 2)
    plt.imshow(curvature_map, cmap='viridis', vmin=0)
    plt.colorbar(label='Curvature (1/nm)')
    plt.title('Pixel Curvature Values')
    
    # Plot 3: Curvature histogram
    plt.subplot(1, 3, 3)
    if all_curvatures:
        plt.hist(all_curvatures, bins=50, color='blue', alpha=0.7)
        plt.xlabel('Curvature (1/nm)')
        plt.ylabel('Frequency')
        plt.title('Curvature Distribution')
        
        # Add statistics
        median_curv = np.median(all_curvatures)
        mean_curv = np.mean(all_curvatures)
        plt.axvline(median_curv, color='red', linestyle='--', label=f'Median: {median_curv:.3f}')
        plt.axvline(mean_curv, color='green', linestyle='--', label=f'Mean: {mean_curv:.3f}')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No curvature data', ha='center', va='center')
    
    plt.tight_layout()
    plt.show()

def save_segment_data(segments, output_dir='segment_data'):
    """Save analysis results to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'summary.csv'), 'w') as f:
        f.write("SegmentID,LengthPx,LengthUm,MeanCurvature,MaxCurvature\n")
        for seg in segments:
            f.write(f"{seg['id']},{seg['length_pixels']},{seg['length_nm']:.2f}," +
                    f"{seg['mean_curvature']:.4f},{seg['max_curvature']:.4f}\n")
    
    for seg in segments:
        with open(os.path.join(output_dir, f"segment_{seg['id']}.csv"), 'w') as f:
            f.write("X,Y,Curvature\n")
            for x, y, curv in zip(seg['smooth_x'], seg['smooth_y'], seg['curvature']):
                f.write(f"{x:.1f},{y:.1f},{curv:.6f}\n")

if __name__ == "__main__":
    # Load mask from text file (1=skeleton, 0=background)
    mask = load_mask_from_txt('afm_skeleton.txt')
    
    # Analysis parameters
    segment_length = 20  # pixels
    pixel_size_nm = 5000/512  # nm/pixel
    
    # Run analysis
    segments, nodes = calculate_segment_curvatures(
        mask,
        segment_length=segment_length,
        pixel_size_nm=pixel_size_nm
    )
    
    # Save and visualize results
    save_segment_data(segments)
    visualize_results(mask, segments, nodes)
    
    # Print summary
    print(f"Analyzed {len(segments)} segments")
    print(f"Found {len(nodes)} branch points")
    if segments:
        print("\nTop 5 most curved segments:")
        for seg in sorted(segments, key=lambda x: x['max_curvature'], reverse=True)[:5]:
            print(f"Segment {seg['id']}: Max curvature = {seg['max_curvature']:.4f}")