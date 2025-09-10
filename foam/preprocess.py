import trimesh
import numpy as np

def add_thickness(mesh, thickness):
    vertices = mesh.vertices
    
    # Force normals in z direction for planar mesh
    if np.any(np.ptp(mesh.vertices, axis=0) < 1e-6):
        normals = np.zeros_like(vertices)
        normals[:, 2] = 1.0  # Set all normals to point in +z direction
    else:
        normals = mesh.vertex_normals
    
    # Create offset vertices
    offset_vertices = vertices + (normals * thickness)
    
    # Get boundary edges
    edges = mesh.edges_unique
    
    # Create side faces
    side_faces = []
    for edge in edges:
        v1, v2 = edge
        v3, v4 = v1 + len(vertices), v2 + len(vertices)
        side_faces.extend([
            [v1, v2, v3],
            [v2, v4, v3]
        ])
    
    # Stack vertices and faces
    new_vertices = np.vstack((vertices, offset_vertices))
    new_faces = np.vstack((
        mesh.faces,
        np.fliplr(mesh.faces) + len(vertices),
        side_faces
    ))
    
    return trimesh.Trimesh(vertices=new_vertices, faces=new_faces)


# Load and process mesh
original_mesh = trimesh.load('../assets/meshes/link_aruco_left_base.STL')

print("Original dimensions:")
print(np.ptp(original_mesh.vertices, axis=0))
print("Is planar:", np.any(np.ptp(original_mesh.vertices, axis=0) < 1e-6))

print("\nNormal directions:")
print(np.unique(original_mesh.vertex_normals, axis=0))

thickened_mesh = add_thickness(original_mesh, thickness=0.01)

print("\nNew dimensions:")
print(np.ptp(thickened_mesh.vertices, axis=0))
print(thickened_mesh.is_watertight)

thickened_mesh.export('New_thickened_mesh.stl')  # You can also use .obj, .ply, etc.
