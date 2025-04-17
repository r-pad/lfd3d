"""
A script for visualizing the weighted displacements predicted by Articubot.
The .glb file is assumed to have been saved by save_weighted_displacement_pcd_viz() in viz_utils.py
"""

import argparse

import open3d as o3d
import trimesh


def load_and_extract_geometries(filename):
    """Load GLB file and extract geometries."""
    mesh = trimesh.load(filename)
    return mesh.geometry  # Assuming this is a dictionary-like object


def create_point_cloud(geometry):
    """Create an Open3D point cloud from the given geometry."""
    points = geometry.vertices
    colors = geometry.colors[:, :3] / 255.0  # Normalize colors to [0, 1]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def create_line_set(geometry):
    """Create an Open3D line set from the given geometry, colored red."""
    line_points = geometry.vertices
    lines = [[i, i + 1] for i in range(0, len(line_points), 2)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(
        [[1, 0, 0] for _ in lines]
    )  # Red lines
    return line_set


def visualize(point_cloud, line_set):
    """Visualize the point cloud and lines using Open3D."""
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(point_cloud)
    vis.add_geometry(line_set)

    opt = vis.get_render_option()
    opt.point_size = 3.0  # Set point size

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Visualize GLB file geometries.")
    parser.add_argument("filename", help="Path to the GLB file")
    args = parser.parse_args()

    filename = args.filename  # Use the provided filename

    try:
        geometries = load_and_extract_geometries(filename)

        if "geometry_0" not in geometries:
            raise ValueError("geometry_0 not found in mesh.")
        point_cloud = create_point_cloud(geometries["geometry_0"])

        if "geometry_1" not in geometries:
            raise ValueError("geometry_1 not found in mesh.")
        line_set = create_line_set(geometries["geometry_1"])

        visualize(point_cloud, line_set)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
