import functools
import numbers
from typing import List, Union

import numpy as np
import open3d as o3d  # type: ignore
import math
from air_corridor.d3.geometry import geom3d
from air_corridor.tools.util import vec2vec_rotation

print(o3d.__file__)


@functools.singledispatch
def to_open3d_geom(geom):
    return geom


@to_open3d_geom.register  # type: ignore[no-redef]
def _(points: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: geom3d.Line, length: numbers.Number = 1):
    points = (
            geom.anchor_point
            + np.stack([geom.orientation_vec, -geom.orientation_vec], axis=0) * length / 2
    )

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    return line_set


@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: geom3d.Sphere):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=geom.radius)
    mesh.translate(geom.center)

    return o3d.geometry.LineSet.create_from_triangle_mesh(mesh)


@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: geom3d.Plane, length: numbers.Number = 1):
    points = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]]) * length / 2

    mesh = o3d.geometry.TetraMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.tetras = o3d.utility.Vector4iVector(np.array([[0, 1, 2, 3]]))

    rotation = vec2vec_rotation([0, 0, 1], geom.normal)
    mesh.rotate(rotation)
    mesh.translate(geom.anchor_point)

    return o3d.geometry.LineSet.create_from_tetra_mesh(mesh)


@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: geom3d.Cylinder):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=geom.radius, height=geom.length)

    mesh.remove_vertices_by_index([0, 1])

    rotation = vec2vec_rotation([0, 0, 1], geom.direction)
    mesh.rotate(rotation)
    mesh.translate(geom.anchor_point)

    return o3d.geometry.LineSet.create_from_triangle_mesh(mesh)


@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: geom3d.Circle3D):
    mesh = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=geom.radius, tube_radius=1e-6
    )
    rotation = vec2vec_rotation([0, 0, 1], geom.orientation_vec)
    mesh.rotate(rotation)
    mesh.translate(geom.center)

    return o3d.geometry.LineSet.create_from_triangle_mesh(mesh)


@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: geom3d.Torus):
    mesh = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=geom.major_radius, tube_radius=geom.minor_radius
    )
    rotation = vec2vec_rotation([0, 0, 1], geom.orientation_vec)
    mesh.rotate(rotation)
    mesh.translate(geom.center)

    return o3d.geometry.LineSet.create_from_triangle_mesh(mesh)


@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: geom3d.directionalPartialTorus):
    mesh = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=geom.major_radius, tube_radius=geom.minor_radius
    )

    begin_degree = int(len(mesh.triangles) * geom.begin_degree / math.pi / 2)
    end_degree = int(len(mesh.triangles) * geom.end_degree / math.pi / 2)

    if begin_degree < 0 and end_degree < 0:
        new_triangles = np.asarray(mesh.triangles)[begin_degree:end_degree]
    elif begin_degree < 0:
        new_triangles = np.concatenate((np.asarray(mesh.triangles)[begin_degree:],
                                        np.asarray(mesh.triangles)[:end_degree]))
    elif begin_degree >= 0:
        new_triangles = np.asarray(mesh.triangles)[begin_degree:end_degree]

    mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    # mesh.triangle_normals = o3d.utility.Vector3dVector(
    #     np.asarray(mesh.triangle_normals)[begin_degree:end_degree, :])

    print(mesh.triangles)
    # o3d.visualization.draw_geometries([mesh])

    rotation = vec2vec_rotation([0, 0, 1], geom.orientation_vec)
    mesh.rotate(rotation)
    mesh.translate(geom.center)

    return o3d.geometry.LineSet.create_from_triangle_mesh(mesh)


def plot(
        geometries_or_points: List[Union[geom3d.GeometricShape, np.ndarray]],
        display_coordinate_frame: bool = False,
):
    geometries = [to_open3d_geom(g) for g in geometries_or_points]
    if display_coordinate_frame:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame())
    o3d.visualization.draw_geometries(geometries)
