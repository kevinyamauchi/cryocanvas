import numpy as np
import pymeshfix
from skimage.measure import marching_cubes
from trimesh import Trimesh
from trimesh.smoothing import filter_mut_dif_laplacian


def binary_mask_to_surface(
    object_mask: np.ndarray,
    n_mesh_smoothing_iterations: int = 10,
    diffusion_coefficient: float = 0.5,
) -> Trimesh:
    """Convert surface of a 3D binary mask (segmented object) into a watertight mesh.

    Parameters
    ----------
    object_mask  : BinaryMask
        A 3D binary image corresponding to the object you want to mesh.
    n_mesh_smoothing_iterations : int
        The number of interations of smoothing to perform. Smoothing is
        done by the trimesh mutable diffusion laplacian filter:
        https://trimsh.org/trimesh.smoothing.html#trimesh.smoothing.filter_mut_dif_laplacian
        Default value is 10.
    diffusion_coefficient : float
        The diffusion coefficient for smoothing. 0 is no diffusion.
        Default value is 0.5.
        https://trimsh.org/trimesh.smoothing.html#trimesh.smoothing.filter_mut_dif_laplacian

    Returns
    -------
    mesh : trimesh.Trimesh
        The resulting mesh as a trimesh.Trimesh object.
        https://trimsh.org/trimesh.base.html#github-com-mikedh-trimesh
    """
    vertices, faces, _, _ = marching_cubes(object_mask, 0)

    vertices_clean, faces_clean = pymeshfix.clean_from_arrays(vertices, faces)

    # create the mesh object
    mesh = Trimesh(vertices=vertices_clean, faces=faces_clean)

    # optionally clean up the mesh
    if n_mesh_smoothing_iterations > 0:
        filter_mut_dif_laplacian(
            mesh, iterations=n_mesh_smoothing_iterations, lamb=diffusion_coefficient
        )

    return mesh
