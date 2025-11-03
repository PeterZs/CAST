"""
3D Mesh Generation Module using Multiple APIs

This module handles 3D mesh generation from inpainted object images
using various API services including Tripo3D and TRELLIS.
"""
import os
import numpy as np
import asyncio
import open3d as o3d 
from typing import List, Optional, Literal
from pathlib import Path
import time
import trimesh

from ..core.common import DetectedObject, Mesh3D, DepthEstimation
from ..utils.api_clients import Tripo3DClient, TrellisClient, Hunyuan3DClient, Hunyuan3DPaintClient
from ..utils.image_utils import save_image

TRIPO_3D_TRANSFORM = np.array([
    [0., 0., -1., 0.], 
    [0., 1., 0., 0.], 
    [1., 0., 0., 0.], 
    [0., 0., 0., 1.]
])


def save_ply_points(filename: str, points: np.ndarray) -> None:
    """
    Save 3D points to a PLY format file.
    
    This function exports a point cloud to the PLY (Polygon File Format) which
    can be viewed in 3D visualization software like MeshLab or Blender.
    
    Args:
        filename (str): Output PLY file path
        points (np.ndarray): Array of 3D points with shape [N, 3]
        
    Returns:
        None
    """
    with open(filename, 'w') as f:
        # Write PLY header
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % len(points))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        
        # Write point coordinates
        for point in points:
            f.write('%f %f %f\n' % (point[0], point[1], point[2]))

class MeshGenerationModule:
    """Module for 3D mesh generation using multiple APIs"""

    def __init__(self, provider: Literal["tripo3d", "trellis", "hunyuan"] = "tripo3d", base_url: Optional[str] = None):
        self.provider = provider
        self.base_url = base_url
        # Default to local TRELLIS deployment if no base_url specified for TRELLIS
        if provider == "tripo3d":
            self.tripo_client = Tripo3DClient()
            self.trellis_client = None
            self.hunyuan_client = None
            self.hunyuan_paint_client = None
        elif provider == "trellis":
            self.trellis_client = TrellisClient(base_url=self.base_url)
            self.tripo_client = None
            self.hunyuan_client = None
            self.hunyuan_paint_client = None
        elif provider == "hunyuan":
            self.hunyuan_client = Hunyuan3DClient()
            self.hunyuan_paint_client = Hunyuan3DPaintClient()
            self.tripo_client = None
            self.trellis_client = None
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        print(f"Initialized MeshGenerationModule with provider: {provider}")
        if provider == "trellis" and self.trellis_client:
            mode = "synchronous" if self.trellis_client.is_local else "asynchronous"
            print(f"TRELLIS mode: {mode} ({'local' if self.trellis_client.is_local else 'remote'} deployment)")
        elif provider == "hunyuan":
            print("Hunyuan3D mode: Geometry generation with point cloud conditioning + Texture painting")

    def _check_existing_mesh(self, detected_object: DetectedObject, output_dir: Optional[Path]) -> Optional[Mesh3D]:
        """
        Check if a mesh file already exists for this object and load it
        
        Args:
            detected_object: Object to check mesh for
            output_dir: Directory where meshes are saved
            
        Returns:
            Loaded Mesh3D if file exists, None otherwise
        """
        if output_dir is None:
            return None
        
        mesh_dir = output_dir / "meshes"
        if not mesh_dir.exists():
            return None
        
        # Check for different provider-specific naming patterns
        possible_filenames = [
            f"object_{detected_object.id}_trellis.glb",  # TRELLIS
            f"object_{detected_object.id}_hunyuan_untextured.glb",  # Hunyuan
            f"object_{detected_object.id}_hunyuan.glb",  # Hunyuan with texture
            f"object_{detected_object.id}.glb",  # Tripo3D
        ]
        
        for filename in possible_filenames:
            mesh_path = mesh_dir / filename
            if mesh_path.exists():
                print(f"  Found existing mesh for object {detected_object.id}: {filename}")
                mesh_3d = self._load_mesh_from_file(mesh_path)
                if mesh_3d:
                    return mesh_3d
        
        return None

    def generate_mesh_for_object_tripo_sync(self,
                                            detected_object: DetectedObject,
                                            output_dir: Optional[Path] = None) -> Optional[Mesh3D]:
        """
        Generate 3D mesh for a single object using its inpainted image
        
        Args:
            detected_object: Object with inpainted image
            output_dir: Directory to save the mesh file
            
        Returns:
            Mesh3D object if successful, None otherwise
        """
        if detected_object.generated_image is None:
            print(f"Warning: No inpainted image for object {detected_object.id}")
            return None

        print(f"Generating 3D mesh for object {detected_object.id}: {detected_object.description}")

        try:
            # Step 1: Upload image to Tripo3D
            print("  Uploading image...")
            file_token = self.tripo_client.upload_image(detected_object.generated_image)

            if not file_token:
                print(f"  Failed to upload image for object {detected_object.id}")
                return None

            # Step 2: Create 3D model task
            print("  Creating 3D model task...")
            task_id = self.tripo_client.create_3d_model(file_token)

            if not task_id:
                print(f"  Failed to create 3D model task for object {detected_object.id}")
                return None

            # Step 3: Wait for completion
            print("  Waiting for 3D model generation...")
            try:
                result = self.tripo_client.wait_for_completion(task_id, timeout=600)  # 10 minutes
            except TimeoutError:
                print(f"  Timeout waiting for object {detected_object.id}")
                return None
            except Exception as e:
                print(f"  Error during generation: {e}")
                return None

            # Step 4: Download the mesh
            if "result" in result and "pbr_model" in result["result"]:
                model_url = result["result"]["pbr_model"]["url"]
                print(f"  Downloading mesh from: {model_url}")

                # Create output path
                if output_dir:
                    mesh_dir = output_dir / "meshes"
                    mesh_dir.mkdir(exist_ok=True, parents=True)
                    mesh_path = mesh_dir / f"object_{detected_object.id}.glb"
                else:
                    mesh_path = Path(f"object_{detected_object.id}.glb")

                # Download mesh file
                if self.tripo_client.download_model(model_url, mesh_path):
                    # Apply a predefined rotation to convert into OpenGL convention 
                    mesh = trimesh.load(mesh_path)
                    mesh.apply_transform(TRIPO_3D_TRANSFORM)
                    mesh.export(mesh_path)
                    # Load mesh using trimesh
                    mesh_3d = self._load_mesh_from_file(mesh_path)
                    if mesh_3d:
                        print(f"  Successfully generated mesh for object {detected_object.id}")
                        return mesh_3d
                    else:
                        print(f"  Failed to load downloaded mesh for object {detected_object.id}")
                else:
                    print(f"  Failed to download mesh for object {detected_object.id}")
            else:
                print(f"  Invalid result format for object {detected_object.id}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error generating mesh for object {detected_object.id}: {e}")

        return None

    def generate_mesh_for_object_trellis_sync(self,
                                              detected_object: DetectedObject,
                                              output_dir: Optional[Path] = None,
                                              **trellis_kwargs) -> Optional[Mesh3D]:
        """
        Generate 3D mesh for a single object using TRELLIS synchronously (for local deployments)
        
        Args:
            detected_object: Object with inpainted image
            output_dir: Directory to save the mesh file
            **trellis_kwargs: Additional parameters for TRELLIS
            
        Returns:
            Mesh3D object if successful, None otherwise
        """
        if detected_object.generated_image is None and detected_object.cropped_image is None:
            print(f"Warning: No inpainted image for object {detected_object.id}")
            return None

        if not self.trellis_client.is_local:
            raise RuntimeError("Synchronous generation requires local TRELLIS deployment")

        print(f"Generating 3D mesh using TRELLIS (sync) for object {detected_object.id}: {detected_object.description}")

        try:
            # Generate synchronously
            result = self.trellis_client.generate_3d_sync(
                detected_object.generated_image
                if detected_object.generated_image is not None else detected_object.cropped_image, **trellis_kwargs)

            if not result:
                print(f"Failed TRELLIS sync generation for object {detected_object.id}")
                return None

            if result["output"] is not None and "model_file" in result["output"]:
                model_url = result["output"]["model_file"]

                # Create output path
                if output_dir:
                    mesh_dir = output_dir / "meshes"
                    mesh_dir.mkdir(exist_ok=True, parents=True)
                    mesh_path = mesh_dir / f"object_{detected_object.id}_trellis.glb"
                else:
                    mesh_path = Path(f"object_{detected_object.id}_trellis.glb")

                with open(mesh_path, "wb") as f:
                    f.write(model_url.read())

                # Download mesh file
                if os.path.exists(mesh_path):
                    # Load mesh using trimesh
                    mesh_3d = self._load_mesh_from_file(mesh_path)
                    if mesh_3d:
                        print(f"Successfully generated TRELLIS sync mesh for object {detected_object.id}")
                        return mesh_3d
                    else:
                        print(f"Failed to load downloaded TRELLIS sync mesh for object {detected_object.id}")
                else:
                    print(f"Failed to download TRELLIS sync mesh for object {detected_object.id}")
            else:
                print(f"Invalid TRELLIS sync result format for object {detected_object.id}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error generating TRELLIS sync mesh for object {detected_object.id}: {e}")

        return None

    async def generate_mesh_for_object_trellis_async(self,
                                                     detected_object: DetectedObject,
                                                     output_dir: Optional[Path] = None,
                                                     **trellis_kwargs) -> Optional[Mesh3D]:
        """
        Generate 3D mesh for a single object using TRELLIS asynchronously
        
        Args:
            detected_object: Object with inpainted image
            output_dir: Directory to save the mesh file
            **trellis_kwargs: Additional parameters for TRELLIS
            
        Returns:
            Mesh3D object if successful, None otherwise
        """
        if detected_object.generated_image is None and detected_object.cropped_image is None:
            print(f"Warning: No inpainted image for object {detected_object.id}")
            return None

        print(f"Generating 3D mesh using TRELLIS for object {detected_object.id}: {detected_object.description}")

        try:
            # Start async generation
            prediction_id = await self.trellis_client.generate_3d_async(
                detected_object.generated_image
                if detected_object.generated_image is not None else detected_object.cropped_image, **trellis_kwargs)

            if not prediction_id:
                print(f"Failed to start TRELLIS generation for object {detected_object.id}")
                return None

            # Wait for completion
            result = await self.trellis_client.wait_for_completion_async(prediction_id)

            if result["output"] is not None and "model_file" in result["output"]:
                model_url = result["output"]["model_file"]

                # Create output path
                if output_dir:
                    mesh_dir = output_dir / "meshes"
                    mesh_dir.mkdir(exist_ok=True, parents=True)
                    mesh_path = mesh_dir / f"object_{detected_object.id}_trellis.glb"
                else:
                    mesh_path = Path(f"object_{detected_object.id}_trellis.glb")

                # Download mesh file
                if await self.trellis_client._download_model_async(model_url, mesh_path):
                    # Load mesh using trimesh
                    mesh_3d = self._load_mesh_from_file(mesh_path)
                    if mesh_3d:
                        print(f"Successfully generated TRELLIS mesh for object {detected_object.id}")
                        return mesh_3d
                    else:
                        print(f"Failed to load downloaded TRELLIS mesh for object {detected_object.id}")
                else:
                    print(f"Failed to download TRELLIS mesh for object {detected_object.id}")
            else:
                print(f"Invalid TRELLIS result format for object {detected_object.id}")

        except Exception as e:
            print(f"Error generating TRELLIS mesh for object {detected_object.id}: {e}")

        return None

    def generate_mesh_for_object_hunyuan(self,
                                         detected_object: DetectedObject,
                                         output_dir: Optional[Path] = None,
                                         point_cloud: Optional[np.ndarray] = None,
                                         **hunyuan_kwargs) -> Optional[Mesh3D]:
        """
        Generate 3D mesh for a single object using Hunyuan3D-Omni with point cloud conditioning
        and texture painting with Hunyuan3D Paint
        
        Args:
            detected_object: Object with inpainted image
            output_dir: Directory to save the mesh file
            point_cloud: Optional point cloud for geometry conditioning (N, 3)
            **hunyuan_kwargs: Additional parameters for Hunyuan3D (num_inference_steps, octree_resolution, etc.)
            
        Returns:
            Mesh3D object if successful, None otherwise
        """
        if detected_object.generated_image is None and detected_object.cropped_image is None:
            print(f"Warning: No inpainted image for object {detected_object.id}")
            return None

        print(f"Generating 3D mesh using Hunyuan3D for object {detected_object.id}: {detected_object.description}")

        # Use generated image if available, otherwise use cropped image
        input_image = detected_object.generated_image if detected_object.generated_image is not None else detected_object.cropped_image

        try:
            # Create output directory
            if output_dir:
                mesh_dir = output_dir / "meshes"
                mesh_dir.mkdir(exist_ok=True, parents=True)
            else:
                mesh_dir = Path(".")

            # Step 1: Generate geometry with point cloud conditioning using Hunyuan3D-Omni
            print("  Step 1/2: Generating geometry with Hunyuan3D-Omni...")

            # Extract Hunyuan-specific parameters
            num_inference_steps = hunyuan_kwargs.get('num_inference_steps', 50)
            octree_resolution = hunyuan_kwargs.get('octree_resolution', 512)
            mc_level = hunyuan_kwargs.get('mc_level', 0)
            guidance_scale = hunyuan_kwargs.get('guidance_scale', 4.5)
            seed = hunyuan_kwargs.get('seed', 1234)

            import open3d as o3d 
            o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud))
            result = self.hunyuan_client.generate_mesh(image=input_image,
                                                       point_cloud=point_cloud,
                                                       num_inference_steps=num_inference_steps,
                                                       octree_resolution=octree_resolution,
                                                       mc_level=mc_level,
                                                       guidance_scale=guidance_scale,
                                                       seed=seed)

            if not result:
                print(f"  Failed Hunyuan3D-Omni geometry generation for object {detected_object.id}")
                return None

            # Extract mesh and apply post-processing
            mesh = result['mesh']
            mesh = self.hunyuan_client.postprocess_mesh(mesh)

            # Save untextured mesh
            untextured_mesh_path = mesh_dir / f"object_{detected_object.id}_hunyuan_untextured.glb"
            mesh.export(str(untextured_mesh_path))
            print(f"  Saved untextured mesh to: {untextured_mesh_path}")
            o3d.io.write_point_cloud(str(mesh_dir / f"object_{detected_object.id}_hunyuan_untextured.ply"), o3d_pcd)
            print(" Saved point cloud to: ", mesh_dir / f"object_{detected_object.id}_hunyuan_untextured.ply")
            save_ply_points(str(mesh_dir / f"object_{detected_object.id}_hunyuan_untextured_normed.ply"), result['sampled_point'].cpu().numpy())
                                                                                                                                             
            # Step 2: Generate texture using Hunyuan3D Paint
            print("  Step 2/2: Generating texture with Hunyuan3D Paint...")

            # Use the original input image for texture (better quality than generated image for texture)
            # If we have the original cropped image, use it; otherwise use generated
            texture_reference_image = detected_object.cropped_image if detected_object.cropped_image is not None else input_image

            # # Generate textured mesh
            # textured_mesh_path = mesh_dir / f"object_{detected_object.id}_hunyuan.obj"

            # result_path = self.hunyuan_paint_client.generate_texture(
            #     mesh_path=untextured_mesh_path,
            #     image=texture_reference_image,
            #     output_path=textured_mesh_path,
            #     use_remesh=True,
            #     save_glb=True
            # )

            # if result_path:
            #     # Load the final textured mesh
            #     # Hunyuan Paint saves both .obj and .glb if save_glb=True
            #     final_mesh_path = Path(result_path).with_suffix('.glb') if Path(result_path).suffix == '.obj' else Path(result_path)

            #     if not final_mesh_path.exists():
            #         # If GLB doesn't exist, use OBJ
            #         final_mesh_path = Path(result_path)

            #     mesh_3d = self._load_mesh_from_file(final_mesh_path)

            #     if mesh_3d:
            #         print(f"  Successfully generated Hunyuan3D mesh for object {detected_object.id}")
            #         return mesh_3d
            #     else:
            #         print(f"  Failed to load final Hunyuan3D mesh for object {detected_object.id}")
            # else:
            if True:
                print(f"  Failed Hunyuan3D Paint texture generation for object {detected_object.id}")
                # Fall back to untextured mesh
                print("  Using untextured mesh as fallback...")
                mesh_3d = self._load_mesh_from_file(untextured_mesh_path)
                if mesh_3d:
                    print(f"  Successfully loaded untextured mesh for object {detected_object.id}")
                    return mesh_3d

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error generating Hunyuan3D mesh for object {detected_object.id}: {e}")

        return None

    def _load_mesh_from_file(self, mesh_path: Path) -> Optional[Mesh3D]:
        """
        Load mesh from file using trimesh
        
        Args:
            mesh_path: Path to mesh file
            
        Returns:
            Mesh3D object if successful
        """
        try:
            # Load mesh using trimesh
            mesh = trimesh.load(str(mesh_path))

            # Handle different mesh types
            if isinstance(mesh, trimesh.Scene):
                # If it's a scene, get the first mesh
                mesh_geometries = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
                if mesh_geometries:
                    mesh = mesh_geometries[0]
                else:
                    print(f"No valid mesh found in scene file: {mesh_path}")
                    return None

            if not isinstance(mesh, trimesh.Trimesh):
                print(f"Loaded object is not a valid mesh: {mesh_path}")
                return None

            # Extract vertices and faces
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.faces)

            # Extract texture/color information if available
            textures = None
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                textures = np.array(mesh.visual.vertex_colors)
            elif hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
                textures = np.array(mesh.visual.face_colors)

            # Create Mesh3D object
            mesh_3d = Mesh3D(vertices=vertices, faces=faces, textures=textures, file_path=mesh_path)

            print(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
            return mesh_3d

        except Exception as e:
            print(f"Error loading mesh from {mesh_path}: {e}")
            return None

    def batch_generate_meshes(self,
                              detected_objects: List[DetectedObject],
                              output_dir: Optional[Path] = None,
                              max_concurrent: int = 3,
                              depth_estimation: Optional[DepthEstimation] = None,
                              image: Optional[np.ndarray] = None,
                              **provider_kwargs) -> List[Optional[Mesh3D]]:
        """
        Generate meshes for multiple objects with controlled concurrency
        
        Args:
            detected_objects: List of objects to generate meshes for
            output_dir: Directory to save mesh files
            max_concurrent: Maximum number of concurrent API calls
            depth_estimation: Optional depth estimation for point cloud extraction (required for Hunyuan)
            image: Optional RGB image for point cloud extraction (required for Hunyuan)
            **provider_kwargs: Additional parameters for the mesh generation provider
            
        Returns:
            List of Mesh3D objects (None for failed generations)
        """
        print(f"Starting batch mesh generation for {len(detected_objects)} objects using {self.provider}...")

        if self.provider == "trellis":
            # Choose sync or async based on deployment type
            if self.trellis_client.is_local:
                return self._batch_generate_meshes_trellis_sync(detected_objects, output_dir, **provider_kwargs)
            else:
                return asyncio.run(self._batch_generate_meshes_async(detected_objects, output_dir, **provider_kwargs))
        elif self.provider == "hunyuan":
            # Use Hunyuan with point cloud conditioning
            return self._batch_generate_meshes_hunyuan(detected_objects, output_dir, depth_estimation, image,
                                                       **provider_kwargs)
        else:
            # Use sync generation for Tripo3D
            return self._batch_generate_meshes_tripo_sync(detected_objects, output_dir, max_concurrent)

    def _batch_generate_meshes_tripo_sync(self,
                                          detected_objects: List[DetectedObject],
                                          output_dir: Optional[Path] = None,
                                          max_concurrent: int = 3) -> List[Optional[Mesh3D]]:
        """Synchronous batch generation for Tripo3D with instance-level resuming"""
        meshes = []

        # Process objects in batches to avoid overwhelming the API
        for i in range(0, len(detected_objects), max_concurrent):
            batch = detected_objects[i:i + max_concurrent]
            print(f"Processing batch {i//max_concurrent + 1}...")

            batch_meshes = []
            for obj in batch:
                # Check if mesh already exists
                existing_mesh = self._check_existing_mesh(obj, output_dir)
                if existing_mesh is not None:
                    print(f"Skipping mesh generation for object {obj.id}: {obj.description} (already exists)")
                    batch_meshes.append(existing_mesh)
                else:
                    mesh = self.generate_mesh_for_object_tripo_sync(obj, output_dir)
                    batch_meshes.append(mesh)
                    # Add delay between requests to be respectful to the API
                    time.sleep(2)

            meshes.extend(batch_meshes)

        successful_meshes = sum(1 for mesh in meshes if mesh is not None)
        print(f"Batch mesh generation complete. {successful_meshes}/{len(detected_objects)} successful.")

        return meshes

    def _batch_generate_meshes_trellis_sync(self,
                                            detected_objects: List[DetectedObject],
                                            output_dir: Optional[Path] = None,
                                            **trellis_kwargs) -> List[Optional[Mesh3D]]:
        """Synchronous batch generation for TRELLIS (local deployments) with instance-level resuming"""
        print(f"Starting sync TRELLIS batch generation for {len(detected_objects)} objects...")

        # Filter objects with valid images
        valid_objects = [
            obj for obj in detected_objects if obj.generated_image is not None or obj.cropped_image is not None
        ]

        if not valid_objects:
            print("No valid objects with inpainted images for TRELLIS sync generation")
            return []

        # Generate meshes synchronously with instance-level resuming
        meshes = []
        for i, obj in enumerate(valid_objects):
            print(f"Processing object {i+1}/{len(valid_objects)} synchronously...")
            
            # Check if mesh already exists
            existing_mesh = self._check_existing_mesh(obj, output_dir)
            if existing_mesh is not None:
                print(f"Skipping mesh generation for object {obj.id}: {obj.description} (already exists)")
                meshes.append(existing_mesh)
            else:
                mesh = self.generate_mesh_for_object_trellis_sync(obj, output_dir, **trellis_kwargs)
                meshes.append(mesh)

        successful_meshes = sum(1 for mesh in meshes if mesh is not None)
        print(f"Sync TRELLIS batch generation complete. {successful_meshes}/{len(valid_objects)} successful.")

        return meshes

    async def _batch_generate_meshes_async(self,
                                           detected_objects: List[DetectedObject],
                                           output_dir: Optional[Path] = None,
                                           **trellis_kwargs) -> List[Optional[Mesh3D]]:
        """Asynchronous batch generation for TRELLIS with instance-level resuming"""
        print(f"Starting async TRELLIS batch generation for {len(detected_objects)} objects...")

        # Separate objects into already-generated and need-generation
        images_to_generate = []
        objects_to_generate = []
        all_meshes = [None] * len(detected_objects)  # Pre-allocate list
        
        for idx, obj in enumerate(detected_objects):
            if obj.generated_image is None:
                continue
            
            # Check if mesh already exists
            existing_mesh = self._check_existing_mesh(obj, output_dir)
            if existing_mesh is not None:
                print(f"Skipping mesh generation for object {obj.id}: {obj.description} (already exists)")
                all_meshes[idx] = existing_mesh
            else:
                images_to_generate.append(obj.generated_image)
                objects_to_generate.append((idx, obj))

        if not images_to_generate:
            print("No objects need mesh generation (all already exist or no valid images)")
            return [m for m in all_meshes if m is not None]

        # Start batch async generation for objects that need it
        print(f"Generating meshes for {len(images_to_generate)} objects asynchronously...")
        prediction_ids = await self.trellis_client.batch_generate_async(images_to_generate, **trellis_kwargs)

        # Collect results
        if output_dir:
            mesh_dir = output_dir / "meshes"
            mesh_dir.mkdir(exist_ok=True, parents=True)
        else:
            mesh_dir = None

        model_paths = await self.trellis_client.collect_results_async(prediction_ids, [obj[1].id for obj in objects_to_generate], mesh_dir)

        # Load meshes from downloaded files and place them in the correct positions
        for (idx, obj), model_path in zip(objects_to_generate, model_paths):
            if model_path:
                mesh_3d = self._load_mesh_from_file(Path(model_path))
                all_meshes[idx] = mesh_3d
            else:
                all_meshes[idx] = None

        # Filter out None values and return
        meshes = [m for m in all_meshes if m is not None]
        successful_meshes = len(meshes)
        print(f"Async TRELLIS batch generation complete. {successful_meshes}/{len(detected_objects)} successful.")

        return meshes

    def _batch_generate_meshes_hunyuan(self,
                                       detected_objects: List[DetectedObject],
                                       output_dir: Optional[Path] = None,
                                       depth_estimation: Optional[DepthEstimation] = None,
                                       image: Optional[np.ndarray] = None,
                                       **hunyuan_kwargs) -> List[Optional[Mesh3D]]:
        """Batch generation for Hunyuan3D with point cloud conditioning and instance-level resuming"""
        print(f"Starting Hunyuan3D batch generation for {len(detected_objects)} objects...")

        # Filter objects with valid images
        valid_objects = [
            obj for obj in detected_objects if obj.generated_image is not None or obj.cropped_image is not None
        ]

        if not valid_objects:
            print("No valid objects with images for Hunyuan3D generation")
            return []

        # Extract point clouds for each object if depth estimation is available
        object_point_clouds = {}
        if depth_estimation is not None and image is not None:
            print("Extracting point clouds for objects...")
            from ..modules.depth_estimation import DepthEstimationModule
            depth_module = DepthEstimationModule()

            for obj in valid_objects:
                if obj.mask is not None:
                    try:
                        # Extract object-specific point cloud
                        obj_pc, obj_normals = depth_module.extract_object_point_cloud(depth_estimation,
                                                                                      image,
                                                                                      obj.mask,
                                                                                      use_opengl_coords=True)

                        # build up the point cloud then remove the outliers 
                        obj_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pc))
                        obj_pcd, _ = obj_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.2)
                        obj_pcd, _ = obj_pcd.remove_radius_outlier(nb_points=25, radius=0.05)
                        # downsample 
                        num_points = 2048 
                        obj_pcd = obj_pcd.random_down_sample(num_points / len(obj_pcd.points))
                        # re-convert back to the numpy array 
                        object_point_clouds[obj.id] = np.asarray(obj_pcd.points)
                    except Exception as e:
                        print(f"  Warning: Failed to extract point cloud for object {obj.id}: {e}")

        # Generate meshes sequentially (Hunyuan models are GPU-intensive) with instance-level resuming
        meshes = []
        for i, obj in enumerate(valid_objects):
            print(f"Processing object {i+1}/{len(valid_objects)}...")

            # Check if mesh already exists
            existing_mesh = self._check_existing_mesh(obj, output_dir)
            if existing_mesh is not None:
                print(f"Skipping mesh generation for object {obj.id}: {obj.description} (already exists)")
                meshes.append(existing_mesh)
            else:
                # Get point cloud for this object if available
                point_cloud = object_point_clouds.get(obj.id, None)

                if point_cloud is not None:
                    print(f"  Using point cloud with {len(point_cloud)} points for conditioning")
                else:
                    print(f"  No point cloud available for object {obj.id}, generating without conditioning")

                mesh = self.generate_mesh_for_object_hunyuan(obj, output_dir, point_cloud=point_cloud, **hunyuan_kwargs)
                meshes.append(mesh)

        successful_meshes = sum(1 for mesh in meshes if mesh is not None)
        print(f"Hunyuan3D batch generation complete. {successful_meshes}/{len(valid_objects)} successful.")

        return meshes

    def run(self,
            detected_objects: List[DetectedObject],
            output_dir: Optional[Path] = None,
            depth_estimation: Optional[DepthEstimation] = None,
            image: Optional[np.ndarray] = None,
            **provider_kwargs) -> List[Optional[Mesh3D]]:
        """
        Run the complete mesh generation pipeline
        
        Args:
            detected_objects: List of objects with inpainted images
            output_dir: Directory to save results
            depth_estimation: Optional depth estimation for point cloud extraction (required for Hunyuan)
            image: Optional RGB image for point cloud extraction (required for Hunyuan)
            **provider_kwargs: Additional parameters for the mesh generation provider
            
        Returns:
            List of generated meshes
        """
        print("Starting mesh generation pipeline...")
        if not detected_objects:
            print("No objects with inpainted images found")
            return []

        print(f"Generating meshes for {len(detected_objects)} objects...")

        # Generate meshes
        meshes = self.batch_generate_meshes(detected_objects,
                                            output_dir,
                                            depth_estimation=depth_estimation,
                                            image=image,
                                            **provider_kwargs)
        assert not any([mesh is None for mesh in meshes])
        # we also pair the mesh the input images
        assert len(meshes) == len(detected_objects)
        for (mesh, obj) in zip(meshes, detected_objects):
            mesh.input_image = obj.generated_image

        # Save summary
        if output_dir:
            self._save_summary(detected_objects, meshes, output_dir)

        print("Mesh generation pipeline complete.")
        return meshes

    def _save_summary(self, detected_objects: List[DetectedObject], meshes: List[Optional[Mesh3D]],
                      output_dir: Path) -> None:
        """Save mesh generation summary"""
        mesh_dir = output_dir / "meshes"
        mesh_dir.mkdir(exist_ok=True, parents=True)

        summary = {
            "total_objects": len(detected_objects),
            "successful_meshes": sum(1 for mesh in meshes if mesh is not None),
            "failed_meshes": sum(1 for mesh in meshes if mesh is None),
            "objects": []
        }

        for i, (obj, mesh) in enumerate(zip(detected_objects, meshes)):
            if mesh.input_image is not None:
                save_image(mesh.input_image, mesh_dir / f"object_{obj.id}_input_image.png")

            obj_summary = {
                "id": obj.id,
                "description": obj.description,
                "mesh_generated": mesh is not None,
                "vertices_count": len(mesh.vertices) if mesh else 0,
                "faces_count": len(mesh.faces) if mesh else 0,
                "mesh_file": str(mesh.file_path) if mesh and mesh.file_path else None
            }
            summary["objects"].append(obj_summary)

        # Save summary as JSON
        import json
        with open(mesh_dir / "generation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved mesh generation summary: {summary['successful_meshes']}/{summary['total_objects']} successful")

    def visualize_mesh(self, mesh: Mesh3D) -> None:
        """
        Visualize a 3D mesh using trimesh
        
        Args:
            mesh: Mesh3D object to visualize
        """
        try:
            # Create trimesh object
            trimesh_obj = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

            # Add colors if available
            if mesh.textures is not None:
                if mesh.textures.shape[0] == len(mesh.vertices):
                    trimesh_obj.visual.vertex_colors = mesh.textures
                elif mesh.textures.shape[0] == len(mesh.faces):
                    trimesh_obj.visual.face_colors = mesh.textures

            # Show mesh
            print("Visualizing mesh... Close the window to continue.")
            trimesh_obj.show()

        except Exception as e:
            print(f"Error visualizing mesh: {e}")

    def export_mesh_formats(self, mesh: Mesh3D, output_path: Path, formats: List[str] = ["obj", "ply"]) -> None:
        """
        Export mesh to different formats
        
        Args:
            mesh: Mesh3D object to export
            output_path: Base output path (without extension)
            formats: List of formats to export to
        """
        try:
            # Create trimesh object
            trimesh_obj = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

            # Add colors if available
            if mesh.textures is not None:
                if mesh.textures.shape[0] == len(mesh.vertices):
                    trimesh_obj.visual.vertex_colors = mesh.textures

            # Export to requested formats
            for fmt in formats:
                export_path = output_path.with_suffix(f".{fmt}")
                trimesh_obj.export(str(export_path))
                print(f"Exported mesh to {export_path}")

        except Exception as e:
            print(f"Error exporting mesh: {e}")
