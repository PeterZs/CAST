"""
Render-and-Compare Module for Rotation Estimation

This module implements a render-and-compare approach for estimating object rotations
when ICP fails. It uses Blender to render the mesh from multiple viewpoints and 
uses Qwen-VL or CLIP to find the best matching orientation.
"""
import sys 
import numpy as np
import cv2
from typing import List, Tuple
from pathlib import Path
import os
from io import StringIO
import bpy

import torch
from transformers import CLIPProcessor, CLIPModel

# Try to import Orient-Anything
try:
    orient_anything_path = Path(__file__).parent.parent.parent / "thirdparty" / "Orient-Anything"
    if orient_anything_path.exists() and str(orient_anything_path) not in sys.path:
        sys.path.insert(0, str(orient_anything_path))
    from orient_anything_wrapper import OrientAnythingPredictor, ORIENT_ANYTHING_AVAILABLE
except ImportError as e:
    ORIENT_ANYTHING_AVAILABLE = False
    OrientAnythingPredictor = None
    print(f"Warning: Orient-Anything not available: {e}")

from ..core.common import Mesh3D, DetectedObject
from ..utils.api_clients import QwenVLClient
from ..utils.image_utils import load_image

NUM_DEBUG = 2

class SuppressOutput:
    """Enhanced context manager to suppress Blender rendering output"""
    
    def __enter__(self):
        # Store original file descriptors
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        
        # Redirect Python stdout/stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        # Also redirect OS-level stdout/stderr (for C/C++ output)
        self._stdout_fd = os.dup(1)
        self._stderr_fd = os.dup(2)
        
        # Create null file descriptor
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        
        # Redirect file descriptors
        os.dup2(self._devnull, 1)
        os.dup2(self._devnull, 2)
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore OS-level stdout/stderr
        os.dup2(self._stdout_fd, 1)
        os.dup2(self._stderr_fd, 2)
        
        # Close file descriptors
        os.close(self._stdout_fd)
        os.close(self._stderr_fd)
        os.close(self._devnull)
        
        # Restore Python stdout/stderr
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

class RenderCompareModule:
    """Module for rotation estimation using render-and-compare approach"""
    
    def __init__(self, backend: str = "qwen"):
        """
        Initialize the render-and-compare module
        
        Args:
            backend: Backend to use for orientation matching ("qwen", "clip", or "orient_anything")
        """
        self.backend = backend.lower()
        
        # Initialize backend-specific models
        self.qwen_client = None
        self.clip_model = None
        self.clip_processor = None
        self.orient_anything_model = None
        
        if self.backend == "qwen":
            self.qwen_client = QwenVLClient()
        elif self.backend == "clip":
            self._initialize_clip()
        elif self.backend == "orient_anything":
            self._initialize_orient_anything()
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'qwen', 'clip', or 'orient_anything'")
        
        # Rendering parameters
        self.image_size = 384    # Size of each rendered image
        
        # Elevation angles (in degrees)
        self.elevations = [-90., -60, -30., 0., 30., 60, 90.][:NUM_DEBUG]  # Low, medium, high viewpoints
        
        # Azimuth angles (in degrees) - evenly distributed around object
        self.azimuths = [i * 45 for i in range(8)][:NUM_DEBUG]  # 0, 45, 90, 135, 180, 225, 270, 315
        
        # Roll angles (in degrees) - camera rotation around viewing axis
        self.rolls = [0., 90., 180., 270.][:NUM_DEBUG]  # 4 roll angles
        
        # Calculated counts
        self.num_elevations = len(self.elevations)
        self.num_azimuths = len(self.azimuths)
        self.num_rolls = len(self.rolls)
        self.total_views = self.num_elevations * self.num_azimuths * self.num_rolls
        
        print(f"RenderCompareModule initialized with backend: {self.backend}")
        print(f"Total views: {self.total_views} ({self.num_elevations} elevations × {self.num_azimuths} azimuths × {self.num_rolls} rolls)")
    
    def _initialize_clip(self):
        """Initialize CLIP model and processor"""
        print("Loading CLIP model...")
        model_name = "openai/clip-vit-large-patch14-336"
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        
        print(f"CLIP model loaded on device: {self.device}")
    
    def _initialize_orient_anything(self):
        """Initialize Orient-Anything model"""
        if not ORIENT_ANYTHING_AVAILABLE:
            raise RuntimeError("Orient-Anything is not available. Please install the required dependencies.")
        
        print("Loading Orient-Anything model...")
        cache_dir = str(Path(__file__).parent.parent.parent / "thirdparty" / "Orient-Anything")
        self.orient_anything_model = OrientAnythingPredictor(cache_dir=cache_dir)
        print("Orient-Anything model loaded successfully")
        
    def setup_blender_scene(self) -> None:
        """Setup Blender scene for rendering"""
        
        # Clear existing objects more carefully
        bpy.ops.object.select_all(action='DESELECT')
        
        # Delete all mesh objects
        for obj in bpy.context.scene.objects:
            if obj.type in ['MESH', 'CAMERA', 'LIGHT']:
                bpy.data.objects.remove(obj, do_unlink=True)
        
        # Set up camera
        bpy.ops.object.camera_add(location=(3, 0, 1))
        camera = bpy.context.object
        camera.name = "RenderCamera"
        
        # Set camera as active camera for the scene
        bpy.context.scene.camera = camera
        
        # Set up lighting - three-point lighting setup
        # Key light
        bpy.ops.object.light_add(type='SUN', location=(2, 2, 3))
        key_light = bpy.context.object
        key_light.data.energy = 3.0
        
        # Fill light
        bpy.ops.object.light_add(type='SUN', location=(-1, 1, 2))
        fill_light = bpy.context.object
        fill_light.data.energy = 1.5
        
        # Back light
        bpy.ops.object.light_add(type='SUN', location=(0, -2, 1))
        back_light = bpy.context.object
        back_light.data.energy = 1.0
        
        # Set render settings
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.render.resolution_x = self.image_size
        bpy.context.scene.render.resolution_y = self.image_size
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.cycles.samples = 64  # Good quality/speed balance
        
        # Set background to white
        world = bpy.context.scene.world
        world.use_nodes = True
        bg_node = world.node_tree.nodes["Background"]
        bg_node.inputs[0].default_value = (1, 1, 1, 1)  # White background
        
    def import_mesh_to_blender(self, mesh: Mesh3D) -> str:
        """
        Import mesh to Blender scene using GLB file if available
        
        Args:
            mesh: 3D mesh to import
            
        Returns:
            Name of the imported object
        """
        # Try to import from GLB file first (preserves materials and textures)
        if mesh.file_path and mesh.file_path.exists() and mesh.file_path.suffix.lower() == '.glb':
            return self._import_glb_file(mesh.file_path)
        else:
            # Fallback to raw mesh data import
            return self._import_raw_mesh(mesh)
    
    def _import_glb_file(self, glb_path: Path) -> str:
        """
        Import GLB file to Blender scene
        
        Args:
            glb_path: Path to GLB file
            
        Returns:
            Name of the imported object
        """
        try:
            # Import GLB file
            bpy.ops.import_scene.gltf(filepath=str(glb_path))
            
            # Get the imported object (should be the last selected object)
            imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            
            if not imported_objects:
                print(f"Warning: No mesh objects found in GLB file {glb_path}")
                return self._create_fallback_object()
            
            # Use the first mesh object
            obj = imported_objects[0]
            obj.name = "ObjectToRender"
            
            # Center the object at origin
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
            obj.location = (0, 0, 0)
            
            print(f"Successfully imported GLB file: {glb_path}")
            return obj.name
            
        except Exception as e:
            print(f"Error importing GLB file {glb_path}: {e}")
            return self._create_fallback_object()
    
    def _import_raw_mesh(self, mesh: Mesh3D) -> str:
        """
        Import raw mesh data to Blender (fallback method)
        
        Args:
            mesh: 3D mesh with vertices and faces
            
        Returns:
            Name of the imported object
        """
        try:
            # Create mesh in Blender
            mesh_data = bpy.data.meshes.new("ObjectMesh")
            mesh_data.from_pydata(mesh.vertices.tolist(), [], mesh.faces.tolist())
            mesh_data.update()
            
            # Create object
            obj = bpy.data.objects.new("ObjectToRender", mesh_data)
            bpy.context.collection.objects.link(obj)
            
            # Center the object at origin
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
            obj.location = (0, 0, 0)
            
            # Add material for better rendering (since raw mesh has no materials)
            material = bpy.data.materials.new(name="ObjectMaterial")
            material.use_nodes = True
            
            # Set up basic material
            bsdf = material.node_tree.nodes["Principled BSDF"]
            bsdf.inputs[0].default_value = (0.8, 0.8, 0.8, 1.0)  # Base color
            bsdf.inputs[4].default_value = 0.5  # Metallic
            bsdf.inputs[7].default_value = 0.3  # Roughness
            
            obj.data.materials.append(material)
            
            print("Imported raw mesh data (no textures/materials from GLB)")
            return obj.name
            
        except Exception as e:
            print(f"Error importing raw mesh: {e}")
            return self._create_fallback_object()
    
    def _create_fallback_object(self) -> str:
        """Create a simple fallback object when import fails"""
        try:
            bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
            obj = bpy.context.object
            obj.name = "FallbackObject"
            return obj.name
        except Exception:
            return "Cube"  # Default cube should exist
    
    def set_camera_position(self, elevation: float, azimuth: float, roll: float = 0.0, distance: float = 2.4) -> None:
        """
        Set camera position based on spherical coordinates with roll
        
        Args:
            elevation: Elevation angle in degrees
            azimuth: Azimuth angle in degrees
            roll: Roll angle in degrees (camera rotation around viewing axis)
            distance: Distance from object center
        """
        camera = bpy.data.objects.get("RenderCamera")
        if camera is None:
            print("Warning: RenderCamera not found, creating new camera")
            bpy.ops.object.camera_add(location=(3, 0, 1))
            camera = bpy.context.object
            camera.name = "RenderCamera"
            bpy.context.scene.camera = camera
        
        # Convert to radians
        elev_rad = np.radians(elevation)
        azim_rad = np.radians(azimuth)
        roll_rad = np.radians(roll)
        
        # Spherical to Cartesian conversion (Blender Z-up coordinates)
        x = distance * np.cos(elev_rad) * np.cos(azim_rad)
        y = distance * np.cos(elev_rad) * np.sin(azim_rad)
        z = distance * np.sin(elev_rad)
        
        camera.location = (x, y, z)
        
        # Calculate camera rotation to look at origin
        direction = np.array([0, 0, 0]) - np.array([x, y, z])  # Look at origin
        direction = direction / np.linalg.norm(direction)
        
        # Get base rotation (looking at origin)
        base_rotation = self._look_at_rotation(direction)
        
        # Apply roll rotation around the viewing axis (local Z-axis of camera)
        # We need to convert Euler angles to rotation matrix, apply roll, and convert back
        from scipy.spatial.transform import Rotation
        rot_base = Rotation.from_euler('xyz', base_rotation)
        rot_roll = Rotation.from_euler('z', roll_rad)
        
        # Combine: first look at origin, then roll around viewing axis
        combined_rotation = rot_base * rot_roll
        camera.rotation_euler = combined_rotation.as_euler('xyz')
        
        # Ensure camera is set as active camera
        bpy.context.scene.camera = camera
    
    def _look_at_rotation(self, direction: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate Euler rotation to look in given direction
        
        Args:
            direction: Normalized direction vector
            
        Returns:
            Euler angles (x, y, z) in radians
        """
        # Calculate rotation to align -Z axis (camera forward) with direction
        z_axis = -direction
        
        # Calculate Y axis (up vector)
        world_up = np.array([0, 0, 1])
        if abs(np.dot(z_axis, world_up)) > 0.99:
            world_up = np.array([0, 1, 0])
        
        x_axis = np.cross(world_up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        
        # Create rotation matrix
        rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        # Convert to Euler angles
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(rot_matrix)
        return tuple(r.as_euler('xyz'))
    
    def render_single_view(self, elevation: float, azimuth: float, roll: float,
                          output_path: str) -> bool:
        """
        Render single view of the object
        
        Args:
            elevation: Elevation angle in degrees
            azimuth: Azimuth angle in degrees
            roll: Roll angle in degrees
            output_path: Path to save rendered image
            
        Returns:
            True if rendering successful
        """
        try:
            # Set camera position
            self.set_camera_position(elevation, azimuth, roll)
            
            # Ensure we have a valid camera set
            if bpy.context.scene.camera is None:
                print("Error: No active camera in scene")
                return False
            
            # Set output path
            bpy.context.scene.render.filepath = output_path
            
            # Render
            with SuppressOutput():
                bpy.ops.render.render(write_still=True)
            
            # Verify file was created
            if os.path.exists(output_path):
                return True
            else:
                print(f"Warning: Render completed but file not found at {output_path}")
                return False
            
        except Exception as e:
            print(f"Error rendering view (elev={elevation}, azim={azimuth}): {e}")
            return False
    
    def render_all_views(self, mesh: Mesh3D, output_dir: Path) -> List[str]:
        """
        Render all viewpoints of the mesh
        
        Args:
            mesh: 3D mesh to render
            output_dir: Directory to save rendered images
            
        Returns:
            List of rendered image paths
        """
        print("Setting up Blender scene for rendering...")
        self.setup_blender_scene()
        
        print("Importing mesh to Blender...")
        self.import_mesh_to_blender(mesh)
        
        rendered_paths = []
        
        print(f"Rendering {self.total_views} views ({self.num_elevations} elevations × {self.num_azimuths} azimuths × {self.num_rolls} rolls)...")
        
        view_count = 0
        for i, elevation in enumerate(self.elevations):
            for j, azimuth in enumerate(self.azimuths):
                for k, roll in enumerate(self.rolls):
                    # Create filename
                    filename = f"render_elev_{elevation}_azim_{azimuth}_roll_{roll}.png"
                    output_path = output_dir / filename
                    
                    # Render view
                    success = self.render_single_view(elevation, azimuth, roll, str(output_path))
                    
                    if success:
                        rendered_paths.append(str(output_path))
                        view_count += 1
                        print(f"  Rendered view {view_count}/{self.total_views}", end='\r')
                    else:
                        print(f"  Failed to render view elev={elevation}, azim={azimuth}, roll={roll}")
        
        print()  # New line after progress
        return rendered_paths
    
    def create_comparison_grid(self, rendered_paths: List[str], 
                              output_path: str) -> np.ndarray:
        """
        Create a grid image with numbered annotations
        Grid layout: columns=azimuths, rows=elevations*rolls
        
        Args:
            rendered_paths: List of rendered image paths
            output_path: Path to save the grid image
            
        Returns:
            Grid image as numpy array
        """
        expected_count = self.total_views
        if len(rendered_paths) != expected_count:
            print(f"Warning: Expected {expected_count} rendered images, got {len(rendered_paths)}")
        
        # Load images
        images = []
        for path in rendered_paths:
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.resize(img, (self.image_size, self.image_size))
                    images.append(img)
                else:
                    # Create placeholder if image failed to load
                    placeholder = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                    images.append(placeholder)
            else:
                # Create placeholder if file doesn't exist
                placeholder = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                images.append(placeholder)
        
        # Pad with placeholders if needed
        while len(images) < expected_count:
            placeholder = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            images.append(placeholder)
        
        # Create grid: rows = elevations * rolls, columns = azimuths
        num_rows = self.num_elevations * self.num_rolls
        num_cols = self.num_azimuths
        grid_height = num_rows * self.image_size
        grid_width = num_cols * self.image_size
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Place images in grid with annotations
        for idx in range(len(images)):
            # Calculate grid position from linear index
            row = idx // num_cols
            col = idx % num_cols
            
            if row < num_rows and col < num_cols:
                y_start = row * self.image_size
                y_end = (row + 1) * self.image_size
                x_start = col * self.image_size
                x_end = (col + 1) * self.image_size
                
                # Place image
                grid_image[y_start:y_end, x_start:x_end] = images[idx]
                
                # Add number annotation
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                
                # Position number in top-left corner
                text = str(idx)
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                
                # White background for number
                cv2.rectangle(grid_image, 
                            (x_start + 5, y_start + 5),
                            (x_start + text_size[0] + 15, y_start + text_size[1] + 15),
                            (255, 255, 255), -1)
                
                # Black text
                cv2.putText(grid_image, text, 
                          (x_start + 10, y_start + text_size[1] + 10),
                          font, font_scale, (0, 0, 0), thickness)
        
        # Save grid image
        cv2.imwrite(output_path, grid_image)
        print(f"Created comparison grid: {output_path} ({num_rows} rows × {num_cols} cols)")
        
        return grid_image
    
    def _extract_clip_features(self, image_rgb: np.ndarray) -> torch.Tensor:
        """
        Extract CLIP features from an image
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            CLIP image features as tensor
        """        
        # Process image with CLIP processor
        from PIL import Image
        pil_image = Image.fromarray(image_rgb.astype(np.uint8))
        
        inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def query_best_orientation_clip(self, rendered_paths: List[str], 
                                    cropped_object: np.ndarray) -> int:
        """
        Use CLIP to find the best matching orientation
        
        Args:
            rendered_paths: List of paths to rendered views
            cropped_object: Cropped image of the object from original scene
            
        Returns:
            Index of best matching view (0-23)
        """
        print("Computing CLIP features for cropped object...")
        # Extract features from cropped object, it's already in RGB
        object_features = self._extract_clip_features(cropped_object)
        
        print(f"Computing CLIP features for {len(rendered_paths)} rendered views...")
        # Extract features from all rendered views
        rendered_features = []
        for i, path in enumerate(rendered_paths):
            if os.path.exists(path):
                rendered_img = load_image(path)
                if rendered_img is not None:
                    features = self._extract_clip_features(rendered_img)
                    rendered_features.append(features)
                else:
                    # Create zero features for failed loads
                    rendered_features.append(torch.zeros_like(object_features))
            else:
                # Create zero features for missing files
                rendered_features.append(torch.zeros_like(object_features))
            
            if (i + 1) % 8 == 0:
                print(f"  Processed {i + 1}/{len(rendered_paths)} views", end='\r')
        
        print(f"  Processed {len(rendered_paths)}/{len(rendered_paths)} views")
        
        # Stack all features
        rendered_features = torch.cat(rendered_features, dim=0)  # (N, feature_dim)
        
        # Compute cosine similarities
        similarities = torch.matmul(rendered_features, object_features.T).squeeze()
        
        # Find best match
        best_idx = torch.argmax(similarities).item()
        best_similarity = similarities[best_idx].item()
        
        print(f"Best match: view {best_idx} with similarity {best_similarity:.4f}")
        
        return best_idx
    
    def _compute_angle_difference(self, angle1: float, angle2: float, period: float = 360.0) -> float:
        """
        Compute the minimum angular difference between two angles
        
        Args:
            angle1: First angle in degrees
            angle2: Second angle in degrees
            period: Period of the angle (360 for azimuth, 180 for elevation, etc.)
            
        Returns:
            Minimum angular difference in degrees (0 to period/2)
        """
        # Normalize angles to [0, period)
        angle1 = angle1 % period
        angle2 = angle2 % period
        
        # Compute difference
        diff = abs(angle1 - angle2)
        
        # Take the shorter path around the circle
        if diff > period / 2:
            diff = period - diff
        
        return diff
    
    def _compute_angle_similarity(self, diff: float, max_diff: float) -> float:
        """
        Convert angle difference to similarity score (0-1)
        
        Args:
            diff: Angle difference in degrees
            max_diff: Maximum possible difference (for normalization)
            
        Returns:
            Similarity score (1.0 for identical, 0.0 for maximum difference)
        """
        # Use cosine-based similarity: 1 when diff=0, 0 when diff=max_diff
        return np.cos(np.pi * diff / (2 * max_diff))
    
    def query_best_orientation_orient_anything(self, rendered_paths: List[str],
                                               cropped_object: np.ndarray) -> int:
        """
        Use Orient-Anything to find the best matching orientation (with batch inference)
        
        Args:
            rendered_paths: List of paths to rendered views
            cropped_object: Cropped image of the object from original scene (RGB)
            
        Returns:
            Index of best matching view
        """
        if self.orient_anything_model is None:
            raise RuntimeError("Orient-Anything model not initialized")
        
        print("Predicting orientation using Orient-Anything (batch mode)...")
        
        # Convert numpy array to PIL Image
        from PIL import Image
        pil_image = Image.fromarray(cropped_object.astype(np.uint8))
        
        # Predict orientation from cropped object
        pred_angles = self.orient_anything_model.predict_orientation(pil_image)
        
        pred_azimuth = pred_angles['azimuth']  # 0-360
        pred_elevation = pred_angles['elevation']  # -90 to 90
        pred_roll = pred_angles['roll']  # -180 to 180
        confidence = pred_angles['confidence']
        
        print(f"Target orientation: azimuth={pred_azimuth:.1f}°, elevation={pred_elevation:.1f}°, roll={pred_roll:.1f}° (confidence={confidence:.3f})")
        
        # Normalize predicted roll to 0-360 range to match our rendering convention
        pred_roll_normalized = (pred_roll + 360) % 360
        
        # Load all rendered images for batch processing
        print(f"Loading {len(rendered_paths)} rendered views...")
        rendered_images = []
        valid_indices = []
        
        for idx, path in enumerate(rendered_paths):
            if os.path.exists(path):
                img = load_image(path)
                if img is not None:
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    rendered_images.append(pil_img)
                    valid_indices.append(idx)
        
        if not rendered_images:
            print("Warning: No valid rendered images found")
            return 0
        
        print(f"Batch predicting orientations for {len(rendered_images)} views...")
        # Batch predict orientations for all rendered views
        rendered_predictions = self.orient_anything_model.predict_orientation_batch(rendered_images)
        
        # Compute similarities for all views efficiently
        print("Computing angle-based similarities...")
        similarities = np.zeros(len(rendered_paths))
        
        for i, pred in enumerate(rendered_predictions):
            idx = valid_indices[i]
            
            rendered_azimuth = pred['azimuth']
            rendered_elevation = pred['elevation']
            rendered_roll = (pred['roll'] + 360) % 360  # Normalize to 0-360
            
            # Compute angular differences
            azim_diff = self._compute_angle_difference(pred_azimuth, rendered_azimuth, period=360.0)
            elev_diff = self._compute_angle_difference(pred_elevation, rendered_elevation, period=180.0)
            roll_diff = self._compute_angle_difference(pred_roll_normalized, rendered_roll, period=360.0)
            
            # Convert to similarities (0-1 range)
            azim_sim = self._compute_angle_similarity(azim_diff, max_diff=180.0)
            elev_sim = self._compute_angle_similarity(elev_diff, max_diff=90.0)
            roll_sim = self._compute_angle_similarity(roll_diff, max_diff=180.0)
            
            # Multiply similarities together (geometric mean approach)
            combined_similarity = azim_sim * elev_sim * roll_sim
            
            similarities[idx] = combined_similarity
        
        # Find best match
        best_idx = int(np.argmax(similarities))
        best_similarity = similarities[best_idx]
        
        # Calculate the view angles for logging
        views_per_elevation = self.num_azimuths * self.num_rolls
        elev_idx = best_idx // views_per_elevation
        remainder = best_idx % views_per_elevation
        azim_idx = remainder // self.num_rolls
        roll_idx = remainder % self.num_rolls
        
        print(f"Best match: view {best_idx} with similarity {best_similarity:.4f}")
        print(f"  View angles: azimuth={self.azimuths[azim_idx]:.1f}°, elevation={self.elevations[elev_idx]:.1f}°, roll={self.rolls[roll_idx]:.1f}°")
        
        # Show predicted angles for best matching view
        best_view_local_idx = valid_indices.index(best_idx) if best_idx in valid_indices else -1
        if best_view_local_idx >= 0:
            best_pred = rendered_predictions[best_view_local_idx]
            print(f"  Predicted angles for best view: azimuth={best_pred['azimuth']:.1f}°, elevation={best_pred['elevation']:.1f}°, roll={best_pred['roll']:.1f}°")
        
        return best_idx
    
    def query_best_orientation(self, grid_image: np.ndarray, 
                              cropped_object: np.ndarray) -> int:
        """
        Use Qwen-VL to find the best matching orientation
        
        Args:
            grid_image: Grid of rendered views
            cropped_object: Cropped image of the object from original scene
            
        Returns:
            Index of best matching view (0 to total_views-1)
        """
        num_rows = self.num_elevations * self.num_rolls
        num_cols = self.num_azimuths
        max_index = self.total_views - 1
        
        system_prompt = f"""
You are an expert in 3D object analysis and orientation matching.
You will be shown two images:
1. A grid of {self.total_views} rendered views of a 3D object ({num_rows} rows x {num_cols} columns, numbered 0-{max_index})
2. A cropped image of the same object from a real scene

Your task is to identify which numbered view in the grid has the most similar orientation/pose to the object in the cropped image.

Consider:
- Object orientation and viewing angle
- Visible surfaces and features
- Lighting and shadows (but focus more on geometry)
- Overall pose similarity

Respond with ONLY the number (0-{max_index}) of the best matching view.
"""
        
        user_prompt = f"""
Please analyze the grid of rendered views and the cropped object image.
Identify which numbered view (0-{max_index}) in the grid has the most similar orientation to the object in the cropped image.

Return only the number of the best matching view.
"""
        
        try:
            # Create a combined image for analysis
            # Resize cropped object to match grid cell size
            cropped_resized = cv2.resize(cropped_object, (self.image_size, self.image_size))
            
            # Create combined image: grid on top, cropped object on bottom
            combined_height = grid_image.shape[0] + self.image_size + 20  # 20px spacing
            combined_width = max(grid_image.shape[1], self.image_size)
            combined_image = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
            
            # Place grid image
            combined_image[:grid_image.shape[0], :grid_image.shape[1]] = grid_image
            
            # Add label for reference image
            cv2.putText(combined_image, "Reference Object:", 
                       (10, grid_image.shape[0] + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Place cropped object
            y_start = grid_image.shape[0] + 20
            combined_image[y_start:y_start + self.image_size, :self.image_size] = cropped_resized
            import datetime 
            from imageio.v2 import imsave
            imsave(f"combined_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", combined_image)
            
            # Query Qwen-VL
            response = self.qwen_client.analyze_image(
                combined_image, system_prompt, user_prompt
            )
            
            # Parse response to get the index
            try:
                # Extract number from response
                import re
                numbers = re.findall(r'\b(\d+)\b', response)
                if numbers:
                    best_idx = int(numbers[0])
                    if 0 <= best_idx < self.total_views:
                        return best_idx
                    else:
                        print(f"Warning: Invalid index {best_idx} from Qwen-VL (expected 0-{max_index}), using 0")
                        return 0
                else:
                    print(f"Warning: No valid index found in response: {response}")
                    return 0
                    
            except Exception as e:
                print(f"Error parsing Qwen-VL response: {e}")
                print(f"Raw response: {response}")
                return 0
                
        except Exception as e:
            print(f"Error querying Qwen-VL for orientation: {e}")
            return 0
    
    def get_rotation_from_index(self, index: int) -> np.ndarray:
        """
        Get rotation matrix from view index
        
        Args:
            index: View index (0 to total_views-1)
            
        Returns:
            3x3 rotation matrix for Y-up coordinate system
        """
        if not (0 <= index < self.total_views):
            print(f"Warning: Invalid index {index}, using 0")
            index = 0
        
        # Calculate 3D indices from linear index
        # Index order: elevation -> azimuth -> roll
        views_per_elevation = self.num_azimuths * self.num_rolls
        
        elev_idx = index // views_per_elevation
        remainder = index % views_per_elevation
        azim_idx = remainder // self.num_rolls
        roll_idx = remainder % self.num_rolls
        
        elevation = self.elevations[elev_idx]
        azimuth = self.azimuths[azim_idx]
        roll = self.rolls[roll_idx]
        
        print(f"Selected view {index}: elevation={elevation}°, azimuth={azimuth}°, roll={roll}°")
        
        # Create rotation matrix in Blender's Z-up coordinate system
        # This represents the camera's view, so we need to invert for object rotation
        from scipy.spatial.transform import Rotation
        
        # Camera rotation in Blender (Z-up):
        # 1. Azimuth rotates around Z-axis
        # 2. Elevation tilts the camera up/down
        # 3. Roll rotates around viewing axis
        
        # For object rotation to match camera view, we invert the camera rotation
        rot_azim = Rotation.from_euler('z', -azimuth, degrees=True)
        rot_elev = Rotation.from_euler('x', elevation, degrees=True)  
        rot_roll = Rotation.from_euler('z', -roll, degrees=True)
        
        # Combine rotations (in reverse order for object)
        combined_rotation_zup = rot_roll * rot_elev * rot_azim
        
        # Convert from Blender's Z-up to Y-up coordinate system
        # Z-up to Y-up conversion: swap Y and Z, then negate new Z
        # Rotation matrix for Z-up to Y-up: [[1,0,0], [0,0,1], [0,-1,0]]
        zup_to_yup = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
        
        # Apply coordinate system conversion
        rotation_zup = combined_rotation_zup.as_matrix()
        rotation_yup = zup_to_yup @ rotation_zup @ zup_to_yup.T
        
        return rotation_yup

    def _create_matted_image(self, cropped_image: np.ndarray, cropped_mask: np.ndarray) -> np.ndarray:
         # Create matted image (cropped image with alpha channel for transparency)
        if cropped_mask is not None:
            # Apply mask to create transparency
            mask_3d = cropped_mask[..., np.newaxis] / 255.0
            matted_image = cropped_image * mask_3d
            
            # Convert to RGBA
            alpha_channel = cropped_mask[..., np.newaxis]
            matted_rgba = np.concatenate([matted_image.astype(np.uint8), alpha_channel], axis=2)
        else:
            # Use original cropped image if no mask available
            matted_rgba = cropped_image
        return matted_rgba
    
    def estimate_rotation_render_compare(self, mesh: Mesh3D, 
                                       detected_object: DetectedObject,
                                       output_dir: Path) -> np.ndarray:
        """
        Estimate object rotation using render-and-compare approach
        
        Args:
            mesh: 3D mesh of the object
            detected_object: Detected object with cropped image
            output_dir: Directory to save intermediate results
            
        Returns:
            3x3 rotation matrix
        """
        if detected_object.cropped_image is None:
            print("Warning: No cropped image available for render-compare")
            return np.eye(3)
        
        print(f"Estimating rotation for object {detected_object.id} using render-and-compare ({self.backend} backend)...")
        
        # Create output directory for this object
        render_dir = output_dir / "render_and_compare" / f"obj_{detected_object.id}"
        render_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            # Render all views
            rendered_paths = self.render_all_views(mesh, render_dir.absolute())
            
            if not rendered_paths:
                print("Warning: No views were rendered successfully")
                return np.eye(3)
            
            # Use matted image for better matching
            matted_image = self._create_matted_image(detected_object.cropped_image, detected_object.cropped_mask)
            
            # Query for best match using selected backend
            if self.backend == "clip":
                best_index = self.query_best_orientation_clip(rendered_paths, matted_image)
                
                # Create comparison grid for visualization (optional for CLIP)
                grid_path = render_dir / "comparison_grid.png"
                grid_image = self.create_comparison_grid(rendered_paths, str(grid_path))
            elif self.backend == "orient_anything":
                # Orient-Anything directly predicts angles, no need for grid
                best_index = self.query_best_orientation_orient_anything(rendered_paths, matted_image)
                
                # Create comparison grid for visualization
                grid_path = render_dir / "comparison_grid.png"
                grid_image = self.create_comparison_grid(rendered_paths, str(grid_path))
            else:  # qwen backend
                # Create comparison grid
                grid_path = render_dir / "comparison_grid.png"
                grid_image = self.create_comparison_grid(rendered_paths, str(grid_path))
                
                # Query Qwen-VL for best match
                best_index = self.query_best_orientation(grid_image, matted_image)
            
            # Get rotation matrix from selected index
            rotation_matrix = self.get_rotation_from_index(best_index)
            
            # Calculate elevation, azimuth, roll from best index
            views_per_elevation = self.num_azimuths * self.num_rolls
            elev_idx = best_index // views_per_elevation
            remainder = best_index % views_per_elevation
            azim_idx = remainder // self.num_rolls
            roll_idx = remainder % self.num_rolls
            
            # Save results
            results = {
                "backend": self.backend,
                "best_view_index": best_index,
                "elevation": self.elevations[elev_idx],
                "azimuth": self.azimuths[azim_idx],
                "roll": self.rolls[roll_idx],
                "rotation_matrix": rotation_matrix.tolist(),
                "rendered_views": len(rendered_paths),
                "coordinate_system": "Y-up"
            }
            
            with open(render_dir / "render_compare_results.json", 'w') as f:
                import json
                json.dump(results, f, indent=2)
            
            # Copy the best matching view image for easy access
            if best_index < len(rendered_paths) and os.path.exists(rendered_paths[best_index]):
                import shutil
                best_view_copy = render_dir / "best_match_view.png"
                shutil.copy(rendered_paths[best_index], best_view_copy)
                print(f"Best matching view copied to: {best_view_copy.name}")
            
            print(f"Render-and-compare complete. Selected view {best_index} (elevation={results['elevation']}°, azimuth={results['azimuth']}°, roll={results['roll']}°)")
            return rotation_matrix
            
        except Exception as e:
            print(f"Error in render-and-compare: {e}")
            import traceback
            traceback.print_exc()
            return np.eye(3)
    
    def cleanup_blender_scene(self) -> None:
        """Clean up Blender scene after rendering"""
        try:
            # Clear all objects
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False, confirm=False)
            
            # Clear materials
            for material in bpy.data.materials:
                bpy.data.materials.remove(material)
            
            # Clear meshes
            for mesh in bpy.data.meshes:
                bpy.data.meshes.remove(mesh)
                
        except Exception as e:
            print(f"Warning: Error cleaning up Blender scene: {e}")
