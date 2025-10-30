"""
Configuration settings for CAST pipeline
"""
import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class APIConfig:
    """API configuration settings"""
    replicate_token: Optional[str] = None
    tripo3d_key: Optional[str] = None
    dashscope_key: Optional[str] = None
    
    def __post_init__(self):
        # Load from environment variables
        self.replicate_token = os.getenv("REPLICATE_API_TOKEN")
        self.tripo3d_key = os.getenv("TRIPO3D_API_KEY")
        self.dashscope_key = os.getenv("DASHSCOPE_API_KEY")


@dataclass
class ModelConfig:
    """Model configuration settings"""
    # Local Grounded-SAM paths
    grounding_dino_config: str = "thirdparty/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounding_dino_checkpoint: str = "thirdparty/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
    ram_checkpoint: str = "thirdparty/Grounded-Segment-Anything/ram_swin_large_14m.pth"
    sam_checkpoint: str = "thirdparty/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
    sam_hq_checkpoint: Optional[str] = None
    
    # Detection thresholds
    box_threshold: float = 0.25
    text_threshold: float = 0.2
    iou_threshold: float = 0.5

    # Stable Diffusion Inpainting settings
    inpainting_model: str = "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3"
    
    # MoGe settings
    moge_model: str = "Ruicheng/moge-2-vitl-normal"
    
    # Qwen-VL settings
    qwen_model: str = "qwen-vl-max-latest"

@dataclass
class ProcessingConfig:
    """Processing configuration settings"""
    # ICP parameters
    icp_max_iterations: int = 1000
    icp_tolerance: float = 1e-6
    icp_fitness_threshold: float = 0.3
    
    # Scene graph optimization parameters (using Open3D SDF)
    sdf_learning_rate: float = 0.01
    sdf_max_iterations: int = 500
    sdf_penetration_weight: float = 1.0
    sdf_contact_weight: float = 0.5

class Config:
    """Main configuration class"""
    def __init__(self):
        self.api = APIConfig()
        self.models = ModelConfig()
        self.processing = ProcessingConfig()
        
    def validate(self) -> bool:
        """Validate configuration"""
        # Check required API keys (Replicate no longer required for detection)
        required_keys = [
            self.api.replicate_token,
            self.api.tripo3d_key,
            self.api.dashscope_key
        ]
        
        missing_keys = [key for key in required_keys if key is None]
        if missing_keys:
            print("Warning: Missing API keys. Please set environment variables.")
            return False
        
        # Check local model paths exist
        from pathlib import Path
        model_paths = [
            Path(self.models.grounding_dino_config),
            Path(self.models.grounding_dino_checkpoint),
            Path(self.models.ram_checkpoint),
            Path(self.models.sam_checkpoint)
        ]
        
        missing_models = [str(p) for p in model_paths if not p.exists()]
        if missing_models:
            print(f"Warning: Missing model files: {missing_models}")
            print("Please ensure Grounded-SAM models are downloaded.")
            return False
        
        return True

# Global config instance
config = Config()