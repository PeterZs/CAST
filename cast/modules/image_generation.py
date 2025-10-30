"""
Image Generation Module

This module handles image generation/completion for detected objects
using Kontext generation model via Replicate API.
"""
import numpy as np
from typing import Optional, List
from pathlib import Path
import cv2

from ..core.common import OCCLUSION_LEVELS, DetectedObject
from ..utils.api_clients import ReplicateClient, QwenVLClient
from ..utils.image_utils import save_image, base64_to_image


class ImageGenerationModule:
    """Module for image generation using Kontext or Qwen"""
    
    def __init__(self, provider: str = "replicate", max_generation_retries: int = 3):
        """
        Initialize the image generation module
        
        Args:
            provider: Generation provider - "replicate" for Kontext or "qwen" for Qwen image edit
            max_generation_retries: Maximum number of generation attempts with quality assessment
        """
        self.provider = provider.lower()
        self.max_retries = max_generation_retries
        
        # Always initialize Qwen client for quality assessment
        self.qwen_client = QwenVLClient()
        
        if self.provider == "replicate":
            self.replicate_client = ReplicateClient()
        elif self.provider == "qwen":
            # Qwen client already initialized
            self.replicate_client = None
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'replicate' or 'qwen'")
    
    def _check_existing_generated_image(self, detected_object: DetectedObject, 
                                       output_dir: Optional[Path]) -> Optional[np.ndarray]:
        """
        Check if a generated image already exists for this object and load it
        
        Args:
            detected_object: Object to check generated image for
            output_dir: Directory where generated images are saved
            
        Returns:
            Loaded image as numpy array if file exists, None otherwise
        """
        if output_dir is None:
            return None
        
        generation_dir = output_dir / "generation" / f"object_{detected_object.id}"
        if not generation_dir.exists():
            return None
        
        # Check for final generated images in order of preference
        possible_filenames = [
            "final_selected.png",  # Best quality image that passed assessment
            "final_best.png",      # Best image from all attempts
        ]
        
        for filename in possible_filenames:
            image_path = generation_dir / filename
            if image_path.exists():
                print(f"  Found existing generated image for object {detected_object.id}: {filename}")
                try:
                    from PIL import Image
                    pil_image = Image.open(image_path)
                    generated_image = np.array(pil_image.convert('RGB'))
                    return generated_image
                except Exception as e:
                    print(f"  Error loading existing image: {e}")
                    return None
        
        return None
    
    def generate_object_image(self, detected_object: DetectedObject, 
                            output_dir: Optional[Path] = None) -> Optional[np.ndarray]:
        """
        Generate/complete an object image using the selected provider
        
        Args:
            detected_object: DetectedObject with cropped and matted image
            output_dir: Optional directory to save results
            
        Returns:
            Generated image or None if generation failed
        """
        if detected_object.cropped_image is None:
            print(f"Warning: No cropped image available for object {detected_object.id}")
            return None
        
        print(f"Generating image for object {detected_object.id}: {detected_object.description} using {self.provider}")
        
        # Create matted image (cropped image with alpha channel for transparency)
        if detected_object.cropped_mask is not None:
            # Apply mask to create transparency
            mask_3d = detected_object.cropped_mask[..., np.newaxis] / 255.0
            matted_image = detected_object.cropped_image * mask_3d
            
            # Convert to RGBA
            alpha_channel = detected_object.cropped_mask[..., np.newaxis]
            matted_rgba = np.concatenate([matted_image.astype(np.uint8), alpha_channel], axis=2)
        else:
            # Use original cropped image if no mask available
            matted_rgba = detected_object.cropped_image
        
        # Generate prompt based on VLM caption (if available) or fallback to description
        # VLM caption is more descriptive and includes visual attributes like color, material, style
        object_description = detected_object.vlm_caption if detected_object.vlm_caption else detected_object.description
        prompt = f"Inpaint the image from visible parts. It's a {object_description}"
        
        try:
            if self.provider == "replicate":
                return self._generate_with_replicate(detected_object, matted_rgba, prompt, output_dir)
            elif self.provider == "qwen":
                return self._generate_with_qwen(detected_object, matted_rgba, prompt, output_dir)
            else:
                print(f"Unsupported provider: {self.provider}")
                return None
                
        except Exception as e:
            print(f"Error during generation for object {detected_object.id}: {e}")
            return None
    
    def _generate_with_replicate(self, detected_object: DetectedObject, matted_rgba: np.ndarray, 
                               prompt: str, output_dir: Optional[Path]) -> Optional[np.ndarray]:
        """Generate image using Replicate Kontext with quality assessment and retry logic"""
        best_image = None
        best_score = 0.0
        
        for attempt in range(1, self.max_retries + 1):
            print(f"Generation attempt {attempt}/{self.max_retries} for object {detected_object.id}")
            
            generation_result = self.replicate_client.run_kontext_generation(
                image=matted_rgba,
                prompt=prompt
            )
            
            if not generation_result:
                print(f"Generation attempt {attempt} failed - no result returned")
                continue
            
            # Handle the response format
            generated_image = None
            
            if hasattr(generation_result, 'read'):
                # File-like object from Replicate
                try:
                    image_data = generation_result.read()
                    from PIL import Image
                    import io
                    pil_image = Image.open(io.BytesIO(image_data))
                    generated_image = np.array(pil_image.convert('RGB'))
                except Exception as e:
                    print(f"Error reading generated image data on attempt {attempt}: {e}")
                    continue
            else:
                print(f"Unexpected generation result format on attempt {attempt}")
                continue

            # Resize to match original cropped image dimensions
            if generated_image is not None:
                h, w = detected_object.cropped_image.shape[:2]
                generated_image = cv2.resize(generated_image, (w, h))
                
                # Assess quality
                assessment = self.qwen_client.assess_inpainted_quality(
                    original_image=detected_object.cropped_image,
                    inpainted_image=generated_image,
                    object_description=detected_object.description
                )
                
                score = assessment.get('score', 0.0)
                passed = assessment.get('passed', False)
                reasoning = assessment.get('reasoning', 'No reasoning provided')
                
                print(f"Attempt {attempt} - Quality score: {score:.2f}/10.0, Passed: {passed}")
                print(f"Assessment reasoning: {reasoning}")
                
                # Save attempt if output directory provided
                if output_dir:
                    generation_dir = output_dir / "generation" / f"object_{detected_object.id}"
                    generation_dir.mkdir(exist_ok=True, parents=True)
                    save_image(generated_image, generation_dir / f"attempt_{attempt}_score_{score:.1f}.png")
                    
                    # Save assessment
                    import json
                    with open(generation_dir / f"attempt_{attempt}_assessment.json", "w") as f:
                        json.dump(assessment, f, indent=2)
                
                # Track best result
                if score > best_score:
                    best_score = score
                    best_image = generated_image
                
                # If quality passed, use this result
                if passed:
                    print(f"Quality assessment passed on attempt {attempt}. Using this result.")
                    if output_dir:
                        save_image(generated_image, generation_dir / "final_selected.png")
                    return generated_image
                else:
                    issues = assessment.get('issues', [])
                    print(f"Quality assessment failed. Issues: {', '.join(issues)}")
        
        # If we exhausted all attempts, use the best one
        if best_image is not None:
            print(f"All {self.max_retries} attempts completed. Using best result with score {best_score:.2f}/10.0")
            if output_dir:
                generation_dir = output_dir / "generation" / f"object_{detected_object.id}"
                save_image(best_image, generation_dir / "final_best.png")
            return best_image
        else:
            print(f"All generation attempts failed for object {detected_object.id}")
            return None
    
    def _generate_with_qwen(self, detected_object: DetectedObject, matted_rgba: np.ndarray,
                          prompt: str, output_dir: Optional[Path]) -> Optional[np.ndarray]:
        """Generate image using Qwen image edit with quality assessment and retry logic"""
        best_image = None
        best_score = 0.0
        
        for attempt in range(1, self.max_retries + 1):
            print(f"Generation attempt {attempt}/{self.max_retries} for object {detected_object.id}")
            
            generation_result = self.qwen_client.run_qwen_image_edit(
                image=matted_rgba,
                prompt=prompt
            )
            
            if not generation_result:
                print(f"Generation attempt {attempt} failed - no result returned")
                continue
            
            try:
                # Handle Qwen response - it should be a URL or base64 data
                generated_image = None
                
                if generation_result.startswith('http'):
                    # It's a URL, download the image
                    import requests
                    response = requests.get(generation_result)
                    if response.status_code == 200:
                        from PIL import Image
                        import io
                        pil_image = Image.open(io.BytesIO(response.content))
                        generated_image = np.array(pil_image.convert('RGB'))
                    else:
                        print(f"Failed to download image on attempt {attempt}: HTTP {response.status_code}")
                        continue
                elif generation_result.startswith('data:image'):
                    # It's base64 data
                    generated_image = base64_to_image(generation_result)
                else:
                    print(f"Unexpected Qwen response format on attempt {attempt}: {type(generation_result)}")
                    continue

                # Resize to match original cropped image dimensions
                if generated_image is not None:
                    h, w = detected_object.cropped_image.shape[:2]
                    generated_image = cv2.resize(generated_image, (w, h))
                    
                    # Assess quality
                    assessment = self.qwen_client.assess_inpainted_quality(
                        original_image=detected_object.cropped_image,
                        inpainted_image=generated_image,
                        object_description=detected_object.description
                    )
                    
                    score = assessment.get('score', 0.0)
                    passed = assessment.get('passed', False)
                    reasoning = assessment.get('reasoning', 'No reasoning provided')
                    
                    print(f"Attempt {attempt} - Quality score: {score:.2f}/10.0, Passed: {passed}")
                    print(f"Assessment reasoning: {reasoning}")
                    
                    # Save attempt if output directory provided
                    if output_dir:
                        generation_dir = output_dir / "generation" / f"object_{detected_object.id}"
                        generation_dir.mkdir(exist_ok=True, parents=True)
                        save_image(generated_image, generation_dir / f"attempt_{attempt}_score_{score:.1f}.png")
                        
                        # Save assessment
                        import json
                        with open(generation_dir / f"attempt_{attempt}_assessment.json", "w") as f:
                            json.dump(assessment, f, indent=2)
                    
                    # Track best result
                    if score > best_score:
                        best_score = score
                        best_image = generated_image
                    
                    # If quality passed, use this result
                    if passed:
                        print(f"Quality assessment passed on attempt {attempt}. Using this result.")
                        if output_dir:
                            save_image(generated_image, generation_dir / "final_selected.png")
                        return generated_image
                    else:
                        issues = assessment.get('issues', [])
                        print(f"Quality assessment failed. Issues: {', '.join(issues)}")
                else:
                    print(f"Failed to process generation result on attempt {attempt}")
                    
            except Exception as e:
                print(f"Error processing generation result on attempt {attempt}: {e}")
                continue
        
        # If we exhausted all attempts, use the best one
        if best_image is not None:
            print(f"All {self.max_retries} attempts completed. Using best result with score {best_score:.2f}/10.0")
            if output_dir:
                generation_dir = output_dir / "generation" / f"object_{detected_object.id}"
                save_image(best_image, generation_dir / "final_best.png")
            return best_image
        else:
            print(f"All generation attempts failed for object {detected_object.id}")
            return None

    def _create_matted_image(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create a matted image from RGB and mask"""
        mask_float = mask[..., np.newaxis] / 255.0
        matted_image = rgb * mask_float
        alpha_channel = mask[..., np.newaxis]
        matted_rgba = np.concatenate([matted_image.astype(np.uint8), alpha_channel], axis=2)
        return matted_rgba

    
    def run(self, detected_objects: List[DetectedObject], generate_threshold: int, 
            output_dir: Optional[Path] = None):
        """
        Run generation for specified objects with instance-level resuming
        
        Args:
            detected_objects: List of all detected objects
            generate_threshold: Minimum occlusion level to trigger generation
            output_dir: Optional directory to save results
            
        Returns:
            Updated detected objects with generated images
        """
        print(f"Starting image generation pipeline using {self.provider}...")
        
        # Generate images for occluded objects with instance-level resuming
        for obj in detected_objects:
            if OCCLUSION_LEVELS.get(obj.occlusion_level, 2) >= generate_threshold:
                # Check if generated image already exists
                existing_image = self._check_existing_generated_image(obj, output_dir)
                if existing_image is not None:
                    print(f"Skipping image generation for object {obj.id}: {obj.description} (already exists)")
                    obj.generated_image = existing_image
                else:
                    generated_image = self.generate_object_image(
                        obj, output_dir=output_dir
                    )
                    if generated_image is not None:
                        obj.generated_image = generated_image
                    else:
                        # the generated image will ONLY be used for mesh generation
                        # if it's NOT created, we mat original cropped image 
                        obj.generated_image = self._create_matted_image(obj.cropped_image, obj.cropped_mask)
                        print(f"Failed to generate image for object {obj.id} ({obj.description}). Create the matted image instead.")
            else:
                obj.generated_image = self._create_matted_image(obj.cropped_image, obj.cropped_mask)
                print(f"Skipping generation for object {obj.id} ({obj.description}) due to occlusion level {obj.occlusion_level}. Create the matted image instead.")
        
        print("Image generation pipeline complete.")
        return detected_objects