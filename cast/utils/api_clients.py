"""
API client wrappers for external services
"""
import os
import requests
import time
import base64
import asyncio
import mimetypes
import dashscope
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import replicate
from httpx import Timeout
from openai import OpenAI

from ..config.settings import config
from .image_utils import image_to_base64
import numpy as np

dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

REPLICATE_TIMEOUT = Timeout(
    10080.0,  # default timeout
    read=10080.0,  # 3 hour minutes read timeout
    write=600.0,  # write timeout
    connect=600.0,  # connect timeout
    pool=600.0  # pool timeout
)


class ReplicateClient:
    """Wrapper for Replicate API calls"""

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or config.api.replicate_token
        self.client = replicate.Client(api_token=self.api_token, timeout=600)

    def run_ram_grounded_sam(self,
                             image: Union[np.ndarray, str],
                             use_sam_hq: bool = True,
                             show_visualisation: bool = False) -> Dict[str, Any]:
        """
        Run RAM-Grounded-SAM for combined object recognition, detection and segmentation
        
        Args:
            image: Input image as numpy array or base64 string
            use_sam_hq: Use sam_hq instead of SAM for prediction (default: False)
            show_visualisation: Output bounding box and masks on the image (default: False)
            
        Returns:
            Dictionary containing:
            - tags: String of detected tags
            - json_data: List of detected objects with bounding boxes and labels
            - masked_img: Base64 encoded masked image (if show_visualisation=True)
            - rounding_box_img: Base64 encoded image with bounding boxes (if show_visualisation=True)
        """

        async def run_core(model_name: str, input: Any):
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(self.client.async_run(model_name, input=input))]
                return await asyncio.gather(*tasks)

        if isinstance(image, np.ndarray):
            image_b64 = image_to_base64(image)
        else:
            image_b64 = image

        try:
            # loop = asyncio.get_event_loop()
            output = asyncio.run(
                run_core(
                    config.models.ram_grounded_sam_model,
                    input={
                        "input_image": image_b64,
                        "use_sam_hq": use_sam_hq,
                        "show_visualisation": show_visualisation
                    },
                ))[0]
            return output
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error running RAM-Grounded-SAM: {e}")
            return {}

    def run_inpainting(self,
                       image: Union[np.ndarray, str],
                       mask: Union[np.ndarray, str],
                       prompt: str = "high quality, detailed") -> str:
        """Run stable diffusion inpainting"""

        async def run_core(model_name: str, input: Any):
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(self.client.async_run(model_name, input=input))]
                return await asyncio.gather(*tasks)

        if isinstance(image, np.ndarray):
            image_b64 = image_to_base64(image)
        else:
            image_b64 = image

        if isinstance(mask, np.ndarray):
            mask_b64 = image_to_base64(mask)
        else:
            mask_b64 = mask

        try:
            output = asyncio.run(
                run_core(config.models.sd_inpainting_model,
                         input={
                             "image": image_b64,
                             "mask": mask_b64,
                             "prompt": prompt,
                             "num_inference_steps": 20,
                             "guidance_scale": 7.5,
                             "strength": 0.99
                         }))[0]
            return output
        except Exception as e:
            print(f"Error running stable diffusion inpainting: {e}")
            return None

    def run_kontext_generation(self, image: Union[np.ndarray, str], prompt: str = "complete the image") -> str:
        """
        Run Flux Kontext for image generation/completion
        
        Args:
            image: Input image as numpy array or base64 string  
            prompt: Generation prompt
            
        Returns:
            Generated image URL or base64 string
        """

        async def run_core(model_name: str, input: Any):
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(self.client.async_run(model_name, input=input))]
                return await asyncio.gather(*tasks)

        if isinstance(image, np.ndarray):
            image_b64 = image_to_base64(image)
        else:
            image_b64 = image

        try:
            loop = asyncio.get_event_loop()
            output = loop.run_until_complete(
                run_core("black-forest-labs/flux-kontext-dev",
                         input={
                             "input_image": image_b64,
                             "prompt": prompt,
                             "num_inference_steps": 30,
                         }))[0]
            return output
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error running Kontext generation: {e}")
            return None


class Tripo3DClient:
    """Wrapper for Tripo3D API calls"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.api.tripo3d_key
        self.base_url = "https://api.tripo3d.ai/v2/openapi"

    def upload_image(self, image: Union[np.ndarray, str, Path]) -> str:
        """Upload image and get file token"""
        if isinstance(image, np.ndarray):
            # Convert to bytes for upload
            import io
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(image)
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
        else:
            # Read from file
            with open(image, 'rb') as f:
                img_bytes = f.read()

        headers = {"Authorization": f"Bearer {self.api_key}"}

        files = {"file": ("image.png", img_bytes, "image/png")}

        try:
            response = requests.post(f"{self.base_url}/upload", headers=headers, files=files)
            response.raise_for_status()
            return response.json()["data"]["image_token"]
        except Exception as e:
            print(f"Error uploading image to Tripo3D: {e}")
            return ""

    def create_3d_model(self, file_token: str) -> str:
        """Create 3D model from uploaded image"""
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        data = {
            "type": "image_to_model",
            "file": {
                "type": "png",
                "file_token": file_token
            },
            "model_version": "v3.0-20250812",
        }

        try:
            response = requests.post(f"{self.base_url}/task", headers=headers, json=data)
            response.raise_for_status()
            return response.json()["data"]["task_id"]
        except Exception as e:
            print(f"Error creating 3D model: {e}")
            return ""

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status and results"""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.get(f"{self.base_url}/task/{task_id}", headers=headers)
            response.raise_for_status()
            return response.json()["data"]
        except Exception as e:
            print(f"Error getting task status: {e}")
            return {}

    def wait_for_completion(self, task_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for task completion with timeout"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_task_status(task_id)

            if status.get("status") == "success":
                return status
            elif status.get("status") == "failed":
                raise Exception(f"Task failed: {status.get('error', 'Unknown error')}")

            time.sleep(5)  # Wait 5 seconds before checking again

        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")

    def download_model(self, download_url: str, output_path: Path) -> bool:
        """Download 3D model file"""
        try:
            response = requests.get(download_url)
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(response.content)

            return True
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False


class QwenVLClient:
    """Wrapper for Qwen-VL API calls"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.api.dashscope_key
        self.client = OpenAI(api_key=self.api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    def analyze_scene_graph(self, image: Union[np.ndarray, str], system_prompt: str, user_prompt: str) -> str:
        """Analyze image to extract scene graph relationships"""
        if isinstance(image, np.ndarray):
            image_url = image_to_base64(image)
        else:
            image_url = image

        try:
            completion = self.client.chat.completions.create(model=config.models.qwen_model,
                                                             messages=[{
                                                                 "role": "system",
                                                                 "content": [{
                                                                     "type": "text",
                                                                     "text": system_prompt
                                                                 }]
                                                             }, {
                                                                 "role":
                                                                     "user",
                                                                 "content": [{
                                                                     "type": "image_url",
                                                                     "image_url": {
                                                                         "url": image_url
                                                                     }
                                                                 }, {
                                                                     "type": "text",
                                                                     "text": user_prompt
                                                                 }]
                                                             }])

            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error analyzing scene graph: {e}")
            return ""

    def analyze_image(self, image: Union[np.ndarray, str], system_prompt: str, user_prompt: str) -> str:
        """
        General image analysis using Qwen-VL
        
        Args:
            image: Input image as numpy array or base64 string
            system_prompt: System prompt for the task
            user_prompt: User prompt for the task
            
        Returns:
            Response from Qwen-VL
        """
        if isinstance(image, np.ndarray):
            image_url = image_to_base64(image)
        else:
            image_url = image

        try:
            completion = self.client.chat.completions.create(model=config.models.qwen_model,
                                                             messages=[{
                                                                 "role": "system",
                                                                 "content": [{
                                                                     "type": "text",
                                                                     "text": system_prompt
                                                                 }]
                                                             }, {
                                                                 "role":
                                                                     "user",
                                                                 "content": [{
                                                                     "type": "image_url",
                                                                     "image_url": {
                                                                         "url": image_url
                                                                     }
                                                                 }, {
                                                                     "type": "text",
                                                                     "text": user_prompt
                                                                 }]
                                                             }])

            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return ""

    def filter_detections(self, original_image: Union[np.ndarray, str], annotated_image: Union[np.ndarray, str],
                          detection_data: Dict[str, Any]) -> str:
        """
        Filter object detections to remove spurious and scene-level detections
        while also assessing occlusion levels
        
        Args:
            original_image: Original RGB image
            annotated_image: Image with numbered bounding boxes
            detection_data: Detection data with bounding boxes and labels
            
        Returns:
            JSON string with filtered object IDs, occlusion levels, and reasoning
        """
        if isinstance(original_image, np.ndarray):
            original_url = image_to_base64(original_image)
        else:
            original_url = original_image

        if isinstance(annotated_image, np.ndarray):
            annotated_url = image_to_base64(annotated_image)
        else:
            annotated_url = annotated_image

        # Create system prompt for detection filtering with occlusion assessment
        system_prompt = """You are an expert computer vision system for filtering object detections and assessing occlusion levels. Your task is to identify meaningful constituent objects, remove spurious detections, rate occlusion levels, and provide concise captions for kept objects.

Rules for filtering:
1. Remove scene-level detections (e.g., room, sky, ground, wall, background)
2. Remove background elements that are not distinct objects
3. Remove the detection that covers a number of independent objects that are also detected and INDICATE NO EXTRA NEW OBJECT (e.g. a box includes 2 books which are also independently detected out)
4. Based on the previous rule, remove parts of objects when the whole object is also detected (e.g., wheel of a car if car is detected)
5. Remove the objects that has a large invisible part in the image (e.g., a table that has only a support plane in the image)
6. Remove objects that are two small or unimportant in the image.
7. Keep meaningful, distinct objects that can be isolated and reconstructed in 3D. Multiple objects of the same type are allowed.
8. Prioritize objects that have clear boundaries and can be manipulated independently

Occlusion Level Assessment:
For each object you decide to keep, assess its occlusion level(Self-Occlusion is NOT considered):
- "no_occlusion": Object is completely visible with NO parts hidden (less than 1%)
- "some_occlusion": Object has little/moderate occlusion (1-50% part occluded)
- "severe_occlusion": Object is heavily occluded, only small parts visible (more than 50% part occluded)

Caption Generation:
For each object you keep, generate a concise, descriptive caption (3-8 words) that captures:
- The object's visual appearance (color, material, style)
- Its key characteristics or distinguishing features or quantity
- Any notable attributes visible in the image
Example: "red leather office chair" or "modern silver laptop computer"

You must respond with a JSON object containing:
- "keep": list of objects to keep, each with {"id": int, "occlusion_level": str, "caption": str}
- "remove": list of object IDs to remove with reasons
- "reasoning": overall reasoning for the filtering decisions"""

        # Create user prompt with detection data
        user_prompt = f"""Analyze these object detections and filter out spurious/scene-level detections.
For each object you decide to keep, assess its occlusion level and generate a concise descriptive caption.

Detection data:
{detection_data}

Please examine both the original image and the annotated image with numbered bounding boxes. Filter the detections to keep only meaningful constituent objects that can be reconstructed in 3D, rate their occlusion levels, and provide descriptive captions for each kept object.

Return your response as a JSON object with "keep" (including id, occlusion_level, and caption for each object), "remove", and "reasoning" fields."""

        try:
            completion = self.client.chat.completions.create(model=config.models.qwen_model,
                                                             messages=[{
                                                                 "role": "system",
                                                                 "content": [{
                                                                     "type": "text",
                                                                     "text": system_prompt
                                                                 }]
                                                             }, {
                                                                 "role":
                                                                     "user",
                                                                 "content": [{
                                                                     "type": "image_url",
                                                                     "image_url": {
                                                                         "url": original_url
                                                                     }
                                                                 }, {
                                                                     "type": "image_url",
                                                                     "image_url": {
                                                                         "url": annotated_url
                                                                     }
                                                                 }, {
                                                                     "type": "text",
                                                                     "text": user_prompt
                                                                 }]
                                                             }])

            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error filtering detections: {e}")
            return ""

    def run_qwen_image_edit(self,
                            image: Union[np.ndarray, str],
                            prompt: str = "Inpaint the image from visible parts") -> Optional[str]:
        """
        Run Qwen image editing/inpainting
        
        Args:
            image: Input image as numpy array or base64 string
            prompt: Inpainting prompt
            
        Returns:
            Generated image URL/data or None if generation failed
        """
        try:
            from dashscope import MultiModalConversation

            # Convert image to base64 format expected by Qwen
            if isinstance(image, np.ndarray):
                # Save image temporarily to get proper base64 encoding
                from PIL import Image as PILImage
                import io
                import tempfile

                # Convert numpy array to PIL Image
                if image.shape[2] == 4:  # RGBA
                    pil_image = PILImage.fromarray(image, 'RGBA')
                else:  # RGB
                    pil_image = PILImage.fromarray(image, 'RGB')

                # Save to temporary file to get proper MIME type
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    pil_image.save(tmp_file.name, format='PNG')
                    image_b64 = self._encode_file_for_qwen(tmp_file.name)

                # Clean up temp file
                import os
                os.unlink(tmp_file.name)
            else:
                # Assume it's already a file path or base64
                if image.startswith('data:'):
                    image_b64 = image
                else:
                    image_b64 = self._encode_file_for_qwen(image)

            messages = [{"role": "user", "content": [{"image": image_b64}, {"text": prompt}]}]

            response = MultiModalConversation.call(api_key=self.api_key,
                                                   model="qwen-image-edit",
                                                   messages=messages,
                                                #    result_format='message',
                                                   stream=False,
                                                   watermark=True,
                                                   negative_prompt="")

            if response.status_code == 200:
                # Extract the generated image from response
                if hasattr(response, 'output') and response.output:
                    if hasattr(response.output, 'choices') and response.output.choices:
                        choice = response.output.choices[0]
                        if hasattr(choice, 'message') and choice.message:
                            if hasattr(choice.message, 'content') and choice.message.content:
                                for content in choice.message.content:
                                    if (hasattr(content, 'image') and content.image):
                                        return content.image
                                    elif 'image' in content and content['image']:
                                        return content['image']
                import ipdb; ipdb.set_trace()
                print("No image found in Qwen response")
                return None
            else:
                print(f"Qwen API error - HTTP {response.status_code}: {response.code} - {response.message}")
                return None

        except ImportError:
            print("dashscope package not available. Please install: pip install dashscope")
            return None
        except Exception as e:
            print(f"Error running Qwen image edit: {e}")
            return None

    def _encode_file_for_qwen(self, file_path: str) -> str:
        """
        Encode file for Qwen API in the required format
        Format: data:{MIME_type};base64,{base64_data}
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError("Unsupported or unrecognizable image format")

        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        return f"data:{mime_type};base64,{encoded_string}"
    
    def assess_inpainted_quality(self, 
                                 original_image: Union[np.ndarray, str],
                                 inpainted_image: Union[np.ndarray, str],
                                 object_description: str) -> Dict[str, Any]:
        """
        Assess the quality of an inpainted/generated image
        
        Args:
            original_image: Original cropped object image (possibly with occlusion)
            inpainted_image: Generated/inpainted image
            object_description: Description of the object
            
        Returns:
            Dictionary with:
            - passed: bool, whether quality assessment passed
            - score: float, quality score (0-10)
            - reasoning: str, explanation of the assessment
        """
        if isinstance(original_image, np.ndarray):
            original_url = image_to_base64(original_image)
        else:
            original_url = original_image
            
        if isinstance(inpainted_image, np.ndarray):
            inpainted_url = image_to_base64(inpainted_image)
        else:
            inpainted_url = inpainted_image
        
        system_prompt = """You are an expert image quality assessor for evaluating inpainted/generated object images.
Your task is to assess whether the generated complete object image is realistic, coherent, and suitable for 3D reconstruction.

Evaluation Criteria:
1. **Completeness**: Is the object fully visible with no missing parts?
2. **Coherence**: Does the generated part blend naturally with the original visible part?
3. **Realism**: Does the generated image look realistic and physically plausible?
4. **Structural Integrity**: Are the object's proportions and structure correct?
5. **Quality**: Is the image clear without artifacts, distortions, or unnatural features?

Scoring Guidelines:
- 9-10: Excellent quality, perfect for 3D reconstruction
- 7-8: Good quality, minor issues but acceptable
- 5-6: Moderate quality, noticeable issues but usable
- 3-4: Poor quality, significant issues
- 0-2: Very poor quality, unusable

Pass Threshold: Score >= 6.0

You must respond with a JSON object containing:
- "score": float (0-10)
- "passed": bool (true if score >= 6.0)
- "reasoning": string explaining the assessment
- "issues": list of specific problems identified (empty if passed)"""

        user_prompt = f"""Assess the quality of the generated/inpainted image for this object: "{object_description}"

The first image shows the original object (may have occlusions or missing parts).
The second image shows the generated complete object.

Please evaluate:
1. Does the generated image complete the object naturally?
2. Is the generated part realistic and coherent with the visible part?
3. Is the quality sufficient for 3D reconstruction?
4. Are there any artifacts, distortions, or unnatural features?

Provide your assessment as a JSON object with score (0-10), passed (bool), reasoning (string), and issues (list)."""

        try:
            completion = self.client.chat.completions.create(
                model=config.models.qwen_model,
                messages=[{
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": system_prompt
                    }]
                }, {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": original_url}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": inpainted_url}
                        },
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }]
            )
            
            response_text = completion.choices[0].message.content
            
            # Parse JSON response
            import json
            # Clean up markdown code blocks if present
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            result = json.loads(response_text)
            
            # Ensure required fields exist
            if 'passed' not in result:
                result['passed'] = result.get('score', 0) >= 7.0
            if 'issues' not in result:
                result['issues'] = []
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"Error parsing Qwen assessment response: {e}")
            print(f"Response was: {response_text}")
            # Default to failed assessment
            return {
                "passed": False,
                "score": 0.0,
                "reasoning": "Failed to parse assessment response",
                "issues": ["Assessment parsing error"]
            }
        except Exception as e:
            print(f"Error assessing image quality: {e}")
            return {
                "passed": False,
                "score": 0.0,
                "reasoning": f"Assessment error: {str(e)}",
                "issues": ["Assessment system error"]
            }


class TrellisClient:
    """Wrapper for TRELLIS 3D generation via Replicate with async and sync support"""

    def __init__(self, api_token: Optional[str] = None, base_url: Optional[str] = None):
        self.api_token = api_token or config.api.replicate_token
        self.base_url = base_url  # No default - None means use Replicate

        # Determine if this is a local deployment
        self.is_local = self._is_local_deployment(self.base_url) if self.base_url else False

        if self.is_local:
            self.client = replicate.Client(api_token=self.api_token, base_url=self.base_url, timeout=300)
            print(f"TrellisClient initialized for local deployment at {self.base_url}")
        else:
            self.client = replicate.Client(api_token=self.api_token, timeout=300)
            print("TrellisClient initialized for remote Replicate deployment")

        self.model_version = "firtoz/trellis:e8f6c45206993f297372f5436b90350817bd9b4a0d52d2a76df50c1c8afa2b3c"

    def _is_local_deployment(self, base_url: str) -> bool:
        """
        Determine if the base_url points to a local deployment
        
        Args:
            base_url: The base URL to check
            
        Returns:
            True if it's a local deployment, False otherwise
        """
        if not base_url:
            return False

        # Extract hostname from URL
        import urllib.parse
        parsed = urllib.parse.urlparse(base_url)
        hostname = parsed.hostname

        if not hostname:
            return False

        # Check for local IP patterns
        local_patterns = ["localhost", "127.0.0.1", "0.0.0.0"]

        # Check for private IP ranges (192.168.x.x, 10.x.x.x, 172.16-31.x.x)
        if hostname.startswith("192.168.") or hostname.startswith("10."):
            return True

        # Check 172.16.0.0 to 172.31.255.255 range
        if hostname.startswith("172."):
            try:
                second_octet = int(hostname.split('.')[1])
                if 16 <= second_octet <= 31:
                    return True
            except (ValueError, IndexError):
                pass

        return hostname in local_patterns

    def create_trellis_input(self,
                             image: Union[np.ndarray, str],
                             texture_size: int = 1024,
                             mesh_simplify: float = 0.9,
                             ss_sampling_steps: int = 12,
                             slat_sampling_steps: int = 12,
                             generate_model: bool = True,
                             generate_color: bool = False,
                             save_gaussian_ply: bool = False) -> Dict[str, Any]:
        """
        Create input dictionary for TRELLIS model
        
        Args:
            image: Input image as numpy array or base64 string
            texture_size: Texture resolution (default: 1024)
            mesh_simplify: Mesh simplification ratio (default: 0.9)
            ss_sampling_steps: Sampling steps for sparse structure (default: 12)
            slat_sampling_steps: Sampling steps for SLAT (default: 12)
            generate_model: Whether to generate 3D model (default: True)
            generate_color: Whether to generate color texture (default: False)
            save_gaussian_ply: Whether to save Gaussian PLY (default: False)
            
        Returns:
            Input dictionary for TRELLIS model
        """
        if isinstance(image, np.ndarray):
            image_b64 = image_to_base64(image)
        else:
            image_b64 = image

        return {
            "images": [image_b64],
            "texture_size": texture_size,
            "mesh_simplify": mesh_simplify,
            "generate_model": generate_model,
            "save_gaussian_ply": save_gaussian_ply,
            "ss_sampling_steps": ss_sampling_steps,
            "slat_sampling_steps": slat_sampling_steps,
            "generate_color": generate_color,
        }

    async def generate_3d_async(self, image: Union[np.ndarray, str], **kwargs) -> str:
        """
        Generate 3D model asynchronously using TRELLIS
        
        Args:
            image: Input image
            **kwargs: Additional parameters for TRELLIS
            
        Returns:
            Prediction ID for tracking the job
        """
        input_data = self.create_trellis_input(image, **kwargs)

        try:
            # Create async prediction
            prediction = await self.client.predictions.async_create(version=self.model_version, input=input_data)

            print(f"Started TRELLIS 3D generation with prediction ID: {prediction.id}")
            return prediction.id

        except Exception as e:
            print(f"Error starting TRELLIS generation: {e}")
            return ""

    def generate_3d_sync(self, image: Union[np.ndarray, str], **kwargs) -> Optional[Dict[str, Any]]:
        """
        Generate 3D model synchronously using TRELLIS (for local deployments)
        
        Args:
            image: Input image
            **kwargs: Additional parameters for TRELLIS
            
        Returns:
            Result dictionary with model file URL or None if failed
        """
        if not self.is_local:
            raise RuntimeError(
                "Synchronous generation is only supported for local deployments. Use generate_3d_async for remote deployments."
            )

        input_data = self.create_trellis_input(image, **kwargs)

        try:
            print("Starting synchronous TRELLIS 3D generation...")
            start_time = time.time()

            # Run synchronous prediction for local deployment
            output = self.client.run(self.model_version, input=input_data)

            end_time = time.time()
            print(f"TRELLIS generation completed in {end_time - start_time:.2f} seconds")

            return {"output": output}

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in synchronous TRELLIS generation: {e}")
            return None

    async def get_prediction_status(self, prediction_id: str) -> Dict[str, Any]:
        """
        Get the status of an async prediction
        
        Args:
            prediction_id: The prediction ID to check
            
        Returns:
            Dictionary with prediction status and results
        """
        try:
            prediction = await replicate.predictions.async_get(prediction_id)
            return {
                "id": prediction.id,
                "status": prediction.status,
                "output": prediction.output,
                "error": prediction.error,
                "logs": prediction.logs
            }
        except Exception as e:
            print(f"Error getting prediction status: {e}")
            return {"status": "error", "error": str(e)}

    async def wait_for_completion_async(self,
                                        prediction_id: str,
                                        timeout: int = 600,
                                        check_interval: int = 5) -> Dict[str, Any]:
        """
        Wait for async prediction to complete
        
        Args:
            prediction_id: The prediction ID to wait for
            timeout: Maximum time to wait in seconds
            check_interval: How often to check status in seconds
            
        Returns:
            Final prediction result
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = await self.get_prediction_status(prediction_id)

            if status["status"] == "succeeded":
                return status
            elif status["status"] == "failed":
                raise Exception(f"TRELLIS generation failed: {status.get('error', 'Unknown error')}")
            elif status["status"] == "canceled":
                raise Exception("TRELLIS generation was canceled")

            await asyncio.sleep(check_interval)

        raise TimeoutError(f"TRELLIS generation did not complete within {timeout} seconds")

    async def batch_generate_async(self, images: List[Union[np.ndarray, str]], **kwargs) -> List[str]:
        """
        Start multiple async 3D generations
        
        Args:
            images: List of input images
            **kwargs: Additional parameters for TRELLIS
            
        Returns:
            List of prediction IDs
        """
        prediction_ids = []

        for i, image in enumerate(images):
            print(f"Starting generation {i+1}/{len(images)}")
            pred_id = await self.generate_3d_async(image, **kwargs)
            if pred_id:
                prediction_ids.append(pred_id)

            # Small delay to avoid overwhelming the API
            await asyncio.sleep(0.5)

        print(f"Started {len(prediction_ids)} async TRELLIS generations")
        return prediction_ids

    async def collect_results_async(self,
                                    prediction_ids: List[str],
                                    obj_ids: List[int],
                                    output_dir: Optional[Path] = None) -> List[Optional[str]]:
        """
        Collect results from multiple async predictions
        
        Args:
            prediction_ids: List of prediction IDs to collect
            output_dir: Optional directory to save downloaded models
            
        Returns:
            List of model file paths (None for failed generations)
        """
        results = []

        # Wait for all predictions to complete
        tasks = [self.wait_for_completion_async(pred_id) for pred_id in prediction_ids]
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and download models
        for i, (pred_id, result) in enumerate(zip(prediction_ids, completed_results)):
            if isinstance(result, Exception):
                print(f"Generation {i+1} failed: {result}")
                results.append(None)
                continue

            try:
                if result["output"] and "model_file" in result["output"]:
                    model_url = result["output"]["model_file"]

                    if output_dir:
                        model_path = output_dir / f"object_{obj_ids[i]}_trellis.glb"
                        if await self._download_model_async(model_url, model_path):
                            results.append(str(model_path))
                        else:
                            results.append(None)
                    else:
                        results.append(model_url)
                else:
                    print(f"No model file in result for prediction {pred_id}")
                    results.append(None)

            except Exception as e:
                print(f"Error processing result for prediction {pred_id}: {e}")
                results.append(None)

        successful = sum(1 for r in results if r is not None)
        print(f"Collected {successful}/{len(prediction_ids)} successful TRELLIS results")

        return results

    def batch_generate_sync(self,
                            images: List[Union[np.ndarray, str]],
                            obj_ids: List[int],
                            output_dir: Optional[Path] = None,
                            **kwargs) -> List[Optional[str]]:
        """
        Generate multiple 3D models synchronously (for local deployments)
        
        Args:
            images: List of input images
            output_dir: Optional directory to save downloaded models
            **kwargs: Additional parameters for TRELLIS
            
        Returns:
            List of model file paths (None for failed generations)
        """
        if not self.is_local:
            raise RuntimeError("Synchronous batch generation is only supported for local deployments.")

        results = []

        for i, image in enumerate(images):
            print(f"Processing image {i+1}/{len(images)} synchronously...")

            result = self.generate_3d_sync(image, **kwargs)

            if result and result["output"] and "model_file" in result["output"]:
                model_url = result["output"]["model_file"]

                if output_dir:
                    model_path = output_dir / f"object_{obj_ids[i]}_trellis.glb"
                    try:
                        with open(model_path, "wb") as f:
                            f.write(model_url.read())
                        if os.path.exists(model_path):
                            results.append(str(model_path))
                        else:
                            results.append(None)
                    except Exception as download_error:
                        print(f"Failed to download model for image {i+1}: {download_error}")
                        results.append(None)
                else:
                    results.append(model_url)
            else:
                print(f"Failed to generate model for image {i+1}")
                results.append(None)

            # Small delay between requests to be respectful
            time.sleep(1)

        successful = sum(1 for r in results if r is not None)
        print(f"Completed synchronous batch generation: {successful}/{len(images)} successful")

        return results

    async def _download_model_async(self, model_url: str, output_path: Path) -> bool:
        """
        Download model file asynchronously
        
        Args:
            model_url: URL of the model file
            output_path: Local path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import aiohttp

            output_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiohttp.ClientSession() as session:
                async with session.get(model_url) as response:
                    if response.status == 200:
                        with open(output_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        print(f"Downloaded TRELLIS model to: {output_path}")
                        return True
                    else:
                        print(f"Failed to download model: HTTP {response.status}")
                        return False

        except ImportError:
            print("aiohttp not available, falling back to sync download")
            return self._download_model_sync(model_url, output_path)
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False

    def _download_model_sync(self, model_url: str, output_path: Path) -> bool:
        """
        Download model file synchronously (fallback)
        
        Args:
            model_url: URL of the model file
            output_path: Local path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            response = requests.get(model_url)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                f.write(response.content)

            print(f"Downloaded TRELLIS model to: {output_path}")
            return True

        except Exception as e:
            print(f"Error downloading model: {e}")
            return False


class Hunyuan3DClient:
    """Wrapper for Hunyuan3D-Omni geometry generation with point cloud conditioning"""

    def __init__(self, repo_id: str = "tencent/Hunyuan3D-Omni", use_ema: bool = False, fast_decode: bool = False):
        """
        Initialize Hunyuan3D-Omni client
        
        Args:
            repo_id: HuggingFace model repository ID
            use_ema: Use EMA model for inference
            fast_decode: Use FlashVDM for faster decoding
        """
        self.repo_id = repo_id
        self.use_ema = use_ema
        self.fast_decode = fast_decode
        self.pipeline = None
        self.device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    def load_model(self):
        """Load the Hunyuan3D-Omni pipeline"""
        if self.pipeline is not None:
            return True

        try:
            import sys
            import warnings
            warnings.filterwarnings('ignore')

            # Add thirdparty paths
            hunyuan_omni_path = Path(__file__).parent.parent.parent / "thirdparty" / "Hunyuan3D-Omni"
            if str(hunyuan_omni_path) not in sys.path:
                sys.path.insert(0, str(hunyuan_omni_path))

            from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline

            print(f"Loading Hunyuan3D-Omni from {self.repo_id}...")
            self.pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(self.repo_id,
                                                                                 fast_decode=self.fast_decode)
            print("Hunyuan3D-Omni model loaded successfully")
            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading Hunyuan3D-Omni model: {e}")
            return False

    def normalize_point_cloud(self, points: np.ndarray, scale: float = 0.98) -> np.ndarray:
        """
        Normalize point cloud to fit within [-scale, scale] range
        
        Args:
            points: Point cloud array (N, 3)
            scale: Target scale range
            
        Returns:
            Normalized point cloud
        """
        # Calculate bounding box
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        center = (bbox_max + bbox_min) / 2
        max_dim = (bbox_max - bbox_min).max()

        # Apply centering and scaling
        normalized = (points - center) / max_dim * 2 * scale
        return normalized

    def generate_mesh(self,
                      image: Union[np.ndarray, str, Path],
                      point_cloud: Optional[np.ndarray] = None,
                      num_inference_steps: int = 50,
                      octree_resolution: int = 512,
                      mc_level: float = 0,
                      guidance_scale: float = 4.5,
                      seed: int = 1234) -> Dict[str, Any]:
        """
        Generate 3D mesh from image with optional point cloud conditioning
        
        Args:
            image: Input image (numpy array, file path, or PIL Image)
            point_cloud: Optional point cloud for conditioning (N, 3)
            num_inference_steps: Number of denoising steps
            octree_resolution: 3D resolution for octree representation
            mc_level: Marching cubes iso-level
            guidance_scale: Classifier-free guidance strength
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing:
            - mesh: Generated trimesh object
            - sampled_point: Sampled point cloud
        """
        if self.pipeline is None:
            if not self.load_model():
                raise RuntimeError("Failed to load Hunyuan3D-Omni model")

        import torch

        # Prepare image path
        if isinstance(image, np.ndarray):
            # Save to temporary file
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(image)
            temp_path = Path("/tmp/hunyuan_input.png")
            pil_image.save(temp_path)
            image_path = str(temp_path)
        elif isinstance(image, Path):
            image_path = str(image)
        else:
            image_path = image

        # Prepare point cloud if provided
        point_tensor = None
        if point_cloud is not None:
            # Normalize point cloud
            normalized_pc = self.normalize_point_cloud(point_cloud, scale=0.98)
            point_tensor = torch.FloatTensor(normalized_pc).unsqueeze(0)
            point_tensor = point_tensor.to(self.pipeline.device).to(self.pipeline.dtype)
            print(f"Using point cloud conditioning: {point_tensor.shape}")

        # Run inference
        try:
            if point_tensor is not None:
                result = self.pipeline(
                    image=image_path,
                    point=point_tensor,
                    num_inference_steps=num_inference_steps,
                    octree_resolution=octree_resolution,
                    mc_level=mc_level,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(self.device).manual_seed(seed),
                )
            else:
                # Generate without point cloud conditioning
                result = self.pipeline(
                    image=image_path,
                    num_inference_steps=num_inference_steps,
                    octree_resolution=octree_resolution,
                    mc_level=mc_level,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(self.device).manual_seed(seed),
                )

            mesh = result['shapes'][0][0]
            sampled_point = result['sampled_point'][0]

            return {'mesh': mesh, 'sampled_point': sampled_point}

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error generating mesh with Hunyuan3D-Omni: {e}")
            return None

    def postprocess_mesh(self, mesh) -> Any:
        """Apply post-processing to clean up mesh"""
        try:
            import sys
            hunyuan_omni_path = Path(__file__).parent.parent.parent / "thirdparty" / "Hunyuan3D-Omni"
            if str(hunyuan_omni_path) not in sys.path:
                sys.path.insert(0, str(hunyuan_omni_path))

            from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover

            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            return mesh
        except Exception as e:
            print(f"Warning: Post-processing failed: {e}")
            return mesh


class Hunyuan3DPaintClient:
    """Wrapper for Hunyuan3D-2.1 texture painting"""

    def __init__(self, max_num_view: int = 6, resolution: int = 1024):
        """
        Initialize Hunyuan3D Paint client
        
        Args:
            max_num_view: Maximum number of views for texture generation
            resolution: Resolution for texture generation
        """
        self.max_num_view = max_num_view
        self.resolution = resolution
        self.pipeline = None

    def load_model(self):
        """Load the Hunyuan3D Paint pipeline"""
        if self.pipeline is not None:
            return True

        try:
            import sys
            import warnings
            warnings.filterwarnings('ignore')

            # Add thirdparty paths
            hunyuan_paint_path = Path(__file__).parent.parent.parent / "thirdparty" / "Hunyuan3D-2.1"
            hunyuan_paint_dr_path = Path(__file__).parent.parent.parent / "thirdparty" / "Hunyuan3D-2.1" / "hy3dpaint"
            if str(hunyuan_paint_path) not in sys.path:
                sys.path.insert(0, str(hunyuan_paint_path))
                sys.path.insert(0, str(hunyuan_paint_dr_path))

            from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

            print("Loading Hunyuan3D Paint pipeline...")
            config = Hunyuan3DPaintConfig(max_num_view=self.max_num_view, resolution=self.resolution)
            self.pipeline = Hunyuan3DPaintPipeline(config=config)
            print("Hunyuan3D Paint model loaded successfully")
            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading Hunyuan3D Paint model: {e}")
            return False

    def generate_texture(self,
                         mesh_path: Union[str, Path],
                         image: Union[np.ndarray, str, Path],
                         output_path: Optional[Union[str, Path]] = None,
                         use_remesh: bool = True,
                         save_glb: bool = True) -> Optional[str]:
        """
        Generate texture for 3D mesh from reference image
        
        Args:
            mesh_path: Path to input mesh (GLB or OBJ)
            image: Reference image for texture (numpy array, file path, or PIL Image)
            output_path: Optional output path for textured mesh
            use_remesh: Whether to remesh the input mesh
            save_glb: Whether to save as GLB format
            
        Returns:
            Path to textured mesh file if successful, None otherwise
        """
        if self.pipeline is None:
            if not self.load_model():
                raise RuntimeError("Failed to load Hunyuan3D Paint model")

        # Prepare image
        if isinstance(image, np.ndarray):
            from PIL import Image as PILImage
            image = PILImage.fromarray(image)
        elif isinstance(image, (str, Path)):
            from PIL import Image as PILImage
            image = PILImage.open(image)

        # Prepare paths
        mesh_path = str(mesh_path)
        if output_path is None:
            mesh_dir = Path(mesh_path).parent
            output_path = mesh_dir / "textured_mesh.obj"
        output_path = str(output_path)

        try:
            result_path = self.pipeline(mesh_path=mesh_path,
                                        image_path=image,
                                        output_mesh_path=output_path,
                                        use_remesh=use_remesh,
                                        save_glb=save_glb)

            print(f"Texture generation complete: {result_path}")
            return result_path

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error generating texture with Hunyuan3D Paint: {e}")
            return None
