"""
Object Detection and Segmentation Module

This module handles object detection and segmentation using locally deployed RAM-Grounded-SAM.
New pipeline: Detection (RAM + Grounding DINO) -> Filtering -> Segmentation (SAM)
"""
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import cv2
from pathlib import Path
import torch
import torchvision
from PIL import Image

# Add Grounded-SAM to path
GROUNDED_SAM_PATH = Path(__file__).parent.parent.parent / "thirdparty" / "Grounded-Segment-Anything"
sys.path.insert(0, str(GROUNDED_SAM_PATH))
sys.path.insert(0, str(GROUNDED_SAM_PATH / "GroundingDINO"))

# Grounding DINO imports
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything imports
from segment_anything import build_sam, build_sam_hq, SamPredictor

# RAM imports
from ram.models import ram
from ram import inference_ram
import torchvision.transforms as TS

from ..core.common import DetectedObject, BoundingBox
from ..utils.image_utils import crop_image_with_bbox, save_image
from ..config.settings import config


class DetectionSegmentationModule:
    """Module for object detection and segmentation using local RAM-Grounded-SAM"""
    
    def __init__(self):
        """Initialize the detection and segmentation module"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model instances (lazy loaded)
        self.ram_model = None
        self.grounding_dino_model = None
        self.sam_predictor = None
        
        # Transforms for RAM
        self.ram_transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(),
            TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_ram_model(self):
        """Lazy load RAM model"""
        if self.ram_model is None:
            print("Loading RAM model...")
            self.ram_model = ram(
                pretrained=config.models.ram_checkpoint,
                image_size=384,
                vit='swin_l'
            )
            self.ram_model.eval()
            self.ram_model = self.ram_model.to(self.device)
            print("RAM model loaded successfully")
        return self.ram_model
    
    def _load_grounding_dino_model(self):
        """Lazy load Grounding DINO model"""
        if self.grounding_dino_model is None:
            print("Loading Grounding DINO model...")
            args = SLConfig.fromfile(config.models.grounding_dino_config)
            args.device = self.device
            model = build_model(args)
            checkpoint = torch.load(config.models.grounding_dino_checkpoint, map_location="cpu")
            load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            print(f"Grounding DINO load result: {load_res}")
            model.eval()
            self.grounding_dino_model = model
            print("Grounding DINO model loaded successfully")
        return self.grounding_dino_model
    
    def _load_sam_predictor(self, use_sam_hq: bool = False):
        """Lazy load SAM predictor"""
        if self.sam_predictor is None:
            print(f"Loading SAM predictor (HQ={use_sam_hq})...")
            if use_sam_hq and config.models.sam_hq_checkpoint:
                sam = build_sam_hq(checkpoint=config.models.sam_hq_checkpoint)
            else:
                sam = build_sam(checkpoint=config.models.sam_checkpoint)
            sam.to(self.device)
            self.sam_predictor = SamPredictor(sam)
            print("SAM predictor loaded successfully")
        return self.sam_predictor
    
    def _load_image_for_grounding_dino(self, image_pil: Image.Image):
        """Load and transform image for Grounding DINO"""
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(image_pil, None)
        return image_transformed
    
    def _get_grounding_output(self, model, image, caption, box_threshold, text_threshold):
        """Run Grounding DINO inference"""
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        
        model = model.to(self.device)
        image = image.to(self.device)
        
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        
        # Filter output
        filt_mask = logits.max(dim=1)[0] > box_threshold
        logits_filt = logits[filt_mask]
        boxes_filt = boxes[filt_mask]
        
        # Get phrases
        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)
        
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            scores.append(logit.max().item())
        
        return boxes_filt, torch.Tensor(scores), pred_phrases
    
    def detect_objects(self, image: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects using RAM + Grounding DINO (bboxes only, no segmentation)
        
        Args:
            image: Input RGB image as numpy array
            
        Returns:
            List of detected objects with bounding boxes and descriptions (no masks yet)
        """
        print("Running RAM + Grounding DINO for object detection...")
        
        # Convert to PIL
        image_pil = Image.fromarray(image)
        
        # Step 1: Run RAM to get tags
        ram_model = self._load_ram_model()
        raw_image = image_pil.resize((384, 384))
        raw_image_tensor = self.ram_transform(raw_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            res = inference_ram(raw_image_tensor, ram_model)
        
        # Convert tags (replace " |" with ",")
        tags = res[0].replace(' |', ',')
        print(f"RAM detected tags: {tags}")
        
        # Step 2: Run Grounding DINO to get bounding boxes
        grounding_model = self._load_grounding_dino_model()
        image_transformed = self._load_image_for_grounding_dino(image_pil)
        
        boxes_filt, scores, pred_phrases = self._get_grounding_output(
            grounding_model, 
            image_transformed, 
            tags,
            config.models.box_threshold,
            config.models.text_threshold
        )
        
        # Convert boxes to image coordinates
        H, W = image.shape[:2]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        
        boxes_filt = boxes_filt.cpu()
        
        # Apply NMS to handle overlapped boxes
        print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_filt, scores, config.models.iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        scores = scores[nms_idx]
        print(f"After NMS: {boxes_filt.shape[0]} boxes")
        
        # Create DetectedObject instances (without masks)
        detected_objects = []
        for i, (box, phrase, score) in enumerate(zip(boxes_filt, pred_phrases, scores)):
            # Extract label and confidence from phrase
            if '(' in phrase:
                label = phrase.split('(')[0]
                logit_str = phrase.split('(')[1].rstrip(')')
                confidence = float(logit_str)
            else:
                label = phrase
                confidence = float(score)
            
            bbox = BoundingBox(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
                confidence=confidence
            )
            
            detected_obj = DetectedObject(
                id=i + 1,  # 1-indexed
                bbox=bbox,
                description=label,
                confidence=confidence
            )
            
            # Crop the object from the image
            crop_coords = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
            detected_obj.cropped_image = crop_image_with_bbox(image, crop_coords)

            detected_objects.append(detected_obj)

        print(f"Detected {len(detected_objects)} objects")
        return detected_objects
    
    def segment_objects(self, image: np.ndarray, detected_objects: List[DetectedObject],
                       use_sam_hq: bool = False) -> List[DetectedObject]:
        """
        Segment detected objects using SAM
        
        Args:
            image: Input RGB image as numpy array
            detected_objects: List of detected objects with bounding boxes
            use_sam_hq: Use SAM-HQ instead of regular SAM
            
        Returns:
            List of detected objects with segmentation masks added
        """
        if not detected_objects:
            print("No objects to segment")
            return detected_objects
        
        print(f"Running SAM segmentation on {len(detected_objects)} objects...")
        
        # Load SAM predictor
        predictor = self._load_sam_predictor(use_sam_hq)
        predictor.set_image(image)
        
        # Prepare bounding boxes for SAM
        boxes_tensor = torch.tensor([
            [obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2]
            for obj in detected_objects
        ], dtype=torch.float32)
        
        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes_tensor, image.shape[:2]
        ).to(self.device)
        
        # Run SAM inference
        with torch.no_grad():
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
        
        # Add masks to detected objects
        for i, obj in enumerate(detected_objects):
            mask = masks[i].cpu().numpy()[0]  # Shape: (H, W)
            
            # Convert to binary mask (0 or 255)
            binary_mask = (mask * 255).astype(np.uint8)
            obj.mask = binary_mask
            
            # Create occlusion mask (objects that are not background or current object)
            # For now, we'll create it based on other objects' masks
            occ_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
            
            # Mark all other object regions as occluding
            for j, other_obj in enumerate(detected_objects):
                if i != j and other_obj.mask is not None:
                    other_mask = (other_obj.mask > 0).astype(np.uint8)
                    occ_mask[other_mask > 0] = 0
            
            # Mark background as non-occluding
            # (For simplicity, we'll keep this basic for now)
            obj.occ_mask = occ_mask
            
            # Crop the mask
            crop_coords = (obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2)
            obj.cropped_mask = crop_image_with_bbox(obj.mask, crop_coords)
            obj.cropped_occ_mask = crop_image_with_bbox(obj.occ_mask, crop_coords)
        
        print(f"Segmentation complete for {len(detected_objects)} objects")
        return detected_objects
    
    def run(self, image: np.ndarray, output_dir: Optional[Path] = None,
            use_sam_hq: bool = False) -> List[DetectedObject]:
        """
        Run the detection pipeline (detection only, segmentation done separately after filtering)
        
        Args:
            image: Input RGB image
            output_dir: Optional directory to save intermediate results
            use_sam_hq: Use SAM-HQ for segmentation (unused in detection-only mode)
            
        Returns:
            List of detected objects with bounding boxes (no masks)
        """
        print("Starting RAM-Grounded-DINO detection pipeline...")
        
        # Run detection only
        detected_objects = self.detect_objects(image)
        
        if not detected_objects:
            print("No objects detected")
            return []
        
        # Save intermediate results if output directory is provided
        if output_dir:
            self._save_detection_results(image, detected_objects, output_dir)
        
        print(f"Detection complete. Found {len(detected_objects)} objects.")
        return detected_objects
    
    def run_segmentation(self, image: np.ndarray, detected_objects: List[DetectedObject],
                        output_dir: Optional[Path] = None, 
                        use_sam_hq: bool = False) -> List[DetectedObject]:
        """
        Run segmentation on filtered detected objects
        
        Args:
            image: Input RGB image
            detected_objects: List of detected objects (already filtered)
            output_dir: Optional directory to save intermediate results
            use_sam_hq: Use SAM-HQ instead of regular SAM
            
        Returns:
            List of detected objects with segmentation masks
        """
        print("Starting SAM segmentation pipeline...")
        
        # Run segmentation
        segmented_objects = self.segment_objects(image, detected_objects, use_sam_hq)
        
        # Save intermediate results if output directory is provided
        if output_dir:
            self._save_segmentation_results(image, segmented_objects, output_dir)
        
        print(f"Segmentation complete for {len(segmented_objects)} objects.")
        return segmented_objects
    
    def _save_detection_results(self, image: np.ndarray, detected_objects: List[DetectedObject], 
                     output_dir: Path) -> None:
        """Save detection results"""
        detection_dir = output_dir / "detection"
        detection_dir.mkdir(exist_ok=True, parents=True)
        
        # Save annotated image with bounding boxes
        annotated_image = self._draw_bounding_boxes(image.copy(), detected_objects)
        save_image(annotated_image, detection_dir / "detected_bboxes.png")
        
        # Save individual object crops
        for obj in detected_objects:
            obj_dir = detection_dir / f"object_{obj.id}"
            obj_dir.mkdir(exist_ok=True)
            
            # Save cropped image
            if obj.cropped_image is not None:
                save_image(obj.cropped_image, obj_dir / "cropped.png")
            
            # Save object info
            obj_info = {
                "id": obj.id,
                "description": obj.description,
                "confidence": obj.confidence,
                "bbox": {
                    "x1": obj.bbox.x1, "y1": obj.bbox.y1,
                    "x2": obj.bbox.x2, "y2": obj.bbox.y2
                }
            }
            
            with open(obj_dir / "info.json", "w") as f:
                json.dump(obj_info, f, indent=2)
    
    def _save_segmentation_results(self, image: np.ndarray, detected_objects: List[DetectedObject], 
                                  output_dir: Path) -> None:
        """Save segmentation results"""
        segmentation_dir = output_dir / "segmentation"
        segmentation_dir.mkdir(exist_ok=True, parents=True)
        
        # Save annotated image with masks
        annotated_image = self._draw_masks_on_image(image.copy(), detected_objects)
        save_image(annotated_image, segmentation_dir / "segmented_objects.png")
        
        # Save individual object masks
        for obj in detected_objects:
            obj_dir = segmentation_dir / f"object_{obj.id}"
            obj_dir.mkdir(exist_ok=True)
            
            # Save masks
            if obj.mask is not None:
                cv2.imwrite(str(obj_dir / "mask.png"), obj.mask)
            if obj.cropped_mask is not None:
                cv2.imwrite(str(obj_dir / "cropped_mask.png"), obj.cropped_mask)
            if obj.occ_mask is not None:
                cv2.imwrite(str(obj_dir / "occ_mask.png"), obj.occ_mask)
            if obj.cropped_occ_mask is not None:
                cv2.imwrite(str(obj_dir / "cropped_occ_mask.png"), obj.cropped_occ_mask)
    
    def _draw_bounding_boxes(self, image: np.ndarray, 
                           detected_objects: List[DetectedObject]) -> np.ndarray:
        """Draw bounding boxes and labels on image"""
        for obj in detected_objects:
            x1, y1, x2, y2 = int(obj.bbox.x1), int(obj.bbox.y1), int(obj.bbox.x2), int(obj.bbox.y2)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{obj.id}: {obj.description}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 0), 2)
        
        return image
    
    def _draw_masks_on_image(self, image: np.ndarray,
                            detected_objects: List[DetectedObject]) -> np.ndarray:
        """Draw masks on image with random colors"""
        overlay = image.copy()
        
        for obj in detected_objects:
            if obj.mask is not None:
                # Generate random color
                color = np.random.randint(0, 255, 3).tolist()
                
                # Apply colored mask
                mask_bool = obj.mask > 0
                overlay[mask_bool] = (overlay[mask_bool] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
                
                # Draw bbox and label
                x1, y1, x2, y2 = int(obj.bbox.x1), int(obj.bbox.y1), int(obj.bbox.x2), int(obj.bbox.y2)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                
                label = f"{obj.id}: {obj.description}"
                cv2.putText(overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 2)
        
        return overlay
