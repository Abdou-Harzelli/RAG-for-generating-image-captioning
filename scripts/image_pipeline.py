import os
from PIL import Image, ImageOps
from torchvision import transforms

class ImageTransformationPipeline:
    """
    Standardizes raw COCO images for Vision-Language Models (CLIP, SigLIP, etc.).
    Handles EXIF rotation, RGB conversion, and aspect-ratio-preserving center crops.
    """
    def __init__(self, target_size=224):
        # 224x224 is the standard input size for most Vision models
        self.target_size = target_size
        
        # Resize shorter edge to 224, then crop center 224x224 to preserve aspect ratio
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.target_size)
        ])

    def process_image(self, image_path):
        """Processes a single image and returns a clean PIL Image ready for AI."""
        try:
            img = Image.open(image_path)
            
            # 1. Correct EXIF orientation
            img = ImageOps.exif_transpose(img)
            
            # 2. Force RGB format
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # 3. Resize and Center Crop
            img_transformed = self.transform(img)
            
            return img_transformed
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

if __name__ == "__main__":
    print("Step 3_1 has been Completed: Image Pipeline is ready for you to use in batch processing!")