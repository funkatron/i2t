from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class BlipCaptionService:
    def __init__(self, device=None):
        self.device = device or self.get_device()
        print(f"Using device: {self.device}")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model.to(self.device)

    @staticmethod
    def get_device():
        """Select the best available device (MPS if available, else CPU)."""
        return "mps" if torch.backends.mps.is_available() else "cpu"

    @staticmethod
    def load_image(image_path):
        """Load and convert an image to RGB."""
        image = Image.open(image_path).convert("RGB")
        return image

    def generate_caption(self, image):
        """Generate a caption for a single PIL image."""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

    def caption_image_path(self, image_path, show=False):
        image = self.load_image(image_path)
        if show:
            image.show()
        return self.generate_caption(image)

def main():
    service = BlipCaptionService()
    image_path = "test.jpg"  # Placeholder for future CLI argument
    caption = service.caption_image_path(image_path, show=True)
    print(f"\nGenerated caption: {caption}\n")

if __name__ == "__main__":
    main()