from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


def get_device():
    """Select the best available device (MPS if available, else CPU)."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model_and_processor(device):
    """Load BLIP processor and model, move model to device."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)
    return processor, model


def load_image(image_path):
    """Load and convert an image to RGB."""
    image = Image.open(image_path).convert("RGB")
    return image


def generate_caption(image, processor, model, device):
    """Generate a caption for a single image."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def main():
    device = get_device()
    print(f"Using device: {device}")
    processor, model = load_model_and_processor(device)
    image_path = "test.jpg"  # Placeholder for future CLI argument
    image = load_image(image_path)
    image.show()  # Optional: visually confirm the image
    caption = generate_caption(image, processor, model, device)
    print(f"\nGenerated caption: {caption}\n")


if __name__ == "__main__":
    main()