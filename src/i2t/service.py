from abc import ABC, abstractmethod
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModelForVision2Seq
from transformers import LlavaForConditionalGeneration
import torch
import sys
import os
from contextlib import contextmanager
import logging
import transformers

# Suppress TQDM progress bars and set HuggingFace transformers logging to error
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger("i2t")

@contextmanager
def suppress_output(enabled):
    if not enabled:
        yield
        return
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

class BaseCaptionService(ABC):
    def __init__(self, device=None, quiet=False):
        self.quiet = quiet
        if self.quiet:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)
        with suppress_output(self.quiet):
            self.device = device or self.get_device()
            logger.info(f"Using device: {self.device}")
            self.model = None
            self.processor = None
            self.load_model_and_processor()

    @staticmethod
    def get_device():
        """Select the best available device (MPS if available, else CPU)."""
        if torch.cuda.is_available():
            logger.info("Using CUDA GPU")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon GPU)")
            return torch.device("mps")
        else:
            logger.info("Using CPU")
            return torch.device("cpu")

    @staticmethod
    def load_image(image_path):
        """Load and convert an image to RGB."""
        image = Image.open(image_path).convert("RGB")
        return image

    @abstractmethod
    def load_model_and_processor(self):
        pass

    @abstractmethod
    def generate_caption(self, image, prompt=None):
        pass

    def caption_image_path(self, image_path, show=False, prompt=None):
        image = self.load_image(image_path)
        if show and not self.quiet:
            image.show()
        with suppress_output(self.quiet):
            return self.generate_caption(image, prompt=prompt)

class BlipCaptionService(BaseCaptionService):
    DEFAULT_PROMPT = "Describe this image in vivid, natural language, mentioning important details, actions, and the overall mood."
    @classmethod
    def precache(cls, quiet=False):
        if quiet:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)
        with suppress_output(quiet):
            logger.info("Caching BLIP model and processor...")
            BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            logger.info("Caching complete.")

    def load_model_and_processor(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model.to(self.device)

    def generate_caption(self, image, prompt=None):
        """Generate a caption for a single PIL image."""
        if prompt:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        else:
            inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=512)
        return self.processor.decode(out[0], skip_special_tokens=True)

class BlipLargeCaptionService(BaseCaptionService):
    DEFAULT_PROMPT = "Describe this image in vivid, natural language, mentioning important details, actions, and the overall mood."
    @classmethod
    def precache(cls, quiet=False):
        if quiet:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)
        with suppress_output(quiet):
            logger.info("Caching BLIP-Large model and processor...")
            BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            logger.info("Caching complete.")

    def load_model_and_processor(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model.to(self.device)

    def generate_caption(self, image, prompt=None):
        """Generate a caption for a single PIL image."""
        if prompt:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        else:
            inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=512)
        return self.processor.decode(out[0], skip_special_tokens=True)