from abc import ABC, abstractmethod
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModelForVision2Seq
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
    def generate_caption(self, image):
        pass

    def caption_image_path(self, image_path, show=False):
        image = self.load_image(image_path)
        if show and not self.quiet:
            image.show()
        with suppress_output(self.quiet):
            return self.generate_caption(image)

class BlipCaptionService(BaseCaptionService):
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

    def generate_caption(self, image):
        """Generate a caption for a single PIL image."""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)

class JoyCaptionService(BaseCaptionService):
    MODEL_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"
    PROMPT = "Write a long descriptive caption for this image in a formal tone."

    @classmethod
    def precache(cls, quiet=False):
        if quiet:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)
        with suppress_output(quiet):
            logger.info("Caching JoyCaption model and processor...")
            AutoProcessor.from_pretrained("fancyfeast/llama-joycaption-beta-one-hf-llava", use_fast=True)
            AutoModelForVision2Seq.from_pretrained("fancyfeast/llama-joycaption-beta-one-hf-llava")
            logger.info("Caching complete.")

    def load_model_and_processor(self):
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.MODEL_ID,
                local_files_only=False,
                revision="main",
                use_fast=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float32,
                device_map=None,
                local_files_only=False,
                revision="main",
            ).to(self.device)
            self.model.eval()

            # Patch out "lanczos" resample setting if present
            if hasattr(self.processor, 'image_processor') and hasattr(self.processor.image_processor, 'resample'):
                if self.processor.image_processor.resample == getattr(Image, "Resampling", Image).LANCZOS:
                    self.processor.image_processor.resample = getattr(Image, "Resampling", Image).BICUBIC
                if hasattr(self.processor.image_processor, 'image_resample_mode'):
                    if self.processor.image_processor.image_resample_mode == "lanczos":
                        self.processor.image_processor.image_resample_mode = "bicubic"
        except Exception as e:
            logger.error(f"Error loading JoyCaption model: {e}")
            raise

    def generate_caption(self, image, prompt=None):
        prompt = prompt or self.PROMPT
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            convo = [
                {"role": "system", "content": "You are a helpful image captioner."},
                {"role": "user", "content": f"{prompt}\n<image>"},
            ]
            convo_string = self.processor.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=convo_string,
                images=image,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    use_cache=True,
                    temperature=0.6,
                    top_p=0.9,
                )[0]
            generate_ids = generate_ids[inputs['input_ids'].shape[1]:]
            caption = self.processor.tokenizer.decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return caption.strip()
        except Exception as e:
            logger.error(f"Error during JoyCaption caption generation: {e}")
            return "Could not generate caption due to an error."