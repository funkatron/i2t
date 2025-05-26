import warnings
warnings.filterwarnings("ignore")

import argparse
import os
from .service import BlipCaptionService, JoyCaptionService


def main():
    parser = argparse.ArgumentParser(description="Image-to-Text Captioning (i2t)")
    parser.add_argument("image", nargs="?", help="Path to the image file")
    parser.add_argument("--model", choices=["blip", "joy"], default="joy", help="Which model to use: blip or joy (default: joy)")
    parser.add_argument("--show", action="store_true", help="Show the image before captioning")
    parser.add_argument("--precache", action="store_true", help="Download and cache the selected model, then exit")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format (default: text)")
    args = parser.parse_args()

    # Suppress TQDM and transformers logging if not text output
    if args.format != "text":
        os.environ["TRANSFORMERS_NO_TQDM"] = "1"
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()

    if args.precache:
        if args.model == "blip":
            BlipCaptionService.precache()
        else:
            JoyCaptionService.precache()
        return

    if not args.image:
        parser.error("the following arguments are required: image (unless using --precache)")

    if args.model == "blip":
        service = BlipCaptionService()
    else:
        service = JoyCaptionService()

    caption = service.caption_image_path(args.image, show=args.show)
    if args.format == "json":
        import json
        print(json.dumps({
            "image": args.image,
            "model": args.model,
            "caption": caption
        }, indent=2))
    else:
        print(f"\nGenerated caption: {caption}\n")

if __name__ == "__main__":
    main()