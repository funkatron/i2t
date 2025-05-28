import warnings
warnings.filterwarnings("ignore")

import argparse
import os
from .service import BlipCaptionService, BlipLargeCaptionService
from glob import glob

SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

def main():
    parser = argparse.ArgumentParser(description="Image-to-Text Captioning (i2t)")
    parser.add_argument("image", nargs="?", help="Path to the image file")
    parser.add_argument("--model", choices=["blip", "blip-large"], default="blip", help="Which model to use: blip or blip-large (default: blip)")
    parser.add_argument("--show", action="store_true", help="Show the image before captioning")
    parser.add_argument("--precache", action="store_true", help="Download and cache the selected model, then exit")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format (default: text)")
    parser.add_argument("--batch-dir", help="Directory containing images to process in batch mode")
    args = parser.parse_args()

    # Suppress TQDM and transformers logging if not text output
    quiet = args.format != "text"
    if quiet:
        os.environ["TRANSFORMERS_NO_TQDM"] = "1"
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()

    if args.precache:
        if args.model == "blip":
            BlipCaptionService.precache(quiet=quiet)
        elif args.model == "blip-large":
            BlipLargeCaptionService.precache(quiet=quiet)
        return

    if args.batch_dir:
        # Batch mode: process all images in the directory
        image_paths = []
        for ext in SUPPORTED_EXTENSIONS:
            image_paths.extend(glob(os.path.join(args.batch_dir, f"*{ext}")))
        image_paths.sort()
        if not image_paths:
            print(f"No supported images found in directory: {args.batch_dir}")
            return
        try:
            if args.model == "blip":
                service = BlipCaptionService(quiet=quiet)
            elif args.model == "blip-large":
                service = BlipLargeCaptionService(quiet=quiet)
        except Exception as e:
            print(f"Failed to initialize image captioning service: {e}")
            return 1
        results = []
        for img_path in image_paths:
            caption = service.caption_image_path(img_path, show=args.show)
            results.append({"image": img_path, "model": args.model, "caption": caption})
        if args.format == "json":
            import json
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                print(f"{r['image']}: {r['caption']}")
        return

    if not args.image:
        parser.error("the following arguments are required: image (unless using --precache or --batch-dir)")

    try:
        if args.model == "blip":
            service = BlipCaptionService(quiet=quiet)
        elif args.model == "blip-large":
            service = BlipLargeCaptionService(quiet=quiet)
    except Exception as e:
        print(f"Failed to initialize image captioning service: {e}")
        return 1

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