# i2t: Image-to-Text Captioning with MPS/Apple Silicon support

## Description

i2t is a simple CLI tool and Python module for generating descriptive captions for images using state-of-the-art models (BLIP and JoyCaption/LLaVA) locally on MacOS running on ARM64/Apple Silicon chips.


## Features
- **BLIP and JoyCaption (LLaVA) model support**
- **Modern src/ package structure**
- **Command-line interface (`i2t`)**
- **Quiet mode for scripting/automation**
- **Output formats: plain text, JSON**
- **Pre-caching for offline/production use**
- **Apple Silicon (MPS) and CUDA support**
- **Gradio demo app (`src/i2t/gradio_app.py`)**

## Installation

1. Clone the repo and install in editable mode:
   ```sh
   git clone <your-repo-url>
   cd blip-mps
   pip install -e .
   ```
2. (Recommended) Set up a virtual environment:
   ```sh
   python -m venv venv-blip
   source venv-blip/bin/activate
   ```

## Usage

### CLI

#### Help:
```sh
i2t --help
```

#### Generate a caption:

```sh
i2t path/to/image.jpg
```

#### Generate a caption (BLIP or JoyCaption):
```sh
i2t path/to/image.jpg --model blip
# or
i2t path/to/image.jpg --model joy
```

#### Output as JSON (quiet mode):
```sh
i2t path/to/image.jpg --model joy --format json
```

#### Pre-cache models for offline use:
```sh
i2t --model joy --precache
```

#### Show the image before captioning:
```sh
i2t path/to/image.jpg --show
```

### Python API

```python
from i2t.service import BlipCaptionService, JoyCaptionService

service = BlipCaptionService()  # or JoyCaptionService()
caption = service.caption_image_path('path/to/image.jpg')
print(caption)
```

### Gradio Demo App

Launch the Gradio demo app with:

```sh
python -m i2t.gradio_app
```

This will open a web UI for uploading an image and selecting a model (BLIP or JoyCaption) to generate a caption.

## Model Options
- `blip`: [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
- `joy`: [fancyfeast/llama-joycaption-beta-one-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava)

## Requirements
- Python 3.11+
- torch >= 2.3.0 (for MPS/Apple Silicon support)
- transformers >= 4.45
- Pillow

## Notes
- Models are cached in the HuggingFace cache directory (`~/.cache/huggingface/hub` by default).
- Use `--precache` to download models in advance for offline/production use.
- Quiet mode (`--format json`) suppresses all extra output except the requested format.

## License
MIT

