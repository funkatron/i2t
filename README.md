# i2t: Image-to-Text Captioning with MPS/Apple Silicon support

## Description

`i2t` is a simple CLI tool and Python module for generating descriptive captions for images using models like BLIP and JoyCaption/LLaVA locally on **MacOS** running on ARM64/Apple Silicon chips.


## Features
- **Image captioning**: Generate descriptive, accurate captions for images using advanced models running locally **on your Mac**.
  - While this tool is primarily designed for MacOS, it will check for CUDA support, so may also run on Linux with CUDA support or Windows with WSL2 and CUDA.
- **BLIP and JoyCaption (LLaVA) model support**
  - Automatically installed on first run from HuggingFace.
- **A Simple Command-line interface (`i2t`)**
- **Quiet mode for scripting/automation**
- **Output formats: plain text, JSON**
- **Pre-caching of models for offline/production use**

## Requirements
- MacOS 14.6+ with ARM64/Apple Silicon
  - As of this writing, testing has been limited to an M2 Ultra-based Mac Studio running MacOS 14.06 🤷‍♀️.
- Python 3.11+

## Installation

1. Clone the repo:
   ```sh
   git clone https://github.com/funkatron/i2t.git
   cd i2t
   ```
2. Set up a virtual environment:
   ```sh
   python3.11 -m venv venv
   source venv/bin/activate
   pip install uv
   ```
3. Install dependencies:
   ```sh
   # note that we use uv for speed and better dependency management, but
   # you can also use `pip` directly if you prefer
   uv pip install -r pyproject.toml
   ```
4. Install the package in editable mode:
   ```sh
   uv pip install -e .
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

## Model Options
- `joy` (default): [fancyfeast/llama-joycaption-beta-one-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava)
- `blip`: [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)


## Notes
- Models are cached in the HuggingFace cache directory (`~/.cache/huggingface/hub` by default).
- Use `--precache` to download models in advance for offline/production use.
- Quiet mode (`--format json`) suppresses all extra output except the requested format.
- This project uses NumPy 1.26.4 for compatibility with PyTorch and other dependencies.
- BLIP is the default and most reliable model. JoyCaption is experimental and may not work on all setups.

## Troubleshooting

If you encounter NumPy compatibility errors:
1. Ensure you have NumPy 1.x installed (this project uses 1.26.4)
2. If upgrading from NumPy 2.x, you may need to recreate your virtual environment:
   ```sh
   deactivate
   rm -rf venv
   python3.11 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install numpy==1.26.4
   pip install -r pyproject.toml
   pip install -e .
   ```

## License
MIT

