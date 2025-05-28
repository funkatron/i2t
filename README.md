# i2t: Image-to-Text Captioning with MPS/Apple Silicon support

## Description

`i2t` is a simple CLI tool and Python module for generating descriptive captions for images using BLIP models locally on **MacOS** running on ARM64/Apple Silicon chips.


## Features
- **Image captioning**: Generate descriptive, accurate captions for images using advanced BLIP models running locally **on your Mac**.
  - While this tool is primarily designed for MacOS, it will check for CUDA support, so may also run on Linux with CUDA support or Windows with WSL2 and CUDA.
- **BLIP and BLIP-Large model support**
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
2. Run the handy dandy setup script:
   ```sh
   ./setup_env.sh
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

#### Generate a caption (BLIP, BLIP-Large):
```sh
i2t path/to/image.jpg --model blip
# or
i2t path/to/image.jpg --model blip-large
```

#### Generate a caption with a custom prompt prefix (BLIP/BLIP-Large):
```sh
i2t path/to/image.jpg --model blip --prompt-prefix "A beautiful photograph of"
```

#### Output as JSON (quiet mode):
```sh
i2t path/to/image.jpg --model blip --format json
```

#### Pre-cache models for offline use:
```sh
i2t --model blip --precache
# or
i2t --model blip-large --precache
```

#### Show the image before captioning:
```sh
i2t path/to/image.jpg --show
```

### Python API

```python
from i2t.service import BlipCaptionService, BlipLargeCaptionService

service = BlipCaptionService()  # or BlipLargeCaptionService()
caption = service.caption_image_path('path/to/image.jpg')
print(caption)
```

## Model Options
- `blip` (default): [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
- `blip-large`: [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large) (more accurate, uses more RAM/VRAM)


## Notes
- Models are cached in the HuggingFace cache directory (`~/.cache/huggingface/hub` by default).
- Use `--precache` to download models in advance for offline/production use.
- Quiet mode (`--format json`) suppresses all extra output except the requested format.
- This project uses NumPy 1.26.4 for compatibility with PyTorch and other dependencies.
- BLIP is the default and most reliable model.
- **Only BLIP and BLIP-Large models are supported in this version.**


## License
MIT

