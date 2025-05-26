from setuptools import setup, find_packages

setup(
    name="i2t",
    version="0.1.0",
    description="Image-to-Text Captioning (BLIP, JoyCaption) with MPS/Apple Silicon support",
    author="funkatron",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "transformers",
        "Pillow",
    ],
    entry_points={
        "console_scripts": [
            "i2t = i2t.cli:main",
        ],
    },
    python_requires=">=3.11",
)