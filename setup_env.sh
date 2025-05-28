#!/bin/bash
# setup_env.sh - Creates a clean environment for i2t

SCRIPT_DIR=$(dirname "$(realpath "$0")")

# deactivate any existing virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating existing virtual environment..."
    deactivate
fi

# Stop on any errors
set -e

echo "Setting up i2t environment..."

# Check if Python 3.11+ is installed
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    echo "Error: Python 3.11 or higher is required"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

# install uv via brew if not already installed (and brew is available)
if ! command -v uv &> /dev/null && command -v brew &> /dev/null; then
    echo "Installing uv via Homebrew..."
    brew install uv
elif ! command -v uv &> /dev/null; then
    echo "Installing uv via pip for this user..."
    pip3 install --user uv
else
    echo "uv is already installed."
fi

# prompt user for confirmation to clear existing virtual environment. If not, keeo the existing one
# and continue with the setup
VENV_DIR="$SCRIPT_DIR/.venv"
read -p "Do you want to clear the existing virtual environment? (y/n): " clear_venv_confirm

if [[ "$clear_venv_confirm" = "y" || "$clear_venv_confirm" = "Y" ]]; then
    echo "Clearing existing virtual environment at $VENV_DIR..."
    if [ -d "$VENV_DIR" ]; then
        rm -rf "$VENV_DIR"
        echo "Existing virtual environment cleared."
    else
        echo "No existing virtual environment found."
    fi

    # Create a new virtual environment
    echo "Creating a new virtual environment at $VENV_DIR"
    uv venv "$VENV_DIR" --prompt "i2t-venv"
else
    echo "Keeping existing virtual environment. Continuing with setup..."
fi

# Always activate the venv if not already active
if [ -z "$VIRTUAL_ENV" ]; then
  if [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
  else
    echo "Virtual environment not found! Please run python3 -m venv .venv first."
    exit 1
  fi
fi

# Install NumPy 1.26.4 first (critical for compatibility)
echo "Installing NumPy 1.26.4..."
uv pip install numpy==1.26.4

# Install other dependencies
echo "Installing dependencies in pyproject.toml..."
uv pip install -r pyproject.toml

# Install the package in development mode
echo "Installing i2t package in development mode..."
uv pip install -e .

# Check if the installation was successful
if uv pip show i2t > /dev/null 2>&1; then
    echo "i2t package installed successfully."
else
    echo "Error: i2t package installation failed."
    exit 1
fi

echo "Try the BLIP model first (more reliable):"
echo "  i2t ./test.jpg --model blip"
