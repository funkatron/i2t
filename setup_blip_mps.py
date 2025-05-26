import os
import sys
import platform
import subprocess
import shutil
import urllib.request
import argparse
from contextlib import contextmanager

# Spinner and progress bar dependencies
try:
    from yaspin import yaspin
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yaspin"])
    from yaspin import yaspin
try:
    from colorama import Fore, Style, init as colorama_init
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "colorama"])
    from colorama import Fore, Style, init as colorama_init
try:
    from tqdm import tqdm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

colorama_init(autoreset=True)

VENV_DIR = "venv-blip"
REQUIREMENTS = [
    "torch==2.1.2",
    "torchvision==0.16.2",
    "torchaudio==2.1.2",
    "transformers==4.40.2",
    "pillow",
    "colorama",
    "yaspin",
    "tqdm"
]
TEST_IMAGE_URL = "https://images.pexels.com/photos/248797/pexels-photo-248797.jpeg?auto=compress&w=512"
TEST_IMAGE_FILE = "test.jpg"

def cprint(msg, color=None, end="\n"):
    if color:
        print(color + msg + Style.RESET_ALL, end=end)
    else:
        print(msg, end=end)

@contextmanager
def spinning(text):
    with yaspin(text=text, color="cyan") as spinner:
        try:
            yield spinner
            spinner.ok("✔")
        except Exception as e:
            spinner.fail("✗")
            raise e

def print_banner(title):
    line = "═" * (len(title) + 2)
    cprint(f"\n╔{line}╗", Fore.CYAN)
    cprint(f"║ {title} ║", Fore.CYAN)
    cprint(f"╚{line}╝", Fore.CYAN)

def check_platform(force):
    os_name = platform.system()
    arch = platform.machine()
    if os_name != "Darwin":
        cprint("❌ This script is intended for macOS. Exiting.", Fore.RED)
        if not force:
            sys.exit(1)
    if arch != "arm64":
        cprint("⚠️  Warning: This script is optimized for Apple Silicon (arm64).", Fore.YELLOW)
        if not force:
            ans = input("Continue anyway? [y/N] ")
            if not ans.lower().startswith("y"):
                print("Exiting.")
                sys.exit(1)

def find_python():
    ver = sys.version_info
    if not (ver.major == 3 and ver.minor >= 10):
        cprint(f"❌ Python 3.10 or newer is required. You are running: {sys.version}", Fore.RED)
        sys.exit(1)

def create_or_overwrite_venv(venv_dir, auto_overwrite=False):
    if os.path.isdir(venv_dir):
        if not auto_overwrite:
            cprint(f"⚠️  Virtual environment '{venv_dir}' already exists.", Fore.YELLOW)
            resp = input("Delete and recreate it? [y/N] ")
            if not resp.lower().startswith("y"):
                cprint("Using existing venv. (Dependencies will still be installed/upgraded.)", Fore.GREEN)
                return
        shutil.rmtree(venv_dir)
        cprint("🗑️  Removed existing virtual environment.", Fore.YELLOW)
    with spinning("Creating virtual environment..."):
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
    cprint("✅ Virtual environment created.", Fore.GREEN)

def run_in_venv(venv_dir, cmd_args, msg=""):
    """Run a command inside the virtualenv with spinner."""
    if platform.system() == "Windows":
        python_bin = os.path.join(venv_dir, "Scripts", "python.exe")
        pip_bin = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        python_bin = os.path.join(venv_dir, "bin", "python")
        pip_bin = os.path.join(venv_dir, "bin", "pip")
    if cmd_args[0] == "python":
        cmd_args[0] = python_bin
    elif cmd_args[0] == "pip":
        cmd_args[0] = pip_bin
    with spinning(msg or "Running command..."):
        subprocess.check_call(cmd_args)

def install_requirements(venv_dir, requirements):
    cprint("📦 Installing dependencies...", Fore.CYAN)
    run_in_venv(venv_dir, ["pip", "install", "--upgrade", "pip"], "Upgrading pip...")
    run_in_venv(venv_dir, ["pip", "install"] + requirements, "Installing Python requirements...")
    cprint("✅ Dependencies installed.", Fore.GREEN)

def download_with_progress(url, filename):
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        with open(filename, "wb") as f, tqdm(
            total=total, unit='B', unit_scale=True, desc=filename, ncols=70
        ) as pbar:
            while True:
                chunk = response.read(1024 * 8)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))

def download_test_image(image_file, url, auto_download=False):
    if os.path.exists(image_file):
        return
    cprint(f"\nNo '{image_file}' image found in the current directory.", Fore.YELLOW)
    if auto_download:
        dl = True
    else:
        resp = input("Download a sample image for testing? [Y/n] ")
        dl = not resp.lower().startswith("n")
    if dl:
        cprint(f"Downloading {url} ...", Fore.CYAN)
        download_with_progress(url, image_file)
        cprint(f"🖼️  Sample image downloaded as {image_file}", Fore.GREEN)

def generate_requirements_txt(venv_dir, auto_generate=False):
    if auto_generate:
        gen = True
    else:
        resp = input("Generate requirements.txt for reproducibility? [Y/n] ")
        gen = not resp.lower().startswith("n")
    if gen:
        if platform.system() == "Windows":
            pip_bin = os.path.join(venv_dir, "Scripts", "pip.exe")
        else:
            pip_bin = os.path.join(venv_dir, "bin", "pip")
        with spinning("Generating requirements.txt..."):
            with open("requirements.txt", "w") as f:
                subprocess.check_call([pip_bin, "freeze"], stdout=f)
        cprint("📄 requirements.txt generated.", Fore.GREEN)

def print_footer(venv_dir):
    cprint("\n🔹 BLIP+MPS setup complete!\n", Fore.CYAN)
    if platform.system() == "Windows":
        activate_cmd = f"{venv_dir}\\Scripts\\activate"
    else:
        activate_cmd = f"source {venv_dir}/bin/activate"
    cprint(f"➡️  To activate your environment in future shells:\n    {activate_cmd}", Fore.YELLOW)
    cprint("➡️  To run your caption script (blip_caption.py):\n    python blip_caption.py\n", Fore.YELLOW)
    cprint(f"• If you want to remove the environment later: rm -rf {venv_dir}", Fore.BLUE)
    cprint("• For reproducible installs: pip install -r requirements.txt\n", Fore.BLUE)
    cprint("🚩 Note: After this script finishes, your shell will NOT be inside the venv.\n"
           "    To use the venv, activate it as above.\n", Fore.MAGENTA)

def main():
    parser = argparse.ArgumentParser(
        description="Robust BLIP+MPS setup script (Python version, fancier output)"
    )
    parser.add_argument("--force", action="store_true", help="Force run on non-AppleSilicon or non-macOS")
    parser.add_argument("--overwrite", action="store_true", help="Automatically overwrite any existing venv")
    parser.add_argument("--download-image", action="store_true", help="Automatically download test image")
    parser.add_argument("--requirements", action="store_true", help="Automatically generate requirements.txt")
    args = parser.parse_args()

    print_banner("BLIP+MPS Robust Setup")
    cprint("• Verifies platform\n• Creates or overwrites venv-blip\n• Installs all dependencies\n"
           "• Optionally downloads test image and requirements.txt\n", Fore.CYAN)

    check_platform(force=args.force)
    find_python()
    create_or_overwrite_venv(VENV_DIR, auto_overwrite=args.overwrite)
    install_requirements(VENV_DIR, REQUIREMENTS)
    download_test_image(TEST_IMAGE_FILE, TEST_IMAGE_URL, auto_download=args.download_image)
    generate_requirements_txt(VENV_DIR, auto_generate=args.requirements)
    print_footer(VENV_DIR)

if __name__ == "__main__":
    main()