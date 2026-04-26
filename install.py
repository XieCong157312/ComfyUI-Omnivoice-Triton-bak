"""Installation script for ComfyUI-Omnivoice-Triton.

BEST PRACTICES (adapted from Saganaki22/ComfyUI-OmniVoice-TTS):
1. NEVER touch torch / torchaudio / torchvision — ComfyUI manages these.
2. Use --no-deps for any package that might pull in a torch version different
   from the one ComfyUI ships with.
3. Install packages individually for better error tracking.
4. Verify installation at the end.

WHY THIS EXISTS:
`omnivoice-triton` depends on `omnivoice`, which in turn declares `torch>=2.4`
and `torchaudio>=2.4`. A naive `pip install omnivoice-triton` can replace
ComfyUI's CUDA-enabled torch with a CPU-only wheel depending on the
resolver's choice, breaking GPU acceleration. We work around this by
installing `omnivoice-triton` and `omnivoice` with `--no-deps` and then
pulling in the remaining, torch-independent dependencies explicitly.

Run from the ComfyUI venv (same Python that runs ComfyUI):

    python ComfyUI/custom_nodes/ComfyUI-Omnivoice-Triton/install.py
"""

import importlib
import importlib.util
import subprocess
import sys


OMNIVOICE_TRITON_VERSION = "0.1.0"
OMNIVOICE_VERSION = "0.1.4"
TRANSFORMERS_MIN = "5.3.0"

CORE_PACKAGES = [
    ("omnivoice", f"omnivoice=={OMNIVOICE_VERSION}"),
    ("omnivoice_triton", f"omnivoice-triton=={OMNIVOICE_TRITON_VERSION}"),
]

# (import_name, pip_spec, label, no_deps?)
EXTRA_PACKAGES = [
    ("accelerate", "accelerate", "HF accelerate", False),
    ("huggingface_hub", "huggingface-hub", "HF hub client", False),
    ("transformers", f"transformers>={TRANSFORMERS_MIN}", "HF transformers", False),
    ("triton", "triton>=3.4.0", "Triton kernels runtime", True),
    # sageattention pins a specific torch ABI; install with --no-deps so pip
    # does not try to replace torch with what sageattention prefers.
    ("sageattention", "sageattention", "SageAttention kernels", True),
    ("soundfile", "soundfile", "Audio I/O", True),
    ("numpy", "numpy", "NumPy", True),
    ("librosa", "librosa", "Audio processing", True),
    ("soxr", "soxr", "Audio resampling", True),
    ("pydub", "pydub", "Audio manipulation", True),
    ("scipy", "scipy", "SciPy (librosa transitive)", True),
    ("lazy_loader", "lazy_loader", "Lazy loader (librosa transitive)", True),
    ("sentencepiece", "sentencepiece", "Tokenizer", True),
    ("jieba", "jieba", "Chinese segmenter", True),
]

REQUIRED_IMPORTS = [
    ("omnivoice", "omnivoice"),
    ("omnivoice_triton", "omnivoice-triton"),
    ("transformers", "transformers"),
    ("triton", "triton"),
    ("accelerate", "accelerate"),
    ("huggingface_hub", "huggingface-hub"),
    ("sageattention", "sageattention"),
    ("soundfile", "soundfile"),
    ("numpy", "numpy"),
    ("librosa", "librosa"),
    ("soxr", "soxr"),
    ("pydub", "pydub"),
    ("scipy", "scipy"),
    ("lazy_loader", "lazy_loader"),
    ("sentencepiece", "sentencepiece"),
    ("jieba", "jieba"),
]


def run_cmd(cmd, timeout=300):
    """Run a command; return (success, stdout, stderr)."""
    print(f"[Omnivoice-Triton] $ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return True, result.stdout, result.stderr
        print(f"[Omnivoice-Triton] Command failed: {result.stderr.strip()[:500]}")
        return False, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"[Omnivoice-Triton] Timeout after {timeout}s")
        return False, "", "Timeout"
    except Exception as exc:  # noqa: BLE001
        print(f"[Omnivoice-Triton] Error: {exc}")
        return False, "", str(exc)


def is_installed(module_name):
    return importlib.util.find_spec(module_name) is not None


def pip_install(package, no_deps=False, upgrade=False, ignore_requires_python=False):
    """Install one package, preferring uv if available, else pip."""
    flags = []
    if no_deps:
        flags.append("--no-deps")
    if upgrade:
        flags.append("--upgrade")
    if ignore_requires_python:
        flags.append("--ignore-requires-python")

    pip_cmd = [sys.executable, "-m", "pip", "install", package] + flags
    if ignore_requires_python:
        # uv pip install does not support pip's --ignore-requires-python flag.
        # Use pip directly for PyPI packages that require Python 3.12 while
        # some ComfyUI installs still run Python 3.11.
        ok, _, _ = run_cmd(pip_cmd)
        return ok

    # Prefer uv (much faster) if the current Python has it.
    uv_cmd = [sys.executable, "-m", "uv", "pip", "install", package] + flags
    ok, _, _ = run_cmd(uv_cmd)
    if ok:
        return True

    ok, _, _ = run_cmd(pip_cmd)
    return ok


def check_torch():
    """Return (version, has_gpu) for the active torch install, if any."""
    try:
        import torch  # noqa: PLC0415

        has_gpu = torch.cuda.is_available() or (
            hasattr(torch, "xpu") and torch.xpu.is_available()
        )
        return torch.__version__, has_gpu
    except ImportError:
        return None, False


def _verify_required_imports(required_imports=REQUIRED_IMPORTS):
    importlib.invalidate_caches()

    ok = True
    for module_name, pip_name in required_imports:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            print(f"  [FAIL] {pip_name}: not found on disk")
            ok = False
            continue
        try:
            importlib.import_module(module_name)
            print(f"  [OK]   {pip_name}")
        except Exception as exc:  # noqa: BLE001
            print(
                f"  [FAIL] {pip_name}: on disk but failed to import "
                f"({type(exc).__name__}: {exc})"
            )
            ok = False

    return ok


def main():
    # Fast path: if everything already imports cleanly, skip.
    try:
        import omnivoice_triton  # noqa: PLC0415
        import omnivoice  # noqa: F401, PLC0415
        import transformers  # noqa: PLC0415

        tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        if tv >= (5, 3):
            print("[Omnivoice-Triton] Core dependencies import; verifying full set.")
            if _verify_required_imports():
                print(
                    f"[Omnivoice-Triton] All dependencies present "
                    f"(omnivoice-triton={omnivoice_triton.__version__}, "
                    f"transformers={transformers.__version__}). Skipping."
                )
                return 0
            print("[Omnivoice-Triton] Some dependencies are missing; continuing install.")
        else:
            print(
                "[Omnivoice-Triton] WARNING: transformers "
                f"{transformers.__version__} is too old (need >= {TRANSFORMERS_MIN}). "
                "Continuing with install — this may upgrade transformers and "
                "could break other custom nodes that pin it."
            )
    except (ImportError, ValueError, AttributeError):
        pass

    print("=" * 60)
    print("[Omnivoice-Triton] Installation starting")
    print("=" * 60)

    # Step 1 — sanity-check torch, but never modify it.
    print("\n[Omnivoice-Triton] Step 1: Checking PyTorch")
    torch_version, has_gpu = check_torch()
    if torch_version is None:
        print(
            "[Omnivoice-Triton] ERROR: PyTorch is not installed. "
            "ComfyUI requires torch — install ComfyUI first."
        )
        return 1

    if has_gpu:
        print(f"[Omnivoice-Triton] PyTorch {torch_version} with GPU — OK")
    else:
        print(
            f"[Omnivoice-Triton] WARNING: PyTorch {torch_version} has no GPU. "
            "omnivoice-triton needs CUDA for the triton / hybrid modes."
        )

    # Step 2 — install omnivoice-triton and omnivoice with --no-deps
    print(
        "\n[Omnivoice-Triton] Step 2: Installing omnivoice + omnivoice-triton "
        "with --no-deps (to protect your PyTorch install)"
    )

    success = True
    for import_name, pip_spec in CORE_PACKAGES:
        if is_installed(import_name):
            # Force reinstall at the pinned version even if already present;
            # --no-deps ensures torch stays untouched.
            pass
        if not pip_install(
            pip_spec,
            no_deps=True,
            upgrade=True,
            ignore_requires_python=True,
        ):
            print(
                f"[Omnivoice-Triton] ERROR: failed to install {pip_spec}. "
                f"Try manually: pip install {pip_spec} --no-deps --ignore-requires-python"
            )
            success = False

    # Step 3 — install the remaining, torch-independent transitive dependencies.
    # These come from omnivoice-triton 0.1.0 + omnivoice 0.1.4 requires_dist,
    # minus anything that could pull in torch.
    print("\n[Omnivoice-Triton] Step 3: Installing remaining dependencies")

    for import_name, pip_name, label, no_deps in EXTRA_PACKAGES:
        if is_installed(import_name):
            print(f"[Omnivoice-Triton] {label} ({pip_name}) — already present")
            continue
        print(f"[Omnivoice-Triton] Installing {label} ({pip_name})")
        ok = pip_install(pip_name, no_deps=no_deps)
        if not ok:
            print(f"[Omnivoice-Triton] ERROR: failed to install {pip_name}")
            success = False

    # Step 4 — final verification
    print("\n" + "=" * 60)
    print("[Omnivoice-Triton] Verification")
    print("=" * 60)

    success = _verify_required_imports() and success

    torch_version_after, has_gpu_after = check_torch()
    if torch_version_after and has_gpu_after:
        print(f"  [OK]   PyTorch {torch_version_after} (GPU)")
    elif torch_version_after:
        print(f"  [WARN] PyTorch {torch_version_after} (no GPU)")
    else:
        print("  [FAIL] PyTorch — not installed")
        success = False

    # If torch lost its GPU support during install (shouldn't happen with
    # --no-deps, but just in case), give a clear recovery command.
    if has_gpu and not has_gpu_after:
        print(
            "\n"
            + "=" * 60
            + "\n[Omnivoice-Triton] WARNING: PyTorch lost its GPU support.\n"
            "Restore it with a CUDA wheel matching your driver, e.g.:\n"
            "  pip install --upgrade --force-reinstall \\\n"
            "    torch torchaudio \\\n"
            "    --index-url https://download.pytorch.org/whl/cu128\n"
            "Or see https://pytorch.org/get-started/locally/\n"
            + "=" * 60
        )
        success = False

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
