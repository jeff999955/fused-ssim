from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension
import torch
import sys
import os

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stderr.reconfigure(line_buffering=True)


def log(msg):
    """Print to both stdout and stderr."""
    print(msg)
    print(msg, file=sys.stderr, flush=True)


def get_configured_cuda_arch_list():
    """Return configured CUDA arch list from env, preferring PyTorch's convention."""
    torch_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if torch_arch_list:
        return "TORCH_CUDA_ARCH_LIST", torch_arch_list

    cuda_architectures = os.environ.get("CUDA_ARCHITECTURES")
    if cuda_architectures:
        return "CUDA_ARCHITECTURES", cuda_architectures

    return None, None


def normalize_cuda_arch(arch):
    """Convert 7.5 / 75 / 8.6+PTX to nvcc-compatible compute capability tokens."""
    arch = arch.strip()
    if not arch:
        raise ValueError("empty CUDA architecture entry")

    arch, plus_ptx, ptx_suffix = arch.partition("+")
    normalized = arch.replace(".", "")

    if not normalized.isdigit():
        raise ValueError(f"invalid CUDA architecture '{arch}'")

    return normalized, bool(plus_ptx and ptx_suffix.upper() == "PTX")


def configure_cuda():
    """Configure CUDA/ROCm backend."""
    fallback_archs = [
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_89,code=sm_89",
    ]

    log("Compiling for CUDA.")
    compiler_args = {"cxx": ["-O3", "-DFUSED_SSIM_CUDA"], "nvcc": ["-O3", "-DFUSED_SSIM_CUDA"]}

    if torch.version.hip:
        log("Detected AMD GPU with ROCm/HIP")
        compiler_args["nvcc"].append("-ffast-math")
        detected_arch = "AMD GPU (ROCm/HIP)"
    else:
        compiler_args["nvcc"].extend(("--maxrregcount=32", "--use_fast_math"))

        # Respect PyTorch's TORCH_CUDA_ARCH_LIST behavior when provided.
        cuda_archs_env_name, cuda_archs_env = get_configured_cuda_arch_list()
        arch_configured = False

        if cuda_archs_env:
            try:
                archs = [arch.strip() for arch in cuda_archs_env.replace(" ", ";").split(';') if arch.strip()]
                log(f"Using CUDA architectures from {cuda_archs_env_name}: {archs}")
                if cuda_archs_env_name == "TORCH_CUDA_ARCH_LIST":
                    # Let torch.utils.cpp_extension translate the env var exactly like PyTorch/PyTorch3D.
                    detected_arch = f"{cuda_archs_env_name}:{';'.join(archs)}"
                else:
                    for arch in archs:
                        normalized_arch, include_ptx = normalize_cuda_arch(arch)
                        compiler_args["nvcc"].append(
                            f"-gencode=arch=compute_{normalized_arch},code=sm_{normalized_arch}"
                        )
                        if include_ptx:
                            compiler_args["nvcc"].append(
                                f"-gencode=arch=compute_{normalized_arch},code=compute_{normalized_arch}"
                            )
                    detected_arch = f"{cuda_archs_env_name}:{';'.join(archs)}"
                arch_configured = True
            except Exception as e:
                log(f"Failed to parse {cuda_archs_env_name}: {e}. Trying device detection.")

        # Try device detection if environment variable not set or failed
        if not arch_configured:
            try:
                device = torch.cuda.current_device()
                compute_capability = torch.cuda.get_device_capability(device)
                arch = f"sm_{compute_capability[0]}{compute_capability[1]}"
                log(f"Detected GPU architecture: {arch}")
                compiler_args["nvcc"].append(f"-arch={arch}")
                detected_arch = arch
                arch_configured = True
            except Exception as e:
                log(f"Failed to detect GPU architecture: {e}. Falling back to multiple architectures.")

        # Fallback to multiple architectures if both methods failed
        if not arch_configured:
            compiler_args["nvcc"].extend(fallback_archs)
            detected_arch = "multiple architectures"

    return CUDAExtension, ["ssim.cu", "ssim3d.cu", "ext.cpp"], "fused_ssim_cuda", compiler_args, [], detected_arch


def configure_mps():
    """Configure Apple MPS backend."""
    log("Compiling for MPS.")
    compiler_args = {"cxx": ["-O3", "-std=c++17", "-ObjC++", "-Wno-unused-parameter"]}
    link_args = ["-framework", "Metal", "-framework", "Foundation"]
    return CppExtension, ["ssim.mm","ext.cpp"], "fused_ssim_mps", compiler_args, link_args, "Apple Silicon (MPS)"


def configure_xpu():
    """Configure Intel XPU (SYCL) backend."""
    log("Compiling for XPU.")
    os.environ['CXX'] = 'icpx'

    compiler_args = {"cxx": ["-O3", "-std=c++17", "-fsycl"]}
    link_args = ["-fsycl"]

    try:
        device_name = torch.xpu.get_device_name(0)
        log(f"Detected Intel XPU: {device_name}")
        detected_arch = f"Intel XPU (SYCL) - {device_name}"
    except Exception:
        log("Detected Intel XPU (SYCL)")
        detected_arch = "Intel XPU (SYCL)"

    return CppExtension, ["ssim_sycl.cpp","ext.cpp"], "fused_ssim_xpu", compiler_args, link_args, detected_arch


# Detect backend
if torch.cuda.is_available():
    extension_type, extension_files, build_name, compiler_args, link_args, detected_arch = configure_cuda()
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    extension_type, extension_files, build_name, compiler_args, link_args, detected_arch = configure_mps()
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    extension_type, extension_files, build_name, compiler_args, link_args, detected_arch = configure_xpu()
else:
    extension_type, extension_files, build_name, compiler_args, link_args, detected_arch = configure_cuda()

# Create a custom class that prints the architecture information
class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        # For SYCL, override compiler to use icpx
        if 'xpu' in build_name:
            self.compiler.compiler_so = ['icpx'] + self.compiler.compiler_so[1:]
            self.compiler.compiler_cxx = ['icpx'] + self.compiler.compiler_cxx[1:]
            self.compiler.linker_so = ['icpx'] + self.compiler.linker_so[1:]

        arch_info = f"Building with GPU architecture: {detected_arch if detected_arch else 'multiple architectures'}"
        print("\n" + "="*50)
        print(arch_info)
        print("="*50 + "\n")
        super().build_extensions()

setup(
    name="fused_ssim",
    packages=['fused_ssim'],
    ext_modules=[
        extension_type(
            name=build_name,
            sources=extension_files,
            extra_compile_args=compiler_args,
            extra_link_args=link_args
        )
    ],
    cmdclass={
        'build_ext': CustomBuildExtension
    }
)

# Print again at the end of setup.py execution
if "nvcc" in compiler_args:
    final_msg = "Setup completed. NVCC args: {}. CXX args: {}. Link args: {}.".format(
        compiler_args["nvcc"], compiler_args["cxx"], link_args
    )
else:
    final_msg = "Setup completed. CXX args: {}. Link args: {}.".format(
        compiler_args["cxx"], link_args
    )
print(final_msg)
