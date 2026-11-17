"""Check version and optional CUDA source completeness of release archives."""
from pathlib import Path
import tarfile
import zipfile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    wheels = sorted((root / "dist").glob("neurosim_gpu-0.2.0-*.whl"))
    sdists = sorted((root / "dist").glob("neurosim_gpu-0.2.0.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("expected one 0.2.0 wheel and one source distribution")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        for name in ("bindings.cpp", "runtime.cu", "compartmental.cu"):
            packaged = archive.read("neurosim_gpu/csrc/" + name)
            if packaged != (root / "csrc" / name).read_bytes():
                raise SystemExit("wheel source mismatch: " + name)
        metadata = archive.read("neurosim_gpu-0.2.0.dist-info/METADATA").decode()
        if "Version: 0.2.0" not in metadata.splitlines():
            raise SystemExit("wheel version mismatch")
        for name in ("compartmental.py", "native_compartmental.py"):
            if "neurosim_gpu/" + name not in names:
                raise SystemExit("wheel module missing: " + name)
    with tarfile.open(sdists[0]) as archive:
        names = archive.getnames()
        for suffix in ("csrc/compartmental.cu", "setup.py", "tests/test_native_compartmental.py"):
            if not any(name.endswith("/" + suffix) for name in names):
                raise SystemExit("sdist missing: " + suffix)
    print("0.2.0 wheel and sdist versions/sources: PASS")


if __name__ == "__main__":
    main()
