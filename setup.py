"""Bundle optional JIT CUDA sources in wheels as well as source distributions."""
from pathlib import Path
from shutil import copy2

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / "neurosim_gpu" / "csrc"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("bindings.cpp", "runtime.cu", "compartmental.cu"):
            copy2(Path(__file__).parent / "csrc" / name, destination / name)


setup(cmdclass={"build_py": BuildPy})
