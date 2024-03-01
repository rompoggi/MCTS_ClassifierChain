import io
import os

import setuptools

ROOT_DIR: str = os.path.dirname(__file__)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    """Read the README file."""
    return io.open(get_path("README.md"), "r", encoding="utf-8").read()


if __name__ == "__main__":
    setuptools.setup(
        name="mcts_inference",
        version="0.1",
        description="Monte Carlo Tree Search for Inference",
        author="Romain Poggi",
        author_email="romainpoggi323@gmail.com",
        packages=["mcts_inference"],
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Reinforcement Learning",
            "Topic :: Scientific/Engineering :: Bachelor Thesis",
            "License :: OSI Approved :: MIT License",
        ],
        python_requires=">=3.11",
    )
