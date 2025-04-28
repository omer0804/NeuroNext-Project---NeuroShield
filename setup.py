# setup for the package
from setuptools import setup, find_packages
from pathlib import Path    


setup(
    name="NeuroShield",
    version="0.1.0",
    author="Omer",
    author_email="omer0804@gmail.com",
    description="NeuroShield",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/omer0804/NeuroShield",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "mne",
        "torch",
        "torchaudio",
        "torchvision",
        "tensorflow",
        "keras",
        "flask",
        "fastapi",
        "uvicorn",
        "pytest",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)