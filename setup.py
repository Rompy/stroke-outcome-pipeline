"""
Setup script for Stroke Outcome Pipeline
Allows installation via: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stroke-outcome-pipeline",
    version="1.0.0",
    author="Junsu Kim, Ji Hoon Kim, Arom Choi",
    author_email="aromchoi@yuhs.ac",
    description="Privacy-preserving pipeline for stroke outcome prediction using local LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/stroke-outcome-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "stroke-extract=src.extraction_pipeline:main",
            "stroke-predict=src.prediction.outcome_predictor:main_prediction_pipeline",
            "stroke-generate-data=scripts.generate_synthetic_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "config/*.json"],
    },
    keywords=[
        "stroke",
        "clinical-nlp",
        "llm",
        "machine-learning",
        "healthcare",
        "privacy-preserving-ai",
        "lora",
        "rag",
        "outcome-prediction",
    ],
)
