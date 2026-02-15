"""Setup script for adaptive-gradient-boosting-with-dynamic-feature-synthesis."""

from setuptools import setup, find_packages

setup(
    name="adaptive-gradient-boosting-with-dynamic-feature-synthesis",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)
