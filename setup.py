from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="marketing-attribution-optimizer",
    version="1.0.0",
    author="Chinmay Shrivastava",
    author_email="your.email@example.com",
    description="Production-ready marketing attribution and ROI optimization system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/marketing-attribution-optimizer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black", "flake8", "mypy"],
        "dashboard": ["streamlit>=1.25.0", "plotly>=5.15.0"],
    },
)
