from setuptools import find_packages, setup  # type: ignore

# Read the content of the README file
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="sonnet",
    version="0.1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)
