from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="your_package",  # Replace with your actual package name
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,  # Use dependencies from requirements.txt
)