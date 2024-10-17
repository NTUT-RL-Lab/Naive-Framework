from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="env",
    version="0.0.1",
    description="ENV",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["gymnasium==0.29.1", "stable-baselines3==2.3.2"],
)
