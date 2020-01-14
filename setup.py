import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Hyphal_feature_tracking-liberjul",
    version="0.0.1",
    author="Julian Liber",
    author_email="liberjul@msu.edu",
    description="A package to track and analyze hyphal features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liberjul/Hyphal_feature_tracking",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
