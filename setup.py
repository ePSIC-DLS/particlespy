import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ParticleSpy",
    version="0.0.1",
    author="Thomas Slater",
    author_email="tjaslater@gmail.com",
    description="A package to perform particle segmentation and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ePSIC-DLS/ParticleSpy",
    packages=setuptools.find_packages(),
    install_requires=[
        "hyperspy",
        "PyQt5",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)