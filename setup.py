import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires=[
    "hyperspy>=1.7",
    "scikit-image>=0.17.1",
    "scikit-learn>=0.21",
    "trackpy",
    "numpy>=1.16.5",
    "qtpy",
]

setuptools.setup(
    name="particlespy",
    package_dir={'particlespy':'particlespy'},
    version="0.6.3",
    author="Thomas Slater",
    author_email="tjaslater@gmail.com",
    description="A package to perform particle segmentation and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ePSIC-DLS/particlespy",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    extras_require={
        "pyqt5": ["PyQt5>=5.14.0"],
        "pyqt6": ["PyQt6"],
    },
    package_data={
        'particlespy':
        [
            'parameters/*'
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
