import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires=["hyperspy",
                  "scikit-image>=0.15",
                  "scikit-learn>=0.21.0]

try:
    import PyQt5  # noqa
except ImportError:
    try:
        import PySide2  # noqa
    except ImportError:
        install_requires.append('PyQt5')

setuptools.setup(
    name="particlespy",
    package_dir={'particlespy':'particlespy'},
    version="0.2.0",
    author="Thomas Slater",
    author_email="tjaslater@gmail.com",
    description="A package to perform particle segmentation and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ePSIC-DLS/ParticleSpy",
    packages=setuptools.find_packages(),
    install_requires=install_requires,   
    include_package_data=True,
    package_data={
        'ParticleSpy':
        [
            'Data/*.dm4',
            'Parameters/*.hdf5'
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
