.. _getting_started:

***************
Getting started
***************

Installing ParticleSpy
======================

Install ParticleSpy
-------------------

The easiest way to install the latest stable build of ParticleSpy is via pip. This will install the package and its dependencies. You can do this by typing the following in to the command line:

.. code-block:: bash

    $ pip install ParticleSpy


Installing from Github
----------------------

If you would like to use a development version of Hyperspy downloaded from Github you need to have a python environment with Hyperspy installed. 
Full instructions for Hyperpsy installation can be found at 
`http://hyperspy.org/hyperspy-doc/v1.3/user_guide/install.html <http://hyperspy.org/hyperspy-doc/v1.3/user_guide/install.html>`_.

You then need to install from the git repository using git. If you have git installed 
you can use the following command to install the package.

.. code-block:: bash

    $ pip install -e git+https://github.com/ePSIC-DLS/ParticleSpy

Using ParticleSpy
=================

In order to use ParticleSpy you must first import the api by entering the following in a python console / Jupyter notebook:

.. code-block:: python

    >>> import ParticleSpy.api as ps

You are then able to start using the functionality of ParticleSpy. If you already have an image loaded as a Hyperspy signal object you can now use the SegUI function to choose segmentation parameters.

.. code-block:: python

    >>> ps.SegUI(image)

This will pop up a QT window in which you can try different segmentation parameters. Once you are happy with the parameters press Update one last time in order to save the current parameters. You can then load these parameters in 
to a parameters object by doing the following:

.. code-block:: python

    >>> params = ps.parameters()
    >>> params.load()

You can now use the chosen parameters to do your particle analysis:

.. code-block:: python

    >>> particles = ps.ParticleAnalysis(image,params)

The variable particles is now assigned to a particle list object which contains data on all of the segmented particles.

For further examples of using the package you can view the example Jupyter notebook `here <https://github.com/TomSlater/ParticleSpy/blob/master/ParticleSpy/Basic%20Example.ipynb>`_. 