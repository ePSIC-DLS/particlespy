.. _particle_analysis:

*****************
Particle Analysis
*****************

Once you have chosen your segmentation parameters and successfully prepared a prameters object you can use this to perform Particle Analysis.

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> data = hs.load('ParticleSpy/Data/JEOL HAADF Image.dm4')
    >>> params = ps.parameters()
    >>> params.load()
    >>> particles = ps.ParticleAnalysis(data, params)

Particle Analysis will run the segmentation on your data and calculate a number of parameters for each particle.

The calculated parameters include:

* Area

* Equivalent circular diameter

* Major and minor axes lengths

* Circularity

* Eccentricity

* Total particle intensity

* Zone axis (if atomic resolution images of fcc structure, looking to expand this!)

.. code-block:: python

    >>> #Syntax for accessing particle properties.
    >>> particles.list[0].properties['area']

Combining Particles from Multiple Images
----------------------------------------

It is possible to analyse particles from multiple images by passing a previously populated Particle_list object to :py:meth:`~.ParticleAnalysis` instead of returning a new Particle_list.
For example:

.. code-block:: python

    >>> ps.ParticleAnalysis(data, params, particles=particles)

EDS Analysis
------------

In addition, Particle Analysis will also segment and process EDS data if given as an additional dataset in the data list.

Particle Analysis can do the following processing on EDS data:

* Obtain the EDS spectrum of each particle.

* Obtain elemental maps of each particle.

* Get the composition of each particle if k-factors or zeta-factors are supplied in the parameters object.

.. code-block:: python

    >>> params = ps.parameters.load()
    >>> params.generate_eds(eds_method='CL',elements=['Pt','Au'],factors=[1.7,1.9],store_maps=False)
    >>> particles = ps.ParticleAnalysis(data, params)

Particle Segmentation with a Pre-Generated Mask
-----------------------------------------------

:py:meth:`~.ParticleAnalysis` will also accept pre-generated masks, either generated externally or through the manual option of :py:meth:`~.SegUI`.
In order to use a pre-generated mask it is possible to pass a mask argument to :py:meth:`~.ParticleAnalysis`.

.. code-block:: python

    >>> generated_mask = hs.load('maskfile')
    >>> params = ps.parameters.load() # This isn't used if you load a pre-generated mask but you still have to pass it.
    >>> particles = ps.ParticleAnalysis(data, params, mask=generated_mask)

If you have used the manual segmentation editor in :py:meth:`~.SegUI` you can simply pass 'UI' as the mask argument.

.. code-block:: python
    
    >>> particles = ps.ParticleAnalysis(data, params, mask='UI')
    
Normalize Particle Image Sizes
------------------------------
Sometimes further processing requires that all particle images have the same dimensions.
In ParticleSpy this can be readily achieved using the :py:meth:`~.Particle_list.normalize_boxing` function.
The function will set all image dimensions to the largest x and y values in the particle list.

.. code-block:: python

    >>> particles.normalize_boxing()
