.. _particle_analysis:

*******************
Particle Properties
*******************

One important aspect of ParticleSpy are the properties associated with each Particle object.

Native Properties
-----------------

When using the :py:meth:`~.ParticleAnalysis` function ParticleSpy is able to calculate a number of "native" properties listed in the table below.

.. table:: Particle properties

    +-----------------------------+------------------------------+----------------------------------------+
    | Property                    | Name                         | Description                            |
    +=============================+==============================+========================================+
    | Area                        | area                         | The particle area in calibrated units. |
    +-----------------------------+------------------------------+----------------------------------------+
    | Equivalent circular diameter| equivalent circular diameter | The diameter of a circle of the same   |
    |                             |                              | area as the particle.                  |
    +-----------------------------+------------------------------+----------------------------------------+
    | Major axis length           | major axis length            | The length of the longest axis of an   |
    |                             |                              | ellipse describing the particle.       |
    +-----------------------------+------------------------------+----------------------------------------+
    | Minor axis length           | minor axis length            | The length of the shortest axis of an  |
    |                             |                              | ellipse describing the particle.       |
    +-----------------------------+------------------------------+----------------------------------------+
    | Circularity                 | circularity                  | The circularity of the particle.       |
    +-----------------------------+------------------------------+----------------------------------------+
    | Eccentricity                | eccentricity                 | The eccentricity of the particle.      |
    +-----------------------------+------------------------------+----------------------------------------+
    | Total intensity             | intensity                    | The sum of intensity contained within  |
    |                             |                              | the segmented region of the image.     |
    +-----------------------------+------------------------------+----------------------------------------+


Adding Properties
-----------------

In addition to the "native" properties it is possible to add any additional properties using the :py:meth:`~.Particle.set_property` function.

.. code-block:: python

    >>> particles.list[i].set_property('property name', value, 'units')