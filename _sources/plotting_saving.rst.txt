.. _plotting_saving:

*******************
Plotting and Saving
*******************

One of the main benefits of ParticleSpy is the ability to plot particle properties natively.
The main method for plotting particle properties is :py:meth:`~.Particle_list.plot`.
This method plots a histogram of any specified property, as long as you have added this property to each particle (either automatically or manually).

An example of plotting a histogram of particle areas in shown here:

.. code-block:: python

    >>> particles = ps.ParticleAnalysis(data, params, mask=generated_mask)
    >>> particles.plot(area, bins = 20)

In the above code it is possible to plot particle area because this is automatically calculated in the ParticleAnalysis function.


Ploting Radial Profiles
-----------------------

ParticleSpy provides the ability to plot a radial profile (that is an intensity profile from particle centre to edge) of image intensity or EDS signal intensity.
A radial profile can be very useful for illustrating distributions in particles.

The following code shows how to plot a radial profile of image intensity.

.. code-block:: python

    >>> rp = ps.radial_profile(particle,['Image'],plot=True)
    
It is also possible to plot the intensity of certain elements from an EDS signal.

.. code-block:: python

    >>> rp = ps.radial_profile(particle,['Pt','Ni'],plot=True)

Saving Particle Images and Maps
-------------------------------

In order to save images and maps of particles it is necessary to use Hyperspy's save function.

.. code-block:: python

    >>> particles.list[0].image.save(filename)