.. _time_series:

***********
Time Series
***********

ParticleSpy is able to segment particles over a series of images and to track how properties evolve over time.
We use the trackpy library in order to relatively quickly track particles through an image series.

In order to peform particle analysis on an image series the :py:meth:`~.particle_analysis_series` function is used:

.. code-block:: python

    >>> particles = ps.particle_analysis_series(data, params)

This provides the exact same functionality as :py:meth:`~.particle_analysis` but also assigns a frame number to each particle.

In order to track particles over a time series, the :py:meth:`~.time_series_analysis` function can be used:

.. code-block:: python

    >>> time_series = ps.time_series_analysis(particles)

This function returns a pandas dataframe whose format is that of a trajectories object in trackpy.

The tracjectories of particles can be plotted using trackpy:

.. code-block:: python

    >>> import trackpy as tp
    >>> tp.plot_traj(time_series)

Particle properties can also be plotted versus frame number:

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> for index, particle in time_series.groupby('particle'):
    ...     plt.plot(particle['frame'], particle['area'], label=index)
