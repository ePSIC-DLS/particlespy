.. _segmentation:

************
Segmentation
************

ParticleSpy provides different options for the segmentation of particles from images.

Using a Pre-Segmented Mask
==========================

The most straightforward method is to supply ParticleSpy with a pre-segmented mask (boolean image).

.. code-block:: python

    >>> ParticleAnalysis(acquisition, parameters, mask = numpy array containg mask data)

In this way other software (e.g. ImageJ) could be used to perform segmentation which is then used by ParticleSpy to segment particles.

Using the Segmentation User Interface
=====================================

ParticleSpy can also perform its own segmentation using functions from scikit-image. A number of methods and options for segmentation are included and therefore ParticleSpy parameterizes the segmentation process. In order to choose the correct parameters for your segmentation ParticleSpy provides a Segmentation User Interface, that can be launched from a python kernel using the :py:meth:`~.SegUI` function.

.. code-block:: python

    >>> SegUI(image)

Once the Segmentation User Interface is launched the image is displayed and a number of options are available on the right hand side.

.. image:: _static/segui.png
    :align: center

(1) Rolling ball size
    
   The rolling ball algorithm is equivalent to a top hat filter. It acts to remove slowly varying background intensity at a size larger than the particle diameter. The default is to not apply a rolling ball (value = 0). To apply a rolling ball enter a value (in pixels) that is significantly larger than your particle diameter.

(2) Gaussian filter kernel
    
This is an option to apply a Gaussian filter to the image before segmentation, to assisst with noisy data. Typically, a value of 1 - 3 works well.

(3) Thresholding options
    
Here are a series of algorithms for determining thresholds at which to segment. Most of the options available in scikit-image are included, including the popular Otsu, Isodata and Li methods. From Otsu to Li are all global methods and will set one threshold value for the whole image. Below Li are local methods that will set threshold values that change over the image. Local and local Otsu work slightly differently but will provide a segmentation in which variations in intensity across an image are taken in to account. Local + Global Otsu performs both local and global thresholding.

.. warning::
   Methods using the Local Otsu may take a long time on large images.

(4) Local filter kernel
    
Select the size of the local filter kernel to use if using one of the local thresholding methods. A value between 2 - 10 times smaller than the image may work.

(5) Watershed
    
Tick to apply a watershed step to the segmented labels. This acts to separate touching objects. The watershed algorithm uses local maxima of the distance transform as seeds, the minimum separation of these local maxima is set by the min particle size option.

(6) Invert
    
Option to invert the image intensity when using a bright field image.

(7) Min particle size
    
This parameter has two uses. Firstly, it acts to remove any labels (objects) that have an area (in pixels) below this value. Secondly, it acts as the minimum separation for seeds in the watershed algorithm.

(8) Update
    
Updates the displayed image with the current parameters and updates the current parameters file.

(9) Get Params
    
Prints the current parameters as output for reference.

(10) Display
     
Option to display either the image with label boundaries displayed, or the solid labels coloured by label number.
