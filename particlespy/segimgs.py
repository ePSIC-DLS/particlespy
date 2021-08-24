import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, util
from skimage.exposure import rescale_intensity
from skimage.measure import label, perimeter, regionprops
from sklearn import preprocessing

from particlespy.custom_kernels import (custom_kernel, laplacian,
                                        membrane_projection)
from particlespy.particle_analysis import trainable_parameters


def create_features(image, parameters=None):
    """
    Creates set of features for data classification

    Parameters
    ----------
    image : greyscale image for segmentation
    trainable segmentation parameters
    
    parameters : trainable parameters object for setting filter kernels and their parameters

    Returns
    -------
    set of chosen features of the inputted image.
    """
    if parameters == None:
        parameters = trainable_parameters()
    shape = [image.shape[0], image.shape[1], 1]

    image_stack = np.zeros(shape, dtype=np.float16)
    one_im = rescale_intensity(image, out_range = (-1,1))
    temp = util.img_as_ubyte(rescale_intensity(image, out_range=(0,1)))

    if parameters.gaussian[0]:
        new_layer = np.reshape(filters.gaussian(image,parameters.gaussian[1]),shape)
        image_stack = np.concatenate((image_stack, new_layer), axis=2)

    if parameters.diff_gaussian[0]:
        par = parameters.diff_gaussian
        if par[1][0]:
            blur = filters.gaussian(image,par[1][1])
            new_layer = np.reshape(filters.difference_of_gaussians(blur, low_sigma=par[2],high_sigma=par[3]), shape)
        else:
            new_layer = np.reshape(filters.difference_of_gaussians(image, low_sigma=par[2],high_sigma=par[3]), shape)
        image_stack = np.concatenate((image_stack, new_layer), axis=2)

    if parameters.median[0]:
        par = parameters.median
        if par[1][0]:
            blur = filters.gaussian(one_im,par[1][1])
            new_layer = np.reshape(filters.median(blur, morphology.disk(par[2])),shape)
        else:
            new_layer = np.reshape(filters.median(one_im, morphology.disk(par[2])),shape)
        image_stack = np.concatenate((image_stack, new_layer), axis=2)

    if parameters.minimum[0]:
        par = parameters.minimum
        if par[1][0]:
            blur = filters.gaussian(temp,par[1][1])
            new_layer = np.reshape(filters.rank.minimum(blur, morphology.disk(par[2])),shape)
        else:
            new_layer = np.reshape(filters.rank.minimum(temp, morphology.disk(par[2])),shape)
        image_stack = np.concatenate((image_stack, new_layer), axis=2)
    
    if parameters.maximum[0]:
        par = parameters.maximum
        if par[1][0]:
            blur = filters.gaussian(temp,par[1][1])
            new_layer = np.reshape(filters.rank.maximum(blur, morphology.disk(par[2])),shape)
        else:
            new_layer = np.reshape(filters.rank.maximum(temp, morphology.disk(par[2])),shape)
        image_stack = np.concatenate((image_stack, new_layer), axis=2)

    if parameters.sobel[0]:
        par = parameters.sobel
        if par[1][0]:
            blur = filters.gaussian(image,par[1][1])
            new_layer = np.reshape(filters.sobel(blur),shape)
        else:
            new_layer = np.reshape(filters.sobel(image),shape)
        image_stack = np.concatenate((image_stack, new_layer), axis=2)

    if parameters.hessian[0]:
        par = parameters.hessian
        if par[1][0]:
            blur = filters.gaussian(image,par[1][1])
            new_layer = np.reshape(filters.hessian(blur),shape)
        else:
            new_layer = np.reshape(filters.hessian(image),shape)
        image_stack = np.concatenate((image_stack, new_layer), axis=2)

    if parameters.laplacian[0]:
        par = parameters.laplacian
        blur = image
        if par[1][0]:
            blur = filters.gaussian(image,par[1][1])
        new_layer = np.reshape(laplacian(blur),shape)
        image_stack = np.concatenate((image_stack, new_layer), axis=2)

    if True in parameters.membrane[1:]:
        par = parameters.membrane
        if par[0][0]:
            temp_im = filters.gaussian(image,par[0][1])
        else:
            temp_im = image
        indexes = np.asarray(par[1:], dtype=np.bool_)
        mem_layers = membrane_projection(image)[:,:,indexes]

        if par[1:] == [1,1,1,1,1,1]:
            mem_layers = np.squeeze(mem_layers)
        image_stack = np.append(image_stack, mem_layers, axis=2)
    
    if parameters.custom[0]:
        par = parameters.custom
        blur = image
        if par[1][0]:
            blur = filters.gaussian(image,par[1][1])
        new_layer = np.reshape(custom_kernel(blur,np.asarray(par[2])), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)
    
    return image_stack[:,:,1:]

def cluster_learn(images, clust, parameters = None):
    """
    Creates masks of given images using scikit learn clustering methods.
    
    Parameters
    ----------
    image: Hyperspy signal object.
        Hyperspy signal object containing nanoparticle images
    method: Clustering algorithm used to generate mask.
    intensity, edges, texture, membrane: different kernel types used for 
    creating features
    disk_size: Size of the local pixel neighbourhood considered by select 
        segmentation methods.
    parameters: List of dictionaries of Parameters for segmentation methods used
    in clustering. The parameters can be inputted manually or use the default.

    Returns
    -------
    generated mask (1channel)
    """
    
    if isinstance(images, list) == False:
        image = [images]
        
    if parameters == None:
        parameters = trainable_parameters()
    
    mask = []
    
    for i, image in enumerate(images):
        image = image.data
        #image = preprocessing.maxabs_scale(image)
        shape = [image.shape[0], image.shape[1], 1]

        image_stack = create_features(image, parameters=parameters)

        pixel_stacks = np.zeros([shape[0]*shape[1],image_stack.shape[2]])
        for ii in range(shape[1]):
            pixel_stacks[ii*shape[0]:(ii+1)*shape[0],:] = image_stack[:,ii,:]
            
        pixel_stacks = preprocessing.scale(pixel_stacks)
        labels = clust.fit_predict(pixel_stacks)

        mask.append(np.zeros_like(image))
        for ii in range(shape[1]):
            mask[i][:,ii] = labels[ii*shape[0]:(ii+1)*shape[0]]
    
    if len(mask) == 1:
        mask = mask[0]
        
    return mask

def cluster_trained(image, labels, classifier, parameters = None):
    """
    Trains classifier and classifies an image.
    
    Parameters
    ----------
    image : Hyperspy signal object or list of hyperspy signal objects.
    labels : user-labelled mask or list of user labelled masks
    classifier : empty classifier to be trained on labelled data

    Returns
    -------
    classified mask (1 channel), trained classifier
    """
    if isinstance(image, list) == False:
        image = [image]
        labels = [labels]
        #changes single images into a list

    if parameters == None:
        parameters = trainable_parameters()
        
    features = []
    for i in range(len(image)):

        if len(labels[i].shape) != 2:
                labels[i] = toggle_channels(labels[i])

        print((labels[i] != 0).any())
        #makes sure labels aren't empty
        if (labels[i] != 0).any() == True:

            thin_mask = labels[i].astype(np.float16)
            shape = image[i].data.shape
            image[i] = image[i].data

            features.append(create_features(image[i], parameters=parameters))
            features[i] = np.rot90(np.rot90(features[i], axes=(2,0)), axes=(1,2))
            #features are num/x/y

            training_data = features[i][:, thin_mask > 0].T
            #training data is number of labeled pixels by number of features
            training_labels = thin_mask[thin_mask > 0].ravel()
            training_labels = training_labels.astype('int')
            #training labels is labelled pixels in 1D array
            
            if i == 0:
                training_data_long = training_data
                training_labels_long = training_labels
            else:
                training_data_long = np.concatenate((training_data_long, training_data))
                training_labels_long = np.concatenate((training_labels_long,training_labels))

    classifier.fit(training_data_long, training_labels_long)
    #will crash for one image with no labels

    output = []
    for i in range(len(image)):

        thin_mask = labels[i].astype(np.float16)
        output.append(np.copy(thin_mask))
        #list assingment index out of range
        if (labels[i] == 0).any() == True:
            #train classifier on  labelled data
            data = features[i][:, thin_mask == 0].T
            #unlabelled data
            pred_labels = classifier.predict(data)
            #predict labels for rest of image

            output[i][thin_mask == 0] = pred_labels
            #adds predicted labels to unlabelled data

    if len(output) == 1:
        output = output[0]
    #changes list of output into one image for single image training

    return output, classifier

def classifier_segment(classifier, image, parameters = None):
    """
    classifies image with pretrained classifier.
    
    Parameters
    ----------
    classifier : sklearn classifier
    image: numpy array of image

    Returns
    -------
    mask of labels (1 channel, indexed)
    """
    features = create_features(image, parameters=parameters)
    features = np.rot90(np.rot90(features, axes=(2,0)), axes=(1,2))
    features = features[:, image == image].T
    mask = classifier.predict(features)

    output = np.copy(image)
    output[image == image] = mask

    return output


def toggle_channels(image, colors = ['#A30015', '#6DA34D', '#51E5FF', '#BD2D87', '#F5E663']):
    """
    changes a 3 channel RGB image into a 1 channel indexed image, and vice versa
    
    Parameters
    ----------
    image : 1/3 channel image for conversion
    colors : ordered list of colors to index labels by
    
    Returns
    ----------
    toggled 1/3 channel image
    
    """
    #colors are in RGB format
    shape = image.shape

    if len(shape) == 3:
        toggled = np.zeros((shape[0],shape[1]), dtype = np.uint8)
        for i in range(len(colors)):
            rgb = [int(colors[i][1:3], 16), int(colors[i][3:5], 16), int(colors[i][5:], 16)]
            toggled[(image == rgb).all(axis=2)] = i+1

    elif len(shape) == 2:
        toggled = np.zeros((shape[0],shape[1],3), dtype = np.uint8)
        for i in range(len(colors)):
            rgb = [int(colors[i][1:3], 16), int(colors[i][3:5], 16), int(colors[i][5:], 16)]
            toggled[image == (i+1),:] = rgb

    return toggled

def remove_large_objects(ar, max_size=200, connectivity=1, in_place=False):
    """Remove objects larger than the specified size.

    Expects ar to be an array with labeled objects, and removes objects
    larger than max_size. If `ar` is bool, the image is first labeled.
    This leads to potentially different behavior for bool and 0-and-1
    arrays.

    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the objects of interest. If the array type is
        int, the ints must be non-negative.
    max_size : int, optional (default: 200)
        The largest allowable object size.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel. Used during
        labelling if `ar` is bool.
    in_place : bool, optional (default: False)
        If ``True``, remove the objects in the input array itself.
        Otherwise, make a copy.

    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or string.
    ValueError
        If the input array contains negative values.

    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small connected components removed.
        # Raising type error if not int or bool
    if not (ar.dtype == bool or np.issubdtype(ar.dtype, np.integer)):
        raise TypeError("Only bool or integer image types are supported. "
                        "Got %s." % ar.dtype)
    """
    if in_place:
        out = ar
    else:
        out = ar.copy()

    if max_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    too_small = component_sizes > max_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out
