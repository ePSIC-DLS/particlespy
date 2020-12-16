import numpy as np

from ParticleSpy.custom_kernels import membrane_projection
from skimage import filters, morphology
from skimage.measure import label, regionprops, perimeter
from skimage.exposure import rescale_intensity
from sklearn import preprocessing
from sklearn.cluster import DBSCAN, KMeans


def CreateFeatures(image, intensity = True, 
                          edges = True, 
                          texture = True, 
                          membrane = True, 
                          sigma = 1, high_sigma = 16, disk_size = 20):
    """
    Creates set of features for data classification

    Parameters
    ----------
    image : greyscale image for segmentation
    intensity : adds intensity based features if set as True
    edges : adds edges based features if set as True
    texture : adds textures based features if set as True
    membrane : adds membrane projection features if set as True

    Returns
    -------
    set of chosen features of the inputted image.
    """

    shape = [image.shape[0], image.shape[1], 1]

    selem = morphology.disk(disk_size)

    image_stack = np.zeros(shape)
    im_blur = filters.gaussian(image, sigma)
    one_im = rescale_intensity(image, out_range = (-1,1))

    if intensity:
        new_layer = np.reshape(filters.gaussian(image, sigma), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)

        new_layer = np.reshape(filters.difference_of_gaussians(image,low_sigma= sigma, high_sigma=high_sigma), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)

        new_layer = np.reshape(filters.median(one_im,selem), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)
        
        new_layer = np.reshape(filters.rank.minimum(one_im,selem), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)
        
        new_layer = np.reshape(filters.rank.maximum(one_im, selem), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)

    if edges:
        new_layer = np.reshape(filters.sobel(im_blur), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)

    if texture:
        new_layer = np.reshape(filters.hessian(im_blur, mode='constant',), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)

    if membrane:
        mem_layers = membrane_projection(image)
        image_stack = np.append(image_stack, mem_layers, axis=2)

    return image_stack[:,:,1:]

def ClusterLearn(image, method='KMeans', 
                        intensity = True, 
                        edges = True, 
                        texture = True, 
                        membrane = False, 
                        sigma = 1, high_sigma = 16, disk_size = 20):
    """
    Creates masks of given images using scikit learn clustering methods.
    
    Parameters
    ----------
    image: Hyperspy signal object or list of hyperspy signal objects.
        Hyperpsy signal object containing nanoparticle images
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

    image = image.data
    #image = preprocessing.maxabs_scale(image)
    shape = [image.shape[0], image.shape[1], 1]

    image_stack = CreateFeatures(image, intensity=intensity, edges=edges, texture=texture, membrane=membrane, 
                                 sigma = sigma, high_sigma = high_sigma, disk_size = disk_size)

    pixel_stacks = np.zeros([shape[0]*shape[1],image_stack.shape[2]])
    for i in range(shape[1]):
        pixel_stacks[i*shape[0]:(i+1)*shape[0],:] = image_stack[:,i,:]
        
    pixel_stacks = preprocessing.scale(pixel_stacks)

    if method == 'KMeans':
        labels = KMeans(n_clusters=2,init='random',n_init=10).fit_predict(pixel_stacks)
    elif method == 'DBscan':
        labels = DBSCAN().fit_predict(pixel_stacks)
        
    mask = np.zeros_like(image)
    for i in range(shape[1]):
        mask[:,i] = labels[i*shape[0]:(i+1)*shape[0]]
    
    return mask

def ClusterLearnSeries(image_set, method='KMeans', 
                        intensity = True, 
                        edges = True, 
                        texture = True, 
                        membrane = False, 
                        sigma = 1, high_sigma = 16, disk_size = 20):
    """
    Creates masks of sets of images using scikit learn clustering methods.
    
    Parameters
    ----------
    image_set: list of hyperspy signal objects.
        List of Hyperpsy signal object containing nanoparticle images
    method: Clustering algorithm used to generate mask.
    intensity, edges, texture, membrane: different kernel types used for 
    creating features
    disk_size: Size of the local pixel neighbourhood considered by select 
        segmentation methods.
    parameters: List of dictionaries of Parameters for segmentation methods used
    in clustering. The parameters can be inputted manually or use the default.

    Returns
    -------
    list of generated mask per image (1channel)
    """

    mask_set = []
    for image in image_set:
        mask_set.append(ClusterLearn(image,method=method, intensity=intensity, edges=edges, texture=texture, membrane=membrane, 
                                     sigma=sigma, high_sigma=high_sigma, disk_size=disk_size))
    
    return mask_set

def ClusterTrained(image, labels, classifier,
                        intensity = True, 
                        edges = True, 
                        texture = True, 
                        membrane = True, 
                        sigma = 1, high_sigma = 16, disk_size = 20):
    """
    Trains classifier and classifies an image.
    
    Parameters
    ----------
    image : Hyperspy signal object or list of hyperspy signal objects.
    labels : user-labelled mask
    classifier : empty or pretrained classifier to be trained on labelled data

    Returns
    -------
    classified mask (1 channel), trained classifier
    """
    if len(labels.shape) != 2:
            labels = toggle_channels(labels)
    #makes sure labels aren't empty
    if (labels != 0).any() == True:

        thin_mask = labels.astype(np.float64)
        shape = image.data.shape
        image = image.data

        features = CreateFeatures(image, intensity=intensity, edges=edges, texture=texture, membrane=membrane, 
                                         sigma=sigma, high_sigma=high_sigma, disk_size=disk_size)
        features = np.rot90(np.rot90(features, axes=(2,0)), axes=(1,2))
        #features are num/x/y

        training_data = features[:, thin_mask > 0].T
        #training data is number of labeled pixels by number of features
        training_labels = thin_mask[thin_mask > 0].ravel()
        training_labels = training_labels.astype('int')
        #training labels is labelled pixels in 1D array

        classifier.fit(training_data, training_labels)

        output = np.copy(thin_mask)
        if (labels == 0).any() == True:
            #train classifier on  labelled data
            data = features[:, thin_mask == 0].T
            #unlabelled data
            pred_labels = classifier.predict(data)
            #predict labels for rest of image

            output[thin_mask == 0] = pred_labels

        return output, classifier

def ClassifierSegment(classifier, image, 
                        intensity = True, 
                        edges = True, 
                        texture = True, 
                        membrane = True, 
                        sigma = 1, high_sigma = 16, disk_size = 20):
    """
    classifies image with pretrained classifier.
    
    Parameters
    ----------
    classifier : sklearn classifier
    image: numpy array of image

    Returns
    -------
    mask of labels (1channel)
    """
    features = CreateFeatures(image, intensity=intensity, edges=edges, texture=texture, membrane=membrane, 
                                     sigma=sigma, high_sigma=high_sigma, disk_size=disk_size)
    features = np.rot90(np.rot90(features, axes=(2,0)), axes=(1,2))
    features = features[:, image == image].T
    print(features.shape)
    mask = classifier.predict(features)

    output = np.copy(image)
    output[image == image] = mask

    return output


def toggle_channels(image, colors = ['#80A30015', '#806DA34D', '#8051E5FF', '#80BD2D87', '#80F5E663']):
    shape = image.shape

    if len(shape) == 3:
        toggled = np.zeros((shape[0],shape[1]), dtype = np.uint8)
        for i in range(len(colors)):
            rgb = [int(colors[i][3:5], 16), int(colors[i][5:7], 16), int(colors[i][7:], 16)]
            toggled[(image == rgb).all(axis=2)] = i+1

    elif len(shape) == 2:
        toggled = np.zeros((shape[0],shape[1],3), dtype = np.uint8)
        for i in range(len(colors)):
            rgb = [int(colors[i][3:5], 16), int(colors[i][5:7], 16), int(colors[i][7:], 16)]
            toggled[image == (i+1),:] = rgb

    return toggled