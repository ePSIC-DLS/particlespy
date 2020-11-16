import numpy as np

from ParticleSpy.custom_kernels import membrane_projection
from skimage import filters, morphology
from skimage.measure import label, regionprops, perimeter
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

    if intensity:
        new_layer = np.reshape(filters.gaussian(image, sigma), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)

        new_layer = np.reshape(filters.difference_of_gaussians(image,low_sigma= sigma, high_sigma=high_sigma), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)
        """
        new_layer = np.reshape(filters.median(image,selem), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)

        new_layer = np.reshape(filters.rank.minimum(image,selem), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)

        new_layer = np.reshape(filters.rank.maximum(image, selem), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)
        """
    if edges:
        new_layer = np.reshape(filters.sobel(im_blur), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)

    if texture:
        new_layer = np.reshape(filters.hessian(im_blur, mode='constant',), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)

    if membrane:
        mem_layers = membrane_projection(image)
        image_stack = np.append(image_stack, mem_layers, axis=2)

    return image_stack

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
    intensity, edges, texture, membrane: 
    disk_size: Size of the local pixel neighbourhood considered by select 
        segmentation methods.
    parameters: List of dictionaries of Parameters for segmentation methods used in clustering
        The parameters can be inputted manually or use the default .

    Returns
    -------
    generated mask
    """

    image = image.data
    #image = preprocessing.maxabs_scale(image)
    shape = [image.shape[0], image.shape[1], 1]

    image_stack = CreateFeatures(image, intensity, edges, membrane=membrane, sigma = sigma, high_sigma = high_sigma, disk_size = disk_size)

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

def ClusterLearnSeries(image_set, method='KMeans', parameters=[{'kernel': 'gaussian', 'sigma': 1}, 
                                                    {'kernel': 'sobel'},
                                                    {'kernel': 'hessian', 'black ridges': False},
                                                    {'kernel': 'rank mean', 'disk size': 20},
                                                    {'kernel': 'median', 'disk size': 20},
                                                    {'kernel': 'minimum', 'disk size': 20},
                                                    {'kernel': 'maximum', 'disk size': 20},
                                                    {'kernel': 'bilateral', 'disk size': 20},
                                                    {'kernel': 'entropy', 'disk size': 20},
                                                    {'kernel': 'gabor', 'frequency': 100},
                                                    {'kernel': 'gaussian diff', 'low sigma': 1, 'high sigma': None}]):
    """
    Creates masks of sets of images using scikit learn clustering methods.
    
    Parameters
    ----------
    image_set: list of hyperspy signal objects.
        List of Hyperpsy signal object containing nanoparticle images
    method: Clustering algorithm used to generate mask.
    disk_size: Size of the local pixel neighbourhood considered by select 
        segmentation methods.
    parameters: List of dictionaries of Parameters for segmentation methods used in clustering
        The parameters can be input manually in to a dictionary or can be generated
        using param_generator().

    Returns
    -------
    list of generated mask per image
    """

    mask_set = []
    for image in image_set:
        mask_set.append(ClusterLearn(image,method))
    
    return mask_set

def ClusterTrained(image, labels, classifier):

    """
    Creates masks of given images by classifying based on .
    
    Parameters
    ----------
    image : Hyperspy signal object or list of hyperspy signal objects.
    labels : user-labelled mask
    classifier : empty or pretrained classifier to be trained on labelled data

    Returns
    -------
    classified mask, trained classifier
    """
    if labels.all() == False:
        print('start training')
        labels = labels.astype(np.float64)
        shape = image.data.shape
        image = image.data

        features = CreateFeatures(image)
        features = np.rot90(np.rot90(features, axes=(2,0)), axes=(1,2))
        #features are num/x/y

        thin_mask = np.zeros([shape[0],shape[1]])

        c = 1
        for i in range(0,3):
            non_zero = (labels[:,:,i] != 0)
            if np.any(non_zero) == True:
                thin_mask += c*non_zero.astype(int)
                c += 1

        training_data = features[:, thin_mask > 0].T
        #training data is number of labeled pixels by number of features
        training_labels = thin_mask[thin_mask > 0].ravel()
        training_labels = training_labels.astype('int')
        #training labels is labelled pixels in 1D array

        classifier.fit(training_data, training_labels)
        print('finish training')
        #train classifier on  labelled data
        data = features[:, thin_mask == 0].T
        #unlabelled data
        pred_labels = classifier.predict(data)
        #predict labels for rest of image

        output = np.copy(thin_mask)
        output[thin_mask == 0] = pred_labels

        return output, classifier

def ClassifierSegment(classifier, image):
    """
    classifier : sklearn classifier
    image: numpy array of image
    """
    shape = image.shape

    features = ps.CreateFeatures(image)
    features = np.rot90(np.rot90(features, axes=(2,0)), axes=(1,2))
    features = features[:, image == image].T
    mask = clf.predict(features)

    output = np.copy(rescaled_im)
    output[image == image] = mask

    mask_im = np.zeros((shape[0],shape[1],3), dtype = np.uint8)
    for c in range(3):
        mask_im[:,:,c] = 255*(output == (c+1))

    return mask_im
