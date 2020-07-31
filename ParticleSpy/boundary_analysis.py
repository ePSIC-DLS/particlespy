# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:04:53 2020

@author: dell
"""
import numpy as np
from skimage.segmentation import find_boundaries
from scipy import optimize

#   2D image boundary analysis

def eight_neighbour_loop(center, start,boundary_matrix): 
    """center is [x,y], start is [x,y] format"""
    x = center[0]
    y = center[1]
    neighbour_dic = {0: [x-1,y],
                     1: [x-1,y-1],
                     2: [x,y-1],
                     3: [x+1,y-1],
                     4: [x+1,y],
                     5: [x+1,y+1],
                     6: [x,y+1],
                     7: [x-1,y+1]}
    key_list = [0,1,2,3,4,5,6,7]
    start_index = [key for key, val in neighbour_dic.items() if val==start][0]
    
    #new_list, a reorder the key_list
    if start_index == 0:
        new_list = list(range(8))#include start point
    else:
        new_list = list(range(start_index,8)) + list(range(start_index))
    for i in range(8):
        #find the new boundary point and break the loop
        key = new_list[i]
        if neighbour_dic[key] in boundary_matrix:
            new_boundary = neighbour_dic[key]
            new_index = i-1
            #new start is always a background pixel
            new_start = neighbour_dic[new_list[new_index]] 
            break
    return (new_boundary, new_start)


def boundary_follow(img, center=None):
    """
    Find boundaries and clock-wise follow boundary as a sequence list.
    
    Parameters
    ----------
    img: (N, M) ndarray
        Input single objective image. Background should be 0.
    center: [int, int]
        The center of the image box.
    
    Returns
    -------
    boundary_list: list
        A list of boundary pixel coordinates in clock-wise sequence.
        
    Example
    -------
    test_im = skimage.morphology.star((20))
    n = 0
    while n<3:
        test_im = skimage.morphology.binary_erosion(test_im)
        n += 1
    test_im = skimage.filters.gaussian(test_im, sigma=1)
    test_im = skimage.transform.rotate(test_im, 25)
    test_im[test_im!=0] = 1
    plt.imshow(test_im)
    
    #correct boundary pixel sequence
    boundary_list, ang_ls = boundary_follow(test_im)
    b_rr, b_cc = np.asarray(boundary_list).transpose()
    plt.figure(figsize=(10,10))
    plt.scatter(b_rr[:10], np.multiply(-1, b_cc[:10]), c='r')
    plt.scatter(b_rr[10:50], np.multiply(-1, b_cc[10:50]), c='g')
    plt.scatter(b_rr[50:], np.multiply(-1, b_cc[50:]), c='b')
    
    #incorrect boundary pixel sequence
    get_boundary = find_boundaries(test_im,mode='inner')
    b_rr1 = np.where(get_boundary==True)[1] 
    b_cc1 = np.where(get_boundary==True)[0] 
    plt.figure(figsize=(10,10))
    plt.scatter(b_rr1[:10], np.multiply(-1, b_cc1[:10]), c='r')
    plt.scatter(b_rr1[10:50], np.multiply(-1, b_cc1[10:50]), c='g')
    plt.scatter(b_rr1[50:], np.multiply(-1, b_cc1[50:]), c='b')
    """

    if center == None:
        center = [img.shape[0]//2, img.shape[1]//2]
    
    #   find_boundaries only works for bindary image
    img[img!=0] = 1
    get_boundary = find_boundaries(img,mode='inner')
    #   get boundary pixel coordinates, 
    #   np.where results opposite x-y coordinate, [0] is y in plt and [1] is x. 
    peri_x = np.where(get_boundary==True)[1] #It is Y in matplotlib
    peri_y = np.where(get_boundary==True)[0] #It is X in matplotlib
    peri_matrix = np.column_stack((peri_x,peri_y)).tolist()
    #the uppermost, leftmost point will be the start point, b0, at boundary
    b0_y = np.min(peri_y)
    index_= np.min(np.where(peri_y==b0_y))
    b0_x = peri_x[index_]
    b0 = [b0_x, b0_y]
    #c0 is always a background point next to b0
    c0_x = b0_x - 1
    c0_y = b0_y
    c0 = [c0_x, c0_y]
    #Do 8-neighbours clockwise loop to find the first non-background pixel b1, 
    #and the background pixel c1 found before b1, 
    #store b0, b1 for the loop-ending comparision
    b1, c1 = eight_neighbour_loop(b0,c0,peri_matrix) #Do 8-neighbour loop
    b = b1
    c = c1
    #loop again
    bn = b
    boundary_list=[b0]
    while b!=b0 or bn!=b1:
        b = bn
        bn, c = eight_neighbour_loop(b,c,peri_matrix) #Do 8-neighbour loop
        boundary_list.append(b)
    del boundary_list[-1]#the last one is the same of the first 
    
    
    #   get angles    
    v1 = np.subtract([center[0], center[1]-1], center) #12 o'clock direction    
    angle_list = []
    for b in boundary_list:
        v2 = np.subtract(b, center)
        #ang = angle_between(v1, v2)
        ang = np.rad2deg(angle_between(v2, v1)) #coordinates are mess!
        if ang < 0:
            ang = 360 - abs(ang)
        angle_list.append(ang)


    #   sort boundary pixel in angle sequence    
    angle_props = np.column_stack((angle_list, boundary_list)).tolist()
    angle_props.sort()
    angle_list, boundary_xs, boundary_ys = np.asarray(angle_props).transpose()
    boundary_list = np.column_stack((boundary_xs.astype(int),boundary_ys.astype(int))).tolist()
    return boundary_list, angle_list


def boundary_ang(boundary_list, center=None):
    '''
    For each pixel on the boundary, 
    get its angle cooresponding to the center of boundary.
    
    '''
    if center == None:
        center = [img.shape[0]//2, img.shape[1]//2]
        
    v1 = center - [center[0], center[1]+1]
    
    ang_ls = []
    for b in boundary_list:
        v2 = center - b
        ang = angle_between(v1, v2)
        ang_ls.append(ang)
        
    return ang_ls

def boundary_dist(boundary_list, xscale=1, yscale=1):
    '''
    Calculate the distance to centre for each boundary pixel''
    '''
    distance_list = []
    for i in boundary_list:
        d = np.sqrt(((i[0]-centre[0])*xscale)**2+((i[1]-centre[1])*yscale)**2)
        distance_list.append(d)
    return distance_list

def boundary_curvature(im_binary, center=None, segment=5):
    '''
    Get signed boundary curvature of a enclosed particle image.
    The curvature of a given pixel 'p' is defined as 1/r,
    where r is the radius of the least square fitted circle
    by fitting a circle to a list of pixels (i.e. segment of boundary) 
    [p-segment, p-segment+1, ..., p, ..., p+segment-1, p+segment].
    
    Negative curvature means concave while positive means convex. 

    Parameters
    -------
    im_binary : 2d numpy array
        binary image.
    center : tuple or list, optional
        Center coordinates of the enclosed particle image. 
        If None, use whole image center.
    segment: int
        Unit in pix. 

    Returns
    -------
    cur_ls: list
        list of boundary curvature values.

    Example
    -------
    test_im = skimage.morphology.star((20))
    n = 0
    while n<3:
        test_im = skimage.morphology.binary_erosion(test_im)
        n += 1
    test_im = skimage.filters.gaussian(test_im, sigma=1)
    test_im = skimage.transform.rotate(test_im, 25)
    test_im[test_im!=0] = 1
    plt.imshow(test_im)
    
    boundary_list, _ = boundary_follow(test_im, center=None)
    b_x, b_y = np.asarray(boundary_list).transpose()
    cur_ls = boundary_curvature(test_im, center=None, segment=5)
    plt.figure(figsize=(10,10))
    plt.xlim(0,test_im.shape[0])
    plt.ylim(0,test_im.shape[1])
    plt.scatter(b_x, np.subtract(test_im.shape[0],b_y), c=cur_ls, cmap='coolwarm')
    #plt.plot(cur_ls, 'o-')
    '''
    if center == None:
        center = [im_binary.shape[0]//2, im_binary.shape[1]//2]
    
    #   get boundary pixels in clock-wise sequence.
    b, _ = boundary_follow(im_binary)
    b_x, b_y = np.asarray(b).transpose()
    
    cur_ls = []
    for i in range(len(b_x)):
        #   get boundary segment pixels indice
        index_ls = np.arange(i-segment, i+segment+1)
        #   correct end boundary pixel index
        for n, ii in enumerate(index_ls):
            if ii >= len(b_x):
                index_ls[n] = ii-len(b_x)
        #print(index_ls)
        
        points = []
        for ind in index_ls:
            points.append([b_x[ind], b_y[ind]])
            
        c, r, radius, _ = least_squares_circle(points)
        
        #sign curvature
        vec_n = np.subtract(points[len(points)//2], [c, r])
        vec_t = np.subtract(points[len(points)//2], center)
        ang = abs(np.rad2deg(angle_between(vec_n, vec_t)))
        if ang < 90:
            cur = 1/radius
        else:
            cur = -1/radius
        
        cur_ls.append(cur)
    cur_ls = np.asarray(cur_ls)
    return cur_ls

def least_squares_circle(coords):
    '''
    Give a list of points, least square fit a circle 
    ref: https://pypi.org/project/circle-fit/

    Parameters
    ----------
    coords: list or numpy array
        list of 2D point coordinates, number of points > 2.
    
    Returns
    -------
    xc: float
        x-coordinate of solution center
    yc: float
        y-coordinate of solution center
    R: float
        Radius of solution
    residu: float
        MSE of fitting
        
    Example
    -------
    points = [[53, 32], [54, 32], [55, 31], [56, 30], [57, 30], 
              [58, 30],
              [59, 29], [60, 29], [61, 30], [62, 30], [63, 31]]
    c, r, radius, _ = least_squares_circle(points)
    p_x = [po[0] for po in points]
    p_y = [po[1] for po in points]
    plt.figure(figsize=(10,10))
    plt.scatter(p_x, p_y, c='pink')
    plt.scatter(p_x[len(p_x)//2], p_y[len(p_x)//2], c='r')
    plt.scatter(c, r, c='r')
    rr, cc = skimage.draw.circle_perimeter(int(r), int(c), int(radius))
    img = np.zeros((70, 70), dtype=np.uint8)
    img[rr, cc] = 1  
    plt.imshow(img)
    '''
    
    x, y = None, None
    if isinstance(coords, np.ndarray):
        x = coords[:, 0]
        y = coords[:, 1]
    elif isinstance(coords, list):
        x = np.array([point[0] for point in coords])
        y = np.array([point[1] for point in coords])

    x_m = np.mean(x)
    y_m = np.mean(y)
    center_mean = x_m, y_m
    center, _ = optimize.leastsq(f, center_mean, args=(x,y))
    xc, yc = center
    ri = calc_R(x, y, *center)
    r = ri.mean()
    residu = np.sum((ri - r)**2)
    return xc, yc, r, residu

def calc_R(x,y, xc, yc):
    """
    calculate the distance of each 2D points from the center (xc, yc)
    ref: https://pypi.org/project/circle-fit/
    """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f(c, x, y):
    """
    calculate the algebraic distance between the data points
    and the mean circle centered at c=(xc, yc)
    ref: https://pypi.org/project/circle-fit/
    """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ 
    Returns signed angle in radians from vectors 'v1' to 'v2'
    direction clock-wise is positive::

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
    ref: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    minor = np.linalg.det(np.stack((v1_u[-2:], v2_u[-2:])))
    if minor == 0:
        sign = 1
    else:
        sign = -np.sign(minor)
    dot_p = np.dot(v1_u, v2_u)
    dot_p = min(max(dot_p, -1.0), 1.0)
    return sign * np.arccos(dot_p)

def ave_by_deg(ang_ls, cur_ls, interval=10):
    '''
    Average boundary curvature by a given angle interval.
    
    Returned is a boundary curvature list 
    but every curvature is the average of raw curvatures 
    within the given angle interval.
    This ensures at same direction boundary curvatures 
    of two particle with different size (i.e. different boundary pixels) 
    are comparable. 
    
    
    Parameters
    -------
    cur_ls: list or 1d np array
        Raw boundary pixel curvature
    ang_ls: list or 1d np array
        Angles associated with raw boundary pixel curvature
    interval: dividable by 360, unit degree
        Degree interval to average boundary curvatures
        This value cannot be too small. If so at certain angle there will be
        no curvature value.
    '''
    ang_interval_ls = np.arange(0,365,interval)
    ang_ave_ls = []
    for i in range(len(ang_interval_ls)-1):
        blank_ls = []
        for n, ang in enumerate(ang_ls):
            if ang_interval_ls[i]<= ang <ang_interval_ls[i+1]:
                blank_ls.append(cur_ls[n])
        ang_ave_ls.append(np.mean(blank_ls))
   
    #plt.plot(ang_interval_ls[1:], ang_ave_ls, marker='o', label=i)
    #plt.xlim(0,360)
    #plt.xticks(np.arange(0,361,interval))
    return ang_ave_ls