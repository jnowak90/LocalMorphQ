#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nowakj
"""
from pathlib import Path
import os
import sys
import numpy as np
from numpy import linalg
import glob
import skimage
from skimage import io, color, morphology, filters, transform, measure, exposure, restoration, feature, draw
from skimage.morphology import disk
from skimage.util import dtype
import scipy as sp
from scipy import ndimage, stats, spatial, cluster
import itertools
import pandas as pd
import networkx as nx
from packaging.version import Version
import math
import shapely
from shapely import geometry
import pickle
import time
import sklearn
from sklearn import decomposition
import read_roi
import matplotlib
from matplotlib.legend_handler import HandlerPatch
import matplotlib.pyplot as plt

# add current script directory to path
pathScript = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathScript))
import GraVisUtils as utils

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = matplotlib.patches.Ellipse(xy=center, width=width*0.5, height=width*0.5)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def import_image(filename):
    """
    import image from filename and convert to 8-bit 2D image
    """
    #print("...Load image and convert to grayscale")
    fileType = filename.split('.')[-1]
    if fileType in ['tif', 'TIF', 'tiff', 'TIFF']:
        rawImage = skimage.io.imread(filename, plugin='tifffile')
    else:
        rawImage = skimage.io.imread(filename)
    if len(rawImage.shape) > 2 and rawImage.shape[2] > 2:
        rawImage = skimage.color.rgb2gray(rawImage)
    if rawImage.dtype != 'uint8':
        rawImage = skimage.util.img_as_ubyte(skimage.exposure.rescale_intensity(rawImage))
    return(rawImage)

def detect_edges(rawImage, cleanEdges):
    """
    use Sobel filter and probabilistic Hough lines to detect artificial edges from image stitching
    """
    backgroundImage = rawImage > 25
    backgroundImage = skimage.morphology.remove_small_objects(backgroundImage, 500)
    backgroundImage = skimage.morphology.remove_small_holes(backgroundImage, 500)
    sobelImage = skimage.filters.sobel(backgroundImage)
    houghLines = skimage.transform.probabilistic_hough_line(sobelImage, threshold=50, line_length=100, line_gap=0)
    if len(houghLines) != 0:
        cleanEdges = 1
        #print("...Artificial edges were detected in the image and will be removed.")
    return(backgroundImage, cleanEdges)

def detect_noisy_image(backgroundImage, cleanNoise):
    """
    calculate noise of image by calculating the ratio of background pixels and image resize
    """
    noiseGrade = np.sum(backgroundImage) / backgroundImage.size
    if noiseGrade > 0.7:
        cleanNoise = 1
        #print("...The image is very noisy and will be cleaned.")
    return(cleanNoise)

def detect_white_pixels(rawImage, changeRescaling):
    """
    calculate the ratio of 0-pixels in the image to infer if intensity rescaling is neccessary
    """
    zeroPixelImage = rawImage == 0
    zeroPixelImage = skimage.morphology.remove_small_objects(zeroPixelImage, 500)
    zeroPixelRatio = np.sum(zeroPixelImage) / zeroPixelImage.size
    if zeroPixelRatio > 0.6:
        changeRescaling = 1
        #print("...The image rescaling was changed.")
    return(zeroPixelImage, changeRescaling)

def remove_artificial_edges(zeroPixelImage, cleanEdges, rawImage):
    """
    remove artificial edges from the image
    """
    if cleanEdges == 1:
        cleanImage = rawImage.copy()
        zeroPixelLabelImage, zeroPixelLabels = sp.ndimage.label(zeroPixelImage)
        zeroPixelBorders = utils.keep_labels_on_border(zeroPixelLabelImage)
        zeroPixelBordersLabelImage, _ = sp.ndimage.label(zeroPixelBorders)
        zeroPixelBordersRP = skimage.measure.regionprops(zeroPixelBordersLabelImage)
        edgeContours = detect_edge_contours(zeroPixelBordersRP, zeroPixelBordersLabelImage)
        for idx in range(len(edgeContours)):
            contourRescaled = rescale_contour_to_original(edgeContours[idx], rawImage)
            cleanImage = remove_pixels_in_periphery(contourRescaled, cleanImage)
    else:
        cleanImage = rawImage.copy()
    return(cleanImage)

def detect_edge_contours(regionProps, labeledImage):
    """
    find the contour of all artificial edges
    """
    edgeContourList = []
    for idx in range(len(regionProps)):
        if regionProps[idx].euler_number == 1:
            imageLabel = labeledImage == idx + 1
            edgeContour = utils.find_edge_contour(imageLabel)
            edgeContourList.append(edgeContour)
    return(edgeContourList)

def rescale_contour_to_original(contour, rawImage):
    """
    rescale contours back to original image shape
    """
    contourRescaled = contour-2
    for idx, (xPos,yPos) in reversed(list(enumerate(contourRescaled))):
        if xPos <= 1 or xPos >= rawImage.shape[0]-1 or yPos <= 1 or yPos >= rawImage.shape[1]-1:
            contourRescaled = np.delete(contourRescaled, idx, 0)
    return(contourRescaled)

def remove_pixels_in_periphery(contourRescaled, cleanImage):
    """
    remove pixels in periphery of contour
    """
    contourPixels = {}
    listOfCorrectedPixels = []
    xLen, yLen = cleanImage.shape[0] - 1, cleanImage.shape[1] - 1
    for idx, (xPos, yPos) in reversed(list(enumerate(contourRescaled))):
        window = cleanImage[xPos-1:xPos+2, yPos-1:yPos+2]
        orientContourPixel = utils.contour_orientation(window)
        for pxl in range(len(orientContourPixel)):
            if pxl < 2:
                contourPixels[(xPos, yPos)] = [orientContourPixel[pxl]]
                listOfCorrectedPixels = utils.measure_intensity_along_contour(cleanImage, xPos, yPos, orientContourPixel[pxl], listOfCorrectedPixels)
    listOfCorrectedPixelsSorted = sorted(listOfCorrectedPixels)
    correctEdges = calculate_consecutive_difference(listOfCorrectedPixelsSorted)

    for start, end in correctEdges:
        edgePixels = listOfCorrectedPixelsSorted[int(start):int(end)]
        for xPos, yPos in edgePixels:
            pixelOrientation = contourPixels[(xPos, yPos)]
            for orient in pixelOrientation:
                xmin, xmax, ymin, ymax = define_pixel_periphery(xPos, yPos, orient, xLen, yLen)
                cleanImage[xmin:xmax, ymin:ymax] = 0
            del contourPixels[xPos, yPos]

    for xPos, yPos in contourPixels:
        xmin, xmax = utils.bounds(xPos-5, 0, xLen), utils.bounds(xPos+6, 0, xLen)
        ymin, ymax = utils.bounds(yPos-5, 0, yLen), utils.bounds(yPos+6, 0, yLen)
        cleanImage[xmin:xmax, ymin:ymax] = 0

    cleanImage[:10, :] = 0
    cleanImage[-9:, :] = 0
    cleanImage[:, :10] = 0
    cleanImage[:, -9:] = 0
    return(cleanImage)

def calculate_consecutive_difference(sequence):
    """
    calculate difference for all consecutive point coordinates in an array and return a range of at least 50 continous, consecutive points (artificial edge)
    """
    correctEdges = []
    consecutiveDifference = np.diff(sequence, axis=0)
    differenceDistances = np.sqrt((consecutiveDifference ** 2).sum(axis=1))
    groupedDifferences = np.array(list([value, len(list(counts))] for value, counts in itertools.groupby(differenceDistances)))
    consecutiveEdges = np.where(groupedDifferences[:,1] > 50)
    for idx in consecutiveEdges[0]:
        correctEdges.append([np.sum(groupedDifferences[:idx,1]),np.sum(groupedDifferences[:idx+1,1])])
    return(correctEdges)

def define_pixel_periphery(x, y, orientation, xLen, yLen):
    """
    find the direction of the pixel periphery to be cleaned
    """
    if orientation == 'right':
        xmin, xmax = utils.bounds(x-1, 0, xLen), utils.bounds(x+2, 0, xLen)
        ymin, ymax = utils.bounds(y-1, 0, yLen), utils.bounds(y+21, 0, yLen)
    if orientation == 'top':
        xmin, xmax = utils.bounds(x-20, 0, xLen), utils.bounds(x+1, 0, xLen)
        ymin, ymax = utils.bounds(y-1, 0, yLen), utils.bounds(y+2, 0, yLen)
    if orientation == 'left':
        xmin, xmax = utils.bounds(x-1, 0, xLen), utils.bounds(x+2, 0, xLen)
        ymin, ymax = utils.bounds(y-20, 0, yLen), utils.bounds(y+1, 0, yLen)
    if orientation == 'bottom':
        xmin, xmax = utils.bounds(x-1, 0, xLen), utils.bounds(x+21, 0, xLen)
        ymin, ymax = utils.bounds(y-1, 0, yLen), utils.bounds(y+2, 0, yLen)
    return(xmin, xmax, ymin, ymax)

def skeletonize_image(cleanImage, cleanNoise, changeRescaling):
    """
    skeletonize cleaned image according to whether the images is noisy or not
    """
    #print("...Image is skeletonized.")
    if cleanNoise == 0:
        branchlessSkeleton, binaryImage, skeletonImage = create_skeletonized_image(cleanImage, changeRescaling)
    else:
        branchlessSkeleton, binaryImage, skeletonImage = remove_noise_from_image(cleanImage, changeRescaling)
    return(branchlessSkeleton, binaryImage, skeletonImage)

def create_skeletonized_image(cleanImage, changeRescaling):
    """
    create the rescaled, filtered, binarized and skeletonized image
    """
    if changeRescaling == 0:
        pLower, pUpper = np.percentile(cleanImage, (2, 98))
    else:
        pLower, pUpper = np.percentile(cleanImage, (2, 90))
    rescaledImage = skimage.exposure.rescale_intensity(cleanImage, (pLower, pUpper))
    gaussianImage = skimage.filters.gaussian(rescaledImage, sigma=3)
    tubeImage = utils.tube_filter(gaussianImage, sigma=3)
    binaryImage = binarize_image(tubeImage)
    smallObjects = skimage.morphology.remove_small_objects(binaryImage, 500)
    smallHoles = skimage.morphology.remove_small_holes(smallObjects, 200)
    skeletonImage = skimage.morphology.skeletonize(smallHoles)
    # remove skeleton that touches the image border
    skeletonImage[:5, :], skeletonImage[-4:, :], skeletonImage[:, :5], skeletonImage[:, -4:] = 0, 0, 0, 0
    correctedSkeletonImage = utils.correct_gaps_in_skeleton(skeletonImage)
    skeletonImage = correctedSkeletonImage
    branchlessSkeleton = utils.detect_branches(correctedSkeletonImage, mode='remove')
    return(branchlessSkeleton, binaryImage, skeletonImage)

def binarize_image(tubeImage):
    """
    binarize image with mean of Otsu threshold and intensity histogram
    """
    #print("...Image is binarized.")
    counts, bins = np.histogram(tubeImage.flatten(), 256, range=(0, 256))
    thresholdHist = np.where(counts == np.max(counts))[0][0]
    thresholdOtsu = skimage.filters.threshold_otsu(tubeImage)
    thresholdImage = np.mean([thresholdHist, thresholdOtsu])
    binaryImage = tubeImage > thresholdImage
    return(binaryImage)

def remove_noise_from_image(cleanImage, changeRescaling):
    """
    remove noise from image and then skeletonize
    """
    denoisedImage = skimage.restoration.denoise_tv_chambolle(cleanImage)
    tophatImage = skimage.morphology.white_tophat(denoisedImage, selem=disk(3))
    adapthistImage = skimage.exposure.equalize_adapthist(tophatImage, clip_limit=0.1)
    otsuImage = skimage.filters.threshold_otsu(adapthistImage)
    binaryImage = adapthistImage > otsuImage
    smallObjects = skimage.morphology.remove_small_objects(binaryImage, 200)
    intermediateSkeletonImage = skimage.morphology.skeletonize(smallObjects)
    continueNoiseCleaning = evaluate_skeleton(intermediateSkeletonImage)
    if continueNoiseCleaning == True:
        correctedSkeletonImage = utils.correct_gaps_in_skeleton(intermediateSkeletonImage)
        intermediateBranchlessSkeleton = utils.detect_branches(correctedSkeletonImage, mode='remove')
        smallHoles = skimage.morphology.remove_small_holes(intermediateBranchlessSkeleton, 100)
        skeletonImage = skimage.morphology.skeletonize(smallHoles)
        skeletonImage = correctedSkeletonImage
        branchlessSkeleton = utils.detect_branches(skeletonImage, mode='remove')
    else:
        branchlessSkeleton, binaryImage, skeletonImage = create_skeletonized_image(cleanImage, changeRescaling)
    return(branchlessSkeleton, binaryImage, skeletonImage)

def evaluate_skeleton(skeletonImage):
    """
    evaluate if denoised image renders usable skeleton
    """
    usableSkeleton = True
    labeledSkeleton, labels = sp.ndimage.label(skeletonImage, np.ones((3,3)))
    if labels > 5:
        usableSkeleton = False
    return(usableSkeleton)

def plot_labeled_image(labeledImage, labels, pathToFolder):
    """
    plot the labeled image with labels
    """
    textPositions = []
    textString = []
    for idx in range(2, labels + 1):
        label = labeledImage == idx
        cmx, cmy = sp.ndimage.center_of_mass(label)
        textPositions.append([cmx, cmy])
        textString.append(str(idx-1))
    labelThreshold = int(80*labels/100)

    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    plt.imshow(labeledImage, cmap='viridis')
    axs.axes.get_yaxis().set_visible(False)
    axs.axes.get_xaxis().set_visible(False)
    for idx in range(len(textString)):
        if int(textString[idx]) >= labelThreshold:
            plt.text(textPositions[idx][1], textPositions[idx][0], textString[idx], fontsize=8, color='black')
        else:
            plt.text(textPositions[idx][1], textPositions[idx][0], textString[idx], fontsize=8, color='white')
    fig.savefig(pathToFolder + '/LabeledPavementCells.png', bbox_inches='tight', dpi=300)
