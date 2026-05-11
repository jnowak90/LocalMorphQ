#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 11:51:31 2026

@author: Jacqueline Nowak, JNowak@mpimp-golm.mpg.de
"""

from pathlib import Path
import sys
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import math
import cv2
from PIL import Image, ImageDraw
import skimage
from skimage import exposure, filters, morphology, measure, draw
from skimage.morphology import disk
import matplotlib
import matplotlib.pyplot as plt

# add current script directory to path
pathScript = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathScript))
import GraVisPreprocessing as GP
import GraVisExtraction as GE
import GraVisUtils as utils


###############################################################################
def process_membrane_image(rawImage, manualChanges, folder):
    
    # set default parameters or adjust according to manualChanges
    sigmaGauss, sigmaTube, manualThreshold, sizeObjects, sizeHoles, manualCorrections, imageType, otsuMethod = set_parameters(manualChanges)

    ##### Cleaning and feature highlighting
    ### different processing of pavement cell surface images
    if imageType == 'other':
        tubeImage = utils.tube_filter(rawImage, sigmaTube)
    else: 
        ### standard pipeline for pavement cell membrane images adapted from GraVis (Nowak et al., 2021)        
        tubeImage = highlight_tube_structures(rawImage, sigmaGauss, sigmaTube)
        
    ###### Binarization
    if imageType == 'other':  
        binaryImage = binarize_other_images(tubeImage, otsuMethod)
    else:
        binaryImage = binarize_membrane_images(tubeImage, manualThreshold)

    ##### Remove noise
    if imageType == 'other': 
        cleanedImage = remove_noise_from_other_images(binaryImage, sizeObjects)
    else:
        smallObjects = skimage.morphology.remove_small_objects(binaryImage, sizeObjects)
        cleanedImage = skimage.morphology.remove_small_holes(smallObjects, sizeHoles)

    ##### Skeletonize image
    skeletonImage, branchlessSkeleton, labeledImage, labels = skeletonize_image(cleanedImage, manualCorrections)
    return skeletonImage, branchlessSkeleton, labeledImage , labels  


def process_cytoskeleton_image(rawImage, gauss, sigma , block , small ,factr):
    filteredCytoskeletonMask = extract_cytoskeleton_structures(rawImage, gauss, sigma, block, small, factr)
    filteredCytoskeleton = rawImage * filteredCytoskeletonMask
    return filteredCytoskeleton
 
    
def create_visibility_graphs(labeledCells, labelsCells, branchlessSkeletonCells, skeletonImageCells, resolution, selectedCells, folder):
    junctions = GE.detect_threeway_junctions(skeletonImageCells, branchlessSkeletonCells, labeledCells, 0, folder)
    visibilityGraphs, shapeResultsTable, cellContours = create_single_and_merged_visibilityGraphs_from_labels(labeledCells, labelsCells, branchlessSkeletonCells, skeletonImageCells, junctions, resolution, selectedCells) 
    return visibilityGraphs, cellContours, junctions, shapeResultsTable


def calculate_mask_densities(visibilityGraphs, cellContours, filteredCytoskeleton, labeledCells, replicate, timepoint, junctions, resolution, filament, genotype, maskType):
    
    if maskType == 'perpendicular' or maskType == 'circular':
        outputTable = create_table_dataframe('other')
        for key in visibilityGraphs.keys():
            contour, filamentCellImage, cellImage = extract_cell_and_filament_ROI(key, labeledCells, cellContours, filteredCytoskeleton)
            closenessCentrality, lobePos, neckPos, pos = get_node_properties(visibilityGraphs[key])    
            for node in pos.keys():
                nodeType = define_node_type(lobePos, neckPos, node)
                nodeDensity = calculate_density(filamentCellImage, node, pos, visibilityGraphs[key], 25, 10, 0, maskType)
                dataAppend = [filament, genotype, timepoint, replicate, str(key), node, nodeType, closenessCentrality[node], nodeDensity]
                outputTable.loc[len(outputTable)] = dataAppend
      
    
    elif maskType == 'polygonal':
        outputTable = create_table_dataframe('polygonal')
        for key in visibilityGraphs.keys():
            contour, filamentCellImage, cellImage = extract_cell_and_filament_ROI(key, labeledCells, cellContours, filteredCytoskeleton)
            vertexlist = list(get_lobe_neck_positions(visibilityGraphs[key])[1].values())
            vertexlist = [t[::-1] for t in vertexlist]
            polyIntensity, nonPolyIntensity = calculate_density(filamentCellImage, 0, 0, visibilityGraphs[key], 0, 0, vertexlist, 'polygonal')
            dataAppend = [filament, genotype, timepoint, replicate, str(key), polyIntensity, nonPolyIntensity]
            outputTable.loc[len(outputTable)] = dataAppend       
    
    else:
        outputTable = create_table_dataframe('global')    
        for key in visibilityGraphs.keys():
            contour, filamentCellImage, cellImage = extract_cell_and_filament_ROI(key, labeledCells, cellContours, filteredCytoskeleton)
            cellDensity = np.sum(filamentCellImage > 0) / np.sum(filamentCellImage >= 0)
            if visibilityGraphs[key].number_of_nodes() > 2:
                properties = calculate_shape_properties(visibilityGraphs[key], contour, junctions, cellImage, resolution)
            else:
                properties = np.zeros(13)
            dataAppend = np.concatenate(([filament, genotype, timepoint, replicate, str(key), cellDensity], properties))
            outputTable.loc[len(outputTable)] = dataAppend
    return outputTable 
    

def calculate_global_properties(visibilityGraphs, cellContours, filteredCytoskeleton, labeledCells, replicate, timepoint, junctions, resolution, filament, genotype, maskType):
    outputTable = create_table_dataframe('global')    
    for key in visibilityGraphs.keys():
        contour,  cellImage = extract_cell_and_filament_ROI(key, labeledCells, cellContours, filteredCytoskeleton)
        cellDensity = None
        if visibilityGraphs[key].number_of_nodes() > 2:
            properties = calculate_shape_properties(visibilityGraphs[key], contour, junctions, cellImage, resolution)
        else:
            properties = np.zeros(13)
        dataAppend = np.concatenate(([filament, genotype, timepoint, replicate, str(key), cellDensity], properties))
        outputTable.loc[len(outputTable)] = dataAppend
    return outputTable 

###############################################################################      

def set_parameters(manualChanges):
    params = {'sigmaGauss': 1.0, 'sigmaTube': 3.0, 'manualThreshold': None, 'sizeObjects': 500, 'sizeHoles': 200, 'manualCorrections': None, 'imageType': 'membrane', 'otsuMethod': 'otsu'}
    for key in params.keys():
        if key in manualChanges:
            params[key] = manualChanges[key]
    return params['sigmaGauss'], params['sigmaTube'], params['manualThreshold'], params['sizeObjects'], params['sizeHoles'], params['manualCorrections'], params['imageType'], params['otsuMethod']


def highlight_tube_structures(rawImage, sigmaGauss, sigmaTube):
    # automated removal of image noise and artificial edges
    cleanEdges, changeRescaling, cleanNoise = 0, 0, 0
    backgroundImage, cleanEdges = GP.detect_edges(rawImage, cleanEdges)
    cleanNoise = GP.detect_noisy_image(backgroundImage, cleanNoise)
    zeroPixelImage, changeRescaling = GP.detect_white_pixels(rawImage, changeRescaling)
    cleanImage = GP.remove_artificial_edges(zeroPixelImage, cleanEdges, rawImage)
    # apply image rescaling
    if changeRescaling == 0:
        pLower, pUpper = np.percentile(cleanImage, (2, 98))
    else:
        pLower, pUpper = np.percentile(cleanImage, (2, 90))
    rescaledImage = skimage.exposure.rescale_intensity(cleanImage, (pLower, pUpper))
    # smooth image and highlight tibe-like structures (membrane)
    gaussianImage = skimage.filters.gaussian(rescaledImage, sigmaGauss)
    tubeImage = utils.tube_filter(gaussianImage, sigmaTube)   
    return tubeImage


def binarize_other_images(tubeImage, otsuMethod):
    counts, bins = np.histogram(tubeImage.flatten(), 256, range=(0, 256))
    thresholdHist = np.where(counts == np.max(counts))[0][0]        
    if otsuMethod != 'otsu':
        otsu = skimage.filters.threshold_multiotsu(tubeImage)
        thresholdImage = np.mean([thresholdHist] + list(otsu))
    else:
        otsu = skimage.filters.threshold_otsu(tubeImage)
        thresholdImage = np.mean([thresholdHist, otsu])            
    binaryImage = tubeImage > thresholdImage
    return binaryImage


def binarize_membrane_images(tubeImage, manualThreshold):
    if manualThreshold == None:
        counts, bins = np.histogram(tubeImage.flatten(), 256, range=(0, 256))
        thresholdHist = np.where(counts == np.max(counts))[0][0]
        thresholdOtsu = skimage.filters.threshold_otsu(tubeImage)
        thresholdImage = np.mean([thresholdHist, thresholdOtsu])
        binaryImage = tubeImage > thresholdImage
    else:
        binaryImage = tubeImage > manualThreshold
    return binaryImage


def remove_noise_from_other_images(binaryImage, sizeObjects):
    smallObjects = skimage.morphology.remove_small_objects(binaryImage, sizeObjects)
    preLabeledImage, _ = sp.ndimage.label(smallObjects)
    rp = skimage.measure.regionprops(preLabeledImage)
    areaLargest = np.argmax([rp[idx].area for idx in range(len(rp))])   
    selectedImage = preLabeledImage == areaLargest + 1  
    return selectedImage
 
   
def skeletonize_image(cleanedImage, manualCorrections):
    skeletonImage = skimage.morphology.skeletonize(cleanedImage)
    skeletonImage[:5, :], skeletonImage[-4:, :], skeletonImage[:, :5], skeletonImage[:, -4:] = 0, 0, 0, 0
    correctedSkeletonImage = utils.correct_gaps_in_skeleton(skeletonImage)
    skeletonImage = correctedSkeletonImage
    if manualCorrections != None:
        for x1, y1, x2, y2 in manualCorrections:
            rr, cc = skimage.draw.line(x1 + 5, y1 + 5, x2 + 5, y2 + 5)
            skeletonImage[rr, cc] = 1
        correctedSkeletonImage = skeletonImage.copy()
    branchlessSkeleton = utils.detect_branches(correctedSkeletonImage, mode='remove')
    labeledImage, labels = sp.ndimage.label(~branchlessSkeleton)
    return skeletonImage, branchlessSkeleton, labeledImage, labels


def extract_cytoskeleton_structures(rawImage, gauss, sigma, block, small, factr):
    ### processing pipeline adjusted from CytoSeg (Breuer et al., 2017; Nowak et al., 2020)
    # image smoothing and rescaling
    gaussianCytoskeleton = skimage.filters.gaussian(rawImage, gauss)
    rescaledCytoskeleton = gaussianCytoskeleton.copy()
    rescaledCytoskeleton -= rescaledCytoskeleton.min()
    rescaledCytoskeleton *= 255.0 / rescaledCytoskeleton.max()
    
    # binarize filaments
    tubenessCytoskeleton = utils.tube_filter(rescaledCytoskeleton, sigma)
    thresholdCytoskeleton = skimage.filters.threshold_local(tubenessCytoskeleton, block)
    binarizedCytoskeleton = tubenessCytoskeleton > thresholdCytoskeleton
    
    # skeletonize and clean structures
    skeletonizedCytoskeleton = skimage.morphology.skeletonize(binarizedCytoskeleton > 0)
    cleanedCytoskeleton = skimage.morphology.remove_small_objects(skeletonizedCytoskeleton, small, connectivity=2) > 0
    
    # label structure and filter out low intensity structures
    labeledCytoskeleton, labelsCytoskeleton = sp.ndimage.label(cleanedCytoskeleton, structure=np.ones((3,3)))
    meanIntensity = rescaledCytoskeleton[cleanedCytoskeleton].mean()
    meanIntensityLabels = [np.mean(rescaledCytoskeleton[labeledCytoskeleton == label]) for label in range(1, labelsCytoskeleton + 1)]
    filteredCytoskeleton = 1.0 * cleanedCytoskeleton.copy()
    for label in range(1, labelsCytoskeleton + 1):
        if (meanIntensityLabels[label - 1] < meanIntensity * factr):
            filteredCytoskeleton[labeledCytoskeleton == label] = 0
    filteredCytoskeleton = skimage.morphology.remove_small_objects(filteredCytoskeleton > 0, 2, connectivity=8)    
    return filteredCytoskeleton


def create_single_and_merged_visibilityGraphs_from_labels(labeledCells, labelsCells, branchlessSkeletonCells, skeletonImageCells, junctions, resolution, selectedCells):
    shapeResultsTable = pd.DataFrame(columns=['CellNumber', 'VisGraphNodes', 'VisGraphEdges', 'Lobes', 'Necks', 'Junctions', 'JunctionLobes', 'TrueLobes', 'Area [µm2]', 'Perimeter [µm]', 'RelativeCompleteness', 'Lobeyness', 'Circularity', 'MinorAxis', 'MajorAxis', 'AspectRatio'])
    visibilityGraphs = {}
    cellContours = {}
    if selectedCells != None:
        if len(selectedCells) != 0:
            for label in selectedCells:
                
                ### create merged visibility graphs
                if isinstance(label, tuple):
                    cellImage = merge_cells_from_labels(label, labeledCells)
                    visGraph, cellContour = GE.create_visibility_graph(cellImage * 1, 1, resolution)
                    
                ### create single visibility graphs
                else:
                    cellImage = labeledCells == label + 1
                    visGraph, cellContour = GE.create_visibility_graph(labeledCells, label + 1, resolution)
                    
                ### calculate shape properties
                if visGraph.number_of_nodes() != 0:
                    properties = calculate_shape_properties(visGraph, cellContour, junctions, cellImage, resolution)
                else:
                    properties = np.zeros(13)
                    
                ### append to table
                dataAppend = np.concatenate(([str(label), visGraph.number_of_nodes(), visGraph.number_of_edges()], properties))
                if isinstance(label, tuple):
                    dataAppend = np.array(dataAppend, dtype='object')
                shapeResultsTable.loc[len(shapeResultsTable)] = dataAppend
                visibilityGraphs[label] = visGraph
                cellContours[label] = cellContour  
    else:
        for label in labelsCells:
            cellImage = labeledCells == label + 1
            visGraph, cellContour = GE.create_visibility_graph(labeledCells, label + 1, resolution)
            if visGraph.number_of_nodes() != 0:
                properties = calculate_shape_properties(visGraph, cellContour, junctions, cellImage, resolution)
            else:
                properties = np.zeros(13)
            dataAppend = np.concatenate(([str(label), visGraph.number_of_nodes(), visGraph.number_of_edges()], properties))
            if isinstance(label, tuple):
                dataAppend = np.array(dataAppend, dtype='object')
            shapeResultsTable.loc[len(shapeResultsTable)] = dataAppend
            visibilityGraphs[label] = visGraph
            cellContours[label] = cellContour              
    return visibilityGraphs, shapeResultsTable, cellContours


def merge_cells_from_labels(labels, labeledCells):
    labeledImage = labeledCells.copy() * 0
    cellContours = []
    for label in labels:
        coords = np.transpose(np.where(labeledCells == label + 1))
        for x, y in coords:
            labeledImage[x, y] = 1
        _, contourCell = GE.extract_cell_contour(label + 1, labeledCells)
        cellContours.append(contourCell)
    mergedContourImage = labeledCells.copy() * 0
    for xPos, yPos in np.concatenate((cellContours), axis=0):
        mergedContourImage[xPos, yPos] += 1
    mergedCellImage = (labeledImage * 2) + mergedContourImage
    mergedCellImage = mergedCellImage > 1
    mergedCellImageHoles = sp.ndimage.binary_fill_holes(mergedCellImage, disk(1))
    return(mergedCellImageHoles)
    
    
def calculate_shape_properties(visGraph, cellContour, junctions, cellImage, resolution):
    cellJunctions = GE.find_number_of_cell_junctions(cellContour, junctions)
    lobes, necks = GE.count_lobes_and_necks(visGraph)
    correlatedJunctions = list(np.unique(GE.correlate_junctions_and_lobes(visGraph, lobes, necks, cellJunctions)))
    trueLobes = np.clip(len(lobes) - len(correlatedJunctions), 0, None)
    visGraph = GE.add_lobe_and_neck_property(visGraph, necks, lobes, correlatedJunctions)
    sigma = 1 - GE.compute_graph_complexity(visGraph)    # relative completeness
    area = np.sum(cellImage) * (resolution ** 2)         # cell area
    perimeter = len(cellContour) * resolution            # cell perimeter
    circularity = 4 * np.pi * area / perimeter ** 2      # circularity
    pos = list(nx.get_node_attributes(visGraph, 'pos').values())
    hull = sp.spatial.ConvexHull(pos)
    lobeyness = len(cellContour) / hull.area             # lobeyness
    labeledCellImage, _ = sp.ndimage.label(cellImage)
    rp = skimage.measure.regionprops(labeledCellImage)
    minAxis = rp[0].axis_minor_length * resolution       # minor axis
    maxAxis = rp[0].axis_major_length * resolution       # major axis
    properties = [len(lobes), len(necks), len(cellJunctions), len(correlatedJunctions), trueLobes, area, perimeter, sigma, lobeyness, circularity, minAxis, maxAxis, maxAxis / minAxis]
    return properties
    

def create_cell_cytoskeleton_overlay(labeledCells, filteredCytoskeleton, pathImage, selectedCells):
    # create dilsted boundaries of membranes and cytoskeleton
    cellMembrane = skimage.morphology.binary_dilation(labeledCells == 0)
    filaments = skimage.morphology.binary_dilation(filteredCytoskeleton)
    
    # flatten list of selected cells
    selectedLabels = expand_selected_cells(selectedCells)
    
    # compute cell centers
    labels = np.unique(labeledCells)
    labels = labels[labels > 1]
    centers = sp.ndimage.center_of_mass(labeledCells > 1, labeledCells, labels)
    
    # overlay membranes and filaments        
    fig, ax = plt.subplots(1, 1)
    plt.imshow(cellMembrane, cmap='gray')
    cmapBinary = matplotlib.colors.ListedColormap([[1.0, 1.0, 1.0, 0.0], [0.8, 0.8, 0.8, 1.0]]) #gray
    plt.imshow(filaments==1, cmap=cmapBinary, alpha=0.5)
    
    for label, (y, x) in zip(labels, centers):
        if selectedCells == None:   # all detected cells are used for analysis
            color = 'gold'
        elif len(selectedCells) == 0:   # none of the segmented cells are used for the analysis
            color = 'white'
        else:   # only selected cells are used for analysis
            if label in selectedLabels:
                color ='gold'
            else:
                color = 'white'
        ax.text(x, y, str(label - 1), color=color, fontsize=4, weight='bold')    
    plt.axis('off')
    fig.savefig(pathImage, bbox_inches='tight', dpi=300)
    plt.close()  
    
def create_membrane_segmentation_overlay(pathFile, labeledCells, pathImage, selectedCells):   
    rawImage = skimage.io.imread(pathFile)
    rawImage = np.pad(rawImage, 5)
    cellMembrane = skimage.morphology.binary_dilation(labeledCells == 0)
       
    # flatten list of selected cells
    selectedLabels = expand_selected_cells(selectedCells)
    
    # compute cell centers
    labels = np.unique(labeledCells)
    labels = labels[labels > 1]
    centers = sp.ndimage.center_of_mass(labeledCells > 1, labeledCells, labels)
    
    # overlay membranes and filaments        
    fig, ax = plt.subplots(1, 1)
    plt.imshow(rawImage, cmap='gray')
    cmapBinary = matplotlib.colors.ListedColormap([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]]) #gray
    plt.imshow(cellMembrane==1, cmap=cmapBinary, alpha=0.5)
    
    for label, (y, x) in zip(labels, centers):
        if selectedCells == None:   # all detected cells are used for analysis
            color = 'gold'
        elif len(selectedCells) == 0:   # none of the segmented cells are used for the analysis
            color = 'white'
        else:   # only selected cells are used for analysis
            if label in selectedLabels:
                color ='gold'
            else:
                color = 'white'
        ax.text(x, y, str(label - 1), color=color, fontsize=4, weight='bold')    
    plt.axis('off')
    fig.savefig(pathImage, bbox_inches='tight', dpi=300)
    plt.close()  

def expand_selected_cells(selectedCells):
    if selectedCells == None:
        return None
    expanded = []
    for cell in selectedCells:
        if isinstance(cell, tuple):
            expanded.extend([n + 1 for n in cell])
        else:
            expanded.append(cell + 1)
    return set(expanded)


def create_table_dataframe(maskType):
    if maskType == 'global':
        table = pd.DataFrame(columns=['Filament', 'Genotype', 'Timepoint', 'Replicate', 'CellNumber', 'Density', 'Lobes', 'Necks', 'Junctions', 'JunctionLobes', 'TrueLobes', 'Area [µm2]', 'Perimeter [µm]', 'RelativeCompleteness', 'Lobeyness', 'Circularity', 'MinorAxis', 'MajorAxis', 'AspectRatio'])
    elif maskType == 'polygonal':
        table = pd.DataFrame(columns=['Filament', 'Genotype', 'Timepoint', 'Replicate', 'CellNumber', 'densityInsidePolygon', 'densityOutsidePolygon'])    
    else:
        table = pd.DataFrame(columns=['Filament', 'Genotype', 'Timepoint', 'Replicate', 'CellNumber', 'NodeIdx', 'NodeType', 'ClosenessCentrality', 'Density'])
    return table    
    
    
def extract_cell_and_filament_ROI(key, labeledCells, cellContours, filteredCytoskeleton):
    keys = np.atleast_1d(key) + 1
    cell = np.isin(labeledCells, keys)
    coordinatesCell = np.transpose(np.where(cell == 1))
    contour = cellContours[key]
    if filteredCytoskeleton is not None:
        filamentCellImage = (filteredCytoskeleton.astype(np.int32).copy() * 0) - 2
        for x, y in coordinatesCell:
            filamentCellImage[x, y] = filteredCytoskeleton[x, y] 
        return contour, filamentCellImage, cell
    else:
        return contour, cell


def define_node_type(lobePos, neckPos, node):            
    if node in lobePos.keys():
        nodeType = 'lobe'
    elif node in neckPos.keys():
        nodeType = 'neck'
    else:
        nodeType = 'none'
    return nodeType


def get_node_properties(visibilityGraph):
    centrality = nx.closeness_centrality(visibilityGraph, distance='length')
    lobePos = get_lobe_neck_positions(visibilityGraph)[0]
    neckPos = get_lobe_neck_positions(visibilityGraph)[1]
    pos = nx.get_node_attributes(visibilityGraph, 'pos')
    return centrality, lobePos, neckPos, pos


def get_lobe_neck_positions(graph):
    nodePos = nx.get_node_attributes(graph, 'pos')
    lobeIndices, neckIndices = GE.count_lobes_and_necks(graph)
    lobePos, neckPos = {}, {}
    nodeIndices = list(graph.nodes)
    for nodeIdx in nodeIndices:
        if nodeIdx in lobeIndices:
            lobePos[nodeIdx] = nodePos[nodeIdx]
        if nodeIdx in neckIndices:
            neckPos[nodeIdx] = nodePos[nodeIdx]
    return(lobePos, neckPos)


def calculate_density(filamentCellImage, node, pos, visGraph, size, width, vertexlist, maskType):
    if maskType == 'perpendicular':
        mask = create_perpendicular_mask(filamentCellImage, visGraph, node, size, width)
    if maskType == 'circular':
        mask = create_circular_mask(filamentCellImage, pos[node][1], pos[node][0], size)
    if maskType == 'polygonal':
        mask = create_polygonal_mask(filamentCellImage, vertexlist)
        
    masked_img = filamentCellImage.copy()
    masked_image_32 = masked_img.astype(np.int32)
    masked_image_32[~mask] = -1
    
    if np.sum(mask) == 0:
        return 0 if maskType in ('perpendicular', 'circular') else [0, 0]
    
    denom_inside = np.sum(masked_image_32 >= 0)
    
    if maskType in ('perpendicular', 'circular'):
        intensity = np.sum(masked_image_32 > 0) / denom_inside if denom_inside > 0 else 0
        
    else:
        inside = np.sum(masked_image_32 > 0) / denom_inside if denom_inside > 0 else 0
        
        denom_outside = np.sum(filamentCellImage >= 0) - denom_inside
        outside = (np.sum(filamentCellImage > 0) - np.sum(masked_image_32 > 0)) / denom_outside if denom_outside > 0 else 0
        intensity = [inside, outside]
        
    return intensity


def create_perpendicular_mask(filamentCellImage, visGraph, n, length, width):
    image = filamentCellImage.copy()
    pos = nx.get_node_attributes(visGraph, 'pos')
    y, x = pos[n]
    if n == 0:
        y1, x1 = list(pos.values())[-1]
        y2, x2 = pos[n+1]
    elif n == len(pos.keys())-1:
        y1, x1 = pos[n-1]
        y2, x2 = pos[0]
    else:
        y1, x1 = pos[n-1]
        y2, x2 = pos[n+1]
    if abs(y2 - y1) == 0:
        midY = y1
        midX = x
    elif abs(x2 - x1) == 0:
        midY = y
        midX = x1
    else:
        midY = min(y2, y1) + (abs(y2 - y1)/2)
        midX = min(x2, x1) + (abs(x2 - x1)/2)
    cX, cY, dX, dY = getPerpCoord(x, y, x1, y1, midX, midY, length)
    cv2.line(image, (cX,cY), (dX,dY), (255,0,0), width)
    mask = image == 255
    return mask   
    

def create_circular_mask(image, x, y, radius):
    h, w = image.shape[:2]
    center = (x, y)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask
    

def create_polygonal_mask(image, *vertices):
    width, height = image.shape[::-1]
    img = Image.new(mode='L', size=(width, height), color=0)  
    draw = ImageDraw.Draw(img)
    if len(vertices[0]) > 2:
        for polygon in vertices:
            draw.polygon(polygon, outline=1, fill=1)
        mask = np.array(img)
        mask = mask > 0
    else:
        mask = np.array(img) * 0
    return mask
        

def getPerpCoord(x, y, aX, aY, bX, bY, length):
    vX = bX-aX
    vY = bY-aY
    mag = math.sqrt(vX*vX + vY*vY)
    vX = vX / mag
    vY = vY / mag
    temp = vX
    vX = 0-vY
    vY = temp
    cX = x + vX * length
    cY = y + vY * length
    dX = x - vX * length
    dY = y - vY * length
    return(int(cX), int(cY), int(dX), int(dY))   