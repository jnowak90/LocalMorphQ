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
from . import GraVisUtils as utils

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = matplotlib.patches.Ellipse(xy=center, width=width*0.5, height=width*0.5)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def detect_threeway_junctions(skeletonImage, branchlessSkeleton, labeledImage, plotIntermediate, outputFolder):
    """
    detect threeway junctions of cells in skeletonized image
    """
    #print("...Detect tri-cellular junctions.")
    finalListJunctions = []
    labeledTrackedImage = utils.create_labeled_and_tracked_image(skeletonImage, labeledImage)
    allJunctionsList = utils.detect_crossings_and_endpoints(skeletonImage, mode='crossings', output='list')
    lenX, lenY = branchlessSkeleton.shape
    for xPos, yPos in allJunctionsList:
        if branchlessSkeleton[xPos, yPos] == 1:
            window, winBounds = utils.create_window(labeledTrackedImage, xPos, yPos, 1, 2, 1, 2)
            tracked = np.transpose(np.where(window == 2))
            if len(tracked) == 0:
                finalListJunctions.append([xPos, yPos])
            else:
                newWindow, _ = utils.create_window(labeledTrackedImage, tracked[0][0]+winBounds[0], tracked[0][1]+winBounds[2], 1, 2, 1, 2)
                if np.sum(newWindow == 0) != 0 and np.sum(newWindow == 0) >= np.sum(newWindow >= 3):
                    finalListJunctions.append([xPos, yPos])

    if plotIntermediate == 1:
        plot_cell_junctions(finalListJunctions, branchlessSkeleton, outputFolder)
    return(finalListJunctions)

def plot_cell_junctions(finalListJunctions, branchlessSkeleton, outputFolder):
    """
    plot the threeway junctions on the branchless skeleton
    """
    graph = nx.Graph()
    for idx, (x,y) in enumerate(finalListJunctions):
        graph.add_node(idx, pos=(y, x))
    posGraph = nx.get_node_attributes(graph, 'pos')

    fig,axs = plt.subplots(1, 1, figsize=(10, 10))
    plt.imshow(branchlessSkeleton, cmap='gray')
    nodes=nx.draw_networkx_nodes(graph, posGraph, node_color='red', node_size=30)
    nodes.set_edgecolor('None')
    axs.axes.get_yaxis().set_visible(False)
    axs.axes.get_xaxis().set_visible(False)
    fig.savefig(outputFolder + '/JunctionsOnSkeleton.png', bbox_inches='tight', dpi=300)

def visibility_graphs(labeledImage, labels, resolution, outputFolder):
    """
    create a visibility graph for all cells
    """
    #print("...Create visibility graphs:")
    visGraphsAll = {}
    cellContoursAll = {}
    for label in range(2, labels + 1):
        #print("......Graph " + str(label - 1) + " of " + str(labels - 1))
        visGraph, cellContour = create_visibility_graph(labeledImage, label, resolution)
        visGraphsAll[label - 1] = visGraph
        cellContoursAll[label - 1] = cellContour
        cellContoursPickle = open(outputFolder + '/cellContours.gpickle', 'ab')
        pickle.dump(cellContour, cellContoursPickle)
        cellContoursPickle.close()
    return(visGraphsAll, cellContoursAll)

def create_visibility_graph(labeledImage, label, resolution):
    """
    create visibilit graph from cell contour
    """
    visGraph = nx.Graph()
    pixelDistance = utils.calculate_pixel_distance(resolution)
    contourImage, cellContourOrdered = extract_cell_contour(label, labeledImage)
    if len(cellContourOrdered) != 0:
        pixelsOnContour = utils.interpolate_contour_pixels(cellContourOrdered, pixelDistance)
        if len(pixelsOnContour) >= 4:
            for key in pixelsOnContour:
                visGraph.add_node(key, pos=(pixelsOnContour[key][0], pixelsOnContour[key][1]))
            visGraph = add_edges_to_visGraph(pixelsOnContour, visGraph)
        else:
            cellContourOrdered = []
    return(visGraph, cellContourOrdered)

def extract_cell_contour(label, labeledImage):
    """
    extract the contour of a specified cell
    """
    cellImage = labeledImage == label
    contourImage = utils.invert(cellImage)
    cellContour = utils.find_contour_of_object(cellImage)
    if (0 not in cellContour) and (cellImage.shape[0] not in cellContour[:, 0]) and (cellImage.shape[1] not in cellContour[:, 1]):
        cellContourOrdered = utils.marching_squares(cellContour, cellImage)
        for xPos, yPos in cellContourOrdered:
            contourImage[xPos, yPos] = 1
    else:
        cellContourOrdered = []
        for xPos, yPos in cellContour:
            contourImage[xPos, yPos] = 1
    return(contourImage, cellContourOrdered)

def add_edges_to_visGraph(pixelsOnContour, visGraph):
    """
    add edge to visGraph if the edge between two nodes lies inside the cell (concave)
    """
    cases = ['FFFF0F212', '0FFF0F212', '1FFF0F212', 'F0FF0F212', '00FF0F212', '10FF0F212', 'F1FF0F212']
    Polygon = shapely.geometry.Polygon([[pixelsOnContour[key][1], pixelsOnContour[key][0]] for key in pixelsOnContour])
    Boundary = shapely.geometry.LineString(list(Polygon.exterior.coords))
    combs = itertools.combinations(range(len(pixelsOnContour)), 2)
    for node1, node2 in list(combs):
        line = shapely.geometry.LineString(((pixelsOnContour[node1][1], pixelsOnContour[node1][0]), (pixelsOnContour[node2][1], pixelsOnContour[node2][0])))
        DE9IM = line.relate(Polygon)
        if DE9IM in cases:
            intersection = Boundary.intersection(line)
            if DE9IM == '10FF0F212' and len(intersection.geoms) <= 3:
                visGraph.add_edge(node1, node2, length=utils.euclidean(pixelsOnContour[node1], pixelsOnContour[node2]))
            if DE9IM == 'F1FF0F212' and intersection.geom_type == 'LineString':
                visGraph.add_edge(node1, node2, length=utils.euclidean(pixelsOnContour[node1], pixelsOnContour[node2]))
            if DE9IM in cases[:5]:
                visGraph.add_edge(node1, node2, length=utils.euclidean(pixelsOnContour[node1], pixelsOnContour[node2]))
    return(visGraph)

def add_data_to_table(visGraphsAll, cellContoursAll, labeledImage, labels, junctions, resolution, outputFolder, shapeResultsTable, lobeParameters, plotLobeOutput):
    """
    summarize all results in a table
    """
    for label in range(1, labels):
        cellContour = cellContoursAll[label]
        visGraph = visGraphsAll[label]
        cell = labeledImage == label + 1
        if visGraph.number_of_nodes() != 0:
            cellJunctions = find_number_of_cell_junctions(cellContour, junctions)
            lobes, necks = count_lobes_and_necks(visGraph)
            visGraph = add_lobe_and_neck_property(visGraph, necks, lobes, cellJunctions)
            visGraphsPickle = open(outputFolder + '/visibilityGraphs.gpickle', 'ab')
            pickle.dump(visGraph, visGraphsPickle)
            visGraphsPickle.close()
            correlatedJunctions = correlate_junctions_and_lobes(visGraph, lobes, necks, cellJunctions)
            calculate_lobe_and_neck_properties(label - 1, labeledImage, visGraph, cellContour, cellJunctions, lobes, necks, correlatedJunctions, resolution, lobeParameters, outputFolder, plotLobeOutput)
            sigma = compute_graph_complexity(visGraph)
            area = np.sum(cell) * (resolution ** 2)
            perimeter = len(cellContour) * resolution
            circularity = 4 * np.pi * area / perimeter ** 2
            dataAppend = [label, visGraph.number_of_nodes(), visGraph.number_of_edges(), len(lobes), len(necks), len(cellJunctions), len(correlatedJunctions), sigma, circularity, area, perimeter]
        else:
            visGraphsPickle = open(outputFolder + '/visibilityGraphs.gpickle', 'ab')
            pickle.dump(visGraph, visGraphsPickle)
            visGraphsPickle.close()
            dataAppend = [label, 0, 0, 0, 0, 0, 0, 0, 0, 0, perimeter]
        shapeResultsTable.loc[0] = dataAppend
        if not os.path.isfile(outputFolder + '/ShapeResultsTable.csv'):
            shapeResultsTable.to_csv(outputFolder + '/ShapeResultsTable.csv', mode='a', index=False, sep=';', decimal=',')
        else:
            shapeResultsTable.to_csv(outputFolder + '/ShapeResultsTable.csv', mode='a', index=False, header=False, sep=';', decimal=',')

def add_lobe_and_neck_property(graph, neckIndices, lobeIndices, junctionIndices):
    """
    add node property describing if a node was identified as lobe or neck
    """
    neckLobeProperty = {}
    for nodeName in list(graph.nodes):
        if nodeName in lobeIndices:
            if nodeName in junctionIndices:
                neckLobeProperty[nodeName] = 'Junction'
            else:
                neckLobeProperty[nodeName] = 'Lobe'
        elif nodeName in neckIndices:
            neckLobeProperty[nodeName] = 'Neck'
        else:
            neckLobeProperty[nodeName] = 'None'
    nx.set_node_attributes(graph, neckLobeProperty, name='NodeLabel')
    return(graph)
    
''' original version from Timon
def add_lobe_and_neck_property(graph, neckIndices, lobeIndices):
    """
    add node property describing if a node was identified as lobe or neck
    """
    neckLobeProperty = {}
    for nodeName in list(graph.nodes):
        if nodeName in lobeIndices:
            neckLobeProperty[nodeName] = "Lobe"
        elif nodeName in neckIndices:
            neckLobeProperty[nodeName] = "Neck"
        else:
            neckLobeProperty[nodeName] = "None"
    nx.set_node_attributes(graph, neckLobeProperty, name="LobeNeckNone")
    return(graph)
'''

def find_number_of_cell_junctions(cellContour, junctions):
    """
    find all junctions on a cell contour, allowing for small derivations
    """
    cellJunctions = []
    for index in range(len(junctions)):
        foundPositions = utils.find_index_of_coordinates(junctions[index], cellContour, [0,1,-1], 'coordinates')
        if len(foundPositions) != 0:
            cellJunctions.append(junctions[index])
    return(cellJunctions)

def count_lobes_and_necks(visGraph):
    """
    count the number of lobes and necks according to closeness centrality of visibility graph nodes
    """
    closenessCentrality = nx.closeness_centrality(visGraph, distance='length')
    closenessCentralityArray = np.asarray(list(closenessCentrality.values()))
    lobes, necks = utils.find_local_extrema(closenessCentralityArray)
    return(lobes, necks)

def correlate_junctions_and_lobes(visGraph, lobes, necks, cellJunctions):
    """
    correlate detected lobes and necks with detected tri-cellular junctions
    """
    nodePositions = nx.get_node_attributes(visGraph, 'pos')
    detectedJunctions = []
    positions = np.asarray([nodePositions[idx] for idx in itertools.chain(lobes, necks)])
    if len(positions) != 0:
        for index in range(len(cellJunctions)):
            foundPositions = utils.find_index_of_coordinates(cellJunctions[index], positions, [0, 1, -1, 2, -2, 3, -3], 'coordinates')
            if len(foundPositions) != 0:
                xShift, yShift = foundPositions[0][0], foundPositions[0][1]
                junction = (cellJunctions[index][0] + xShift, cellJunctions[index][1] + yShift)
                key = utils.get_key_from_value(nodePositions, junction)
                detectedJunctions.append(key)
    return(detectedJunctions)

def calculate_lobe_and_neck_properties(key, labeledImage, visGraph, cellContour, cellJunctions, lobes, necks, correlatedJunctions, resolution, lobeParameters, outputFolder, plotLobeOutput):
    """
    calculate the neck width and lobe length for a selected pavement cell and create a graphic output for lobe and neck positions
    """
    pos = nx.get_node_attributes(visGraph, 'pos')
    #protrusionDepthOfCell, protrusionWidthOfCell = calculate_protrusion_depth_and_width(key, labeledImage, visGraph, cellContour, cellJunctions)
    for index in range(len(necks)):
        if index == len(necks) - 1:
            neck1, neck2 = necks[index], necks[0]
            nodes = np.append(np.arange(neck1, visGraph.number_of_nodes(), 1), np.arange(0, neck2, 1))
            lobe = find_lobe_between_necks(lobes, nodes)
        else:
            neck1, neck2 = necks[index], necks[index + 1]
            nodes = np.arange(neck1, neck2 + 1, 1)
            lobe = find_lobe_between_necks(lobes, nodes)
        neckwidth = utils.euclidean(pos[neck1], pos[neck2])
        #protrusionDepth = protrusionDepthOfCell[index]
        #protrusionWidth = protrusionWidthOfCell[index]
        protrusionDepth, protrusionWidth = np.NaN, np.NaN
        if len(lobe) == 0:
            dataLobes = [key, 'no lobe found', None, None, neck1, neck2, None, neckwidth * resolution, protrusionDepth, protrusionWidth]
        elif len(lobe) > 2:
            for matchedLobe in lobe:
                lobelength = calculate_lobe_length(neck1, neck2, matchedLobe, pos)
                dataLobes = [key, matchedLobe, pos[matchedLobe][0], pos[matchedLobe][1], neck1, neck2, lobelength * resolution, neckwidth * resolution, protrusionDepth, protrusionWidth]
        else:
            lobelength = calculate_lobe_length(neck1, neck2, lobe[0], pos)
            dataLobes = [key, lobe[0], pos[lobe[0]][0], pos[lobe[0]][1], neck1, neck2, lobelength * resolution, neckwidth * resolution, protrusionDepth, protrusionWidth]
        lobeParameters.loc[0] = dataLobes
        if not os.path.isfile(outputFolder + '/LobeParameters.csv'):
            lobeParameters.to_csv(outputFolder + '/LobeParameters.csv', mode='a', index=False, sep=';', decimal=',')
        else:
            lobeParameters.to_csv(outputFolder + '/LobeParameters.csv', mode='a', index=False, header=False, sep=';', decimal=',')
    if plotLobeOutput == 1:
        create_visual_output(key, visGraph, cellContour, cellJunctions, lobes, necks, pos, outputFolder)

'''
def calculate_protrusion_depth_and_width(key, labeledImage, visGraph, cellContour, cellJunctions):
    """
    calculate the protrusion depth and width of cells
    """
    extractionDictForTriWayJunction = {"cellLabel": key, "labeledImage": labeledImage}
    protrusionCalculator = CellProtrusionPropertyCalculator(visGraph, cellJunctions, cellContour, extractionDictForTriWayJunction)
    protrusionDepthOfCell = protrusionCalculator.GetProtrusionDepths()
    protrusionWidthOfCell = protrusionCalculator.GetProtrusionWidthAtHalfHeight()
    return(protrusionDepthOfCell, protrusionWidthOfCell)
'''
def find_lobe_between_necks(lobes, nodes):
    """
    find the adjacent necks of a lobe
    """
    matchedLobe = []
    for lobe in lobes:
        if lobe in nodes:
            matchedLobe.append(lobe)
    return(matchedLobe)

def calculate_lobe_length(neck1, neck2, lobe1, pos):
    """
    calculate the length of a lobe based on its adjacent necks
    """
    pos1, pos2, posL = pos[neck1], pos[neck2], pos[lobe1]
    if pos2[0] - pos1[0] == 0:
        basePointX = pos1[0]
        basePointY = posL[1]
    elif pos2[1] - pos1[1] == 0:
        basePointX = pos1[1]
        basePointY = posL[0]
    else:
        slopeBase = (pos2[1] - pos1[1]) / (pos2[0] - pos1[0])
        interceptBase = pos1[1] - slopeBase * pos1[0]
        slopeLobe = -1 / slopeBase
        interceptLobe = posL[1] - slopeLobe * posL[0]
        basePointX = (interceptLobe - interceptBase) / (slopeBase - slopeLobe)
        basePointY = slopeBase * basePointX + interceptBase
    lobeLength = utils.euclidean(posL, (basePointX, basePointY))
    return(lobeLength)

def create_visual_output(key, visGraph, contour, junctions, lobes, necks, pos, outputFolder):
    """
    create a visual output for the positions of detected lobes, necks and tri-cellular junctions
    """
    xmin, xmax, ymin, ymax = np.min(contour[:, 0]), np.max(contour[:, 0]), np.min(contour[:, 1]), np.max(contour[:, 1])
    xminB, yminB = utils.bounds(xmin, 0, xmin - 10), utils.bounds(ymin, 0, ymin - 10)
    contourImage = np.zeros(((xmax + 10) - xminB, (ymax + 10) - yminB))
    for x, y in contour:
        contourImage[x - xminB, y - yminB] = 1

    legend_elements = [matplotlib.patches.Circle(([0], [0]), radius=5, ec='gray', fc='None', lw=2, label='Junction'),
                   matplotlib.patches.Circle(([0], [0]), radius=3, ec='#56b4e9', fc='None', lw=2, label='Lobe'),
                   matplotlib.patches.Circle(([0], [0]), radius=3, ec='#e69f00', fc='None', lw=2, label='Neck')]

    fig, ax = plt.subplots(1, 1)
    plt.imshow(contourImage, cmap='gray_r', interpolation='None')
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Pixel')
    ax.set_title('Cell ' + str(key))
    for junc in junctions:
        circ = plt.Circle((junc[1] - yminB, junc[0] - xminB), radius=5, ec='gray', fc='None', lw=2)
        ax.add_patch(circ)
    for neck in necks:
        posNeck = pos[neck]
        circ = plt.Circle((posNeck[1] - yminB, posNeck[0] - xminB), radius=3, ec='#e69f00', fc='None', lw=2)
        ax.add_patch(circ)
        ax.text(posNeck[1] - yminB + 4, posNeck[0] - xminB + 3, neck, color='#af7900', fontsize=7)
    for lobe in lobes:
        posLobe = pos[lobe]
        circ = plt.Circle((posLobe[1] - yminB, posLobe[0] - xminB), radius=3, ec='#56b4e9', fc='None', lw=2)
        ax.add_patch(circ)
        ax.text(posLobe[1] - yminB + 4, posLobe[0] - xminB + 3, lobe, color='#136492', fontsize=7)
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.4), fancybox=False, shadow=False, ncol=3, fontsize=7, handler_map={matplotlib.patches.Circle: HandlerEllipse()})
    if os.path.exists(outputFolder + '/ResultsLobePositions'):
        fig.savefig(outputFolder + '/ResultsLobePositions/Cell' + str(key) + '_detectedFeatures.png', bbox_inches='tight', dpi=300)
    else:
        fig.savefig(outputFolder + '/Cell' + str(key) + '_detectedFeatures.png', bbox_inches='tight', dpi=300)

def compute_graph_complexity(visGraph):
    """
    compute the complexity of the graph using the relative density of the clique
    """
    edgesCompleteGraph = (visGraph.number_of_nodes() * (visGraph.number_of_nodes() - 1)) * 0.5
    delta = visGraph.number_of_edges() / edgesCompleteGraph
    return(delta)

"""
Timon's code for the calculation of the protrusion depth and width

class CellProtrusionPropertyCalculator(object):

    def __init__(self, visGraph, triWayJunctionPos, cellContour, extractionDictForTriWayJunction=False):
        self.visGraph = visGraph # visibility graph (networkx graph) with 'pos' and 'LobeNeckNone' node attributes
        self.triWayJunctionPos = np.asarray(triWayJunctionPos) # 2D array with each row being the coordinates of one tri-way junction in an ordered fashion
        self.cellContour = cellContour
        self.extractionDictForTriWayJunction = extractionDictForTriWayJunction
        self.calcProtrusionDepthAndWidth()

    def calcProtrusionDepthAndWidth(self):
        self.orderedContour = self.extractOrderedContour()
        self.cellOutlineRing = self.createLinearRingFromContour()
        if self.extractionDictForTriWayJunction:
            labeledImage = self.extractionDictForTriWayJunction["labeledImage"]
            cellLabel = self.extractionDictForTriWayJunction["cellLabel"]
            self.triWayJunctionPos = self.extractTriWayJunctions(cellLabel, self.visGraph, self.triWayJunctionPos, labeledImage)
        self.lobePos, self.selectedLobeKeys = self.extractPositionOfFromVisibilityGraph(undulationType="Lobe")
        triWayJunctionPosList = [(x, y) for x, y in self.triWayJunctionPos]
        self.uniqueJunctions = len(list(set(triWayJunctionPosList)))
        self.protrusionDepths = self.calcShortestDistanceOfPointsToPolygon(self.triWayJunctionPos, self.lobePos, self.uniqueJunctions)
        self.protrusionWidthAtHalfHeight = self.calcWidthAtHalfHeightFor(self.triWayJunctionPos, self.protrusionDepths, self.uniqueJunctions)

    def extractOrderedContour(self):
        orderedContour = self.cellContour
        if np.any(orderedContour[0, :] != orderedContour[-1, :]):
            orderedContour = np.concatenate([orderedContour[:, :], orderedContour[0, :].reshape(1, 2)])
        return(orderedContour)

    def createLinearRingFromContour(self):
        cellOutlineRing = self.createLinearRingFromCoordinates(self.orderedContour)
        assert cellOutlineRing.is_valid, "The cell contour of cell label is not valid. Contour: {}".format(self.orderedContour)
        return(cellOutlineRing)

    def createLinearRingFromCoordinates(self, coordinates):
        return(shapely.geometry.LinearRing(shapely.geometry.asLineString(coordinates)))

    def extractTriWayJunctions(self, cellLabel, visGraph, allTriWayJunctions, labeledImage):
        cellLabelInLabeledImage = cellLabel + 1
        unorderedJunctions = self.extractJunctionOfCell(allTriWayJunctions, cellLabelInLabeledImage, labeledImage)
        unorderedJunctions = self.clipTriWayJunctionToCellOutline(self.cellOutlineRing, unorderedJunctions)
        orderedTriWayJunctions = self.orderJunctions(unorderedJunctions, self.orderedContour)
        return(orderedTriWayJunctions)

    def extractJunctionOfCell(self, triWayJunctions, cellLabel, labeledImage):
        cellImage = np.zeros(labeledImage.shape)
        cellImage[labeledImage == cellLabel] = 1
        cellImage = skimage.morphology.dilation(cellImage, np.ones((8, 8)))
        idx = np.where(cellImage[triWayJunctions[:, 0], triWayJunctions[:, 1]] == 1)[0]
        return(triWayJunctions[idx, :].copy())

    def clipTriWayJunctionToCellOutline(self, cellOutlineRing, junctionCoordinates):
        for i in range(len(junctionCoordinates)):
            currentTriWayJunction = shapely.geometry.Point(junctionCoordinates[i])
            correctedTriWayJunction = cellOutlineRing.interpolate(cellOutlineRing.project(currentTriWayJunction))
            if not self.isCorrectedTriWayJunctionOnOutline(np.asarray(list(cellOutlineRing.coords)), junctionCoordinates[i]):
                junctionCoordinates[i] = self.correctToOutline(np.asarray(list(cellOutlineRing.coords)), list(correctedTriWayJunction.coords))
            else:
                junctionCoordinates[i] = list(correctedTriWayJunction.coords)[0]
        return(junctionCoordinates)

    def isCorrectedTriWayJunctionOnOutline(self, cellOutline, junction):
        return(np.any((cellOutline[:, 0] == junction[0]) & (cellOutline[:, 1] == junction[1])))

    def correctToOutline(self, cellOutline, junction):
        xyDistances = cellOutline - junction
        bestIdx = np.argmin(np.linalg.norm(xyDistances, axis=1))
        return(cellOutline[bestIdx, :])

    def orderJunctions(self, unorderedJunctions, orderedContour, threshold=5):
        orderedJunctions = []
        if unorderedJunctions.shape[0] < 3:
            return unorderedJunctions
        for xyCoor in orderedContour:
            distances = np.linalg.norm(unorderedJunctions - xyCoor, axis=1)
            idx = np.where(distances < threshold)[0]
            if len(idx) == 1:
                orderedJunctions.append(unorderedJunctions[idx, :].tolist()[0])
                unorderedJunctions = np.delete(unorderedJunctions, idx, axis=0)
                if len(unorderedJunctions) == 0:
                    break
        orderedJunctions = np.asarray(orderedJunctions)
        return(orderedJunctions)

    def extractPositionOfFromVisibilityGraph(self, undulationType="Lobe"):
        selectedNodeKeys = []
        coordinates = []
        nodeUndulationAttribute = nx.get_node_attributes(self.visGraph, 'LobeNeckNone')
        allCoordinates = nx.get_node_attributes(self.visGraph, 'pos')
        for nodeKey, undulationProperty in nodeUndulationAttribute.items():
            if undulationProperty == undulationType:
                coordinates.append(allCoordinates[nodeKey])
                selectedNodeKeys.append(nodeKey)
        coordinates = np.asarray(coordinates)
        return(coordinates, selectedNodeKeys)

    def calcShortestDistanceOfPointsToPolygon(self, polygonVertices, points, uniqueJunctions):
        if uniqueJunctions > 2:
            cellPolygon = shapely.geometry.asPolygon(polygonVertices)
            cellPolygonRing = self.createLinearRingFromCoordinates(polygonVertices)
            nrOfUndulations = points.shape[0]
            undulationDepths = np.zeros(nrOfUndulations)
            for i in range(nrOfUndulations):
                undulationPoint = shapely.geometry.Point(points[i, :])
                isInsideCell = cellPolygon.contains(undulationPoint)
                undulationDepths[i] = cellPolygonRing.distance(undulationPoint)
                if isInsideCell:
                    undulationDepths[i] *= -1
        else:
            undulationDepths = [np.NaN] * len(points)
        return(undulationDepths)

    def calcWidthAtHalfHeightFor(self, polygonVertices, depthOfCell, uniqueJunctions):
        nodeCoordinates = nx.get_node_attributes(self.visGraph, 'pos')
        if uniqueJunctions > 2:
            cellPolygonRing = self.createLinearRingFromCoordinates(polygonVertices)
            widths = np.zeros(len(self.selectedLobeKeys))
            for i in range(len(self.selectedLobeKeys)):
                currentLobeNodeKey = self.selectedLobeKeys[i]
                undulationPoint = shapely.geometry.Point(nodeCoordinates[currentLobeNodeKey])
                if depthOfCell[i] != 0:
                    widths[i] = self.calcWidthAtHalfHeight(undulationPoint, cellPolygonRing)
                else:
                    widths[i] = np.NaN
        else:
            widths = [np.NaN] * len(self.selectedLobeKeys)
        return(widths)

    def calcWidthAtHalfHeight(self, undulationPoint, cellPolygonRing,
                              lengthOfParallelSegment=400, recursiveCallIncreasingSegmentLength=False,
                              recursiveCallNr=0):
        auxiliaryLineAtHalfHeight = self.createParallelToPolygonLineAtHalfHeight(undulationPoint,
                                            cellPolygonRing, lengthOfParallelSegment=lengthOfParallelSegment)
        if auxiliaryLineAtHalfHeight != None:
            intersects = self.cellOutlineRing.intersection(auxiliaryLineAtHalfHeight)
            intersects = self.correctPotentialLineString(intersects, undulationPoint)
            widthAtHalfHeight = np.NaN
            if type(intersects) != type(shapely.geometry.Point()):
                if len(intersects) > 2:
                    lotOfUDToPolygon = shapely.geometry.LineString([undulationPoint, cellPolygonRing.interpolate(cellPolygonRing.project(undulationPoint))])
                    intersects = self.reduceToImportantentIntersections(intersects, lotOfUDToPolygon)
                    undulationWidthAtHalfHeightLine = shapely.geometry.LineString(intersects)
                    widthAtHalfHeight = undulationWidthAtHalfHeightLine.length
                if len(intersects) == 2:
                    undulationWidthAtHalfHeightLine = shapely.geometry.LineString(intersects)
                    widthAtHalfHeight = undulationWidthAtHalfHeightLine.length
        else:
            widthAtHalfHeight = np.NaN
        return(widthAtHalfHeight)

    def createParallelToPolygonLineAtHalfHeight(self, undulationPoint, cellPolygonRing,
                                                lengthOfParallelSegment=200):
        halfUDPoint = self.calcPointAtHalfUndulationDepth(undulationPoint, cellPolygonRing)
        parallelSegmentVector = self.calcVectorOfClosestPolygonSegment(undulationPoint, cellPolygonRing)
        if np.linalg.norm(parallelSegmentVector) != 0:
            factorForVector = lengthOfParallelSegment / (2 * np.linalg.norm(parallelSegmentVector))
            startPointOfParallelSegment = np.asarray(halfUDPoint.coords)[0] + factorForVector * parallelSegmentVector
            endPointOfParallelSegment = np.asarray(halfUDPoint.coords)[0] - factorForVector * parallelSegmentVector
            auxiliaryLineAtHalfHeight = shapely.geometry.LineString([startPointOfParallelSegment, endPointOfParallelSegment])
            return(auxiliaryLineAtHalfHeight)
        else:
            return(None)

    def calcPointAtHalfUndulationDepth(self, undulationPoint, cellPolygonRing):
        pointOnRing = cellPolygonRing.interpolate(cellPolygonRing.project(undulationPoint))
        undulationDepthLine = shapely.geometry.LineString([undulationPoint, pointOnRing])
        distance = undulationDepthLine.length
        halfUDPoint = undulationDepthLine.interpolate(distance / 2)
        return(halfUDPoint)

    def calcVectorOfClosestPolygonSegment(self, undulationPoint, cellPolygonRing):
        closestPolygonSegment = self.findClosestLineSegment(cellPolygonRing, undulationPoint)
        startOfSegement, endOfSegment = list(closestPolygonSegment.coords)
        closestSegmentVector = np.asarray(endOfSegment) - np.asarray(startOfSegement)
        return(closestSegmentVector)

    def findClosestLineSegment(self, linearRing, point):
        oldPoint = None
        splitRingSegments = []
        for newPoint in linearRing.coords:
            if oldPoint:
                splitRingSegments.append(shapely.geometry.LineString([newPoint, oldPoint]))
            oldPoint = newPoint
        argMin = None
        minDistance = np.inf
        for i in range(len(splitRingSegments)):
            distance = splitRingSegments[i].distance(point)
            if distance < minDistance:
                minDistance = distance
                argMin = i
        return(splitRingSegments[argMin])

    def correctPotentialLineString(self, potentialLineStrings, referencePoint):
        if type(potentialLineStrings) != type(shapely.geometry.Point()):
            if type(potentialLineStrings) == type(shapely.geometry.LineString()):
                potentialLineStrings = list(potentialLineStrings.coords)
            else:
                potentialLineStrings = list(potentialLineStrings)
                containsLine = False
                for i in range(len(potentialLineStrings)):
                    if type(potentialLineStrings[i]) == type(shapely.geometry.LineString()):
                        minDistances = np.inf
                        closestPointToReferencePoint = None
                        for point in potentialLineStrings[i].coords:
                            point = shapely.geometry.Point(point)
                            d = referencePoint.distance(point)
                            if d < minDistances:
                                closestPointToReferencePoint = point
                        potentialLineStrings[i] = closestPointToReferencePoint
            potentialLineStrings = shapely.geometry.MultiPoint(potentialLineStrings)
        return(potentialLineStrings)

    def reduceToImportantentIntersections(self, intersects, lotOfUDToPolygon):
        intersectionCoordinates = list(intersects)
        direction = 0
        interlinkedIntersects = shapely.geometry.LineString(intersectionCoordinates)
        directionChecked = 0
        while interlinkedIntersects.intersects(lotOfUDToPolygon) or directionChecked < 2:
            if len(intersectionCoordinates) == 2:
                break
            removedIntersection = intersectionCoordinates.pop(direction)
            interlinkedIntersects = shapely.geometry.LineString(intersectionCoordinates)
            if not interlinkedIntersects.intersects(lotOfUDToPolygon):
                intersectionCoordinates.insert(direction, removedIntersection)
                direction = -1
                directionChecked += 1
        return(intersectionCoordinates)

    def GetProtrusionDepths(self):
        return(self.protrusionDepths)

    def GetProtrusionWidthAtHalfHeight(self):
        return(self.protrusionWidthAtHalfHeight)
"""
