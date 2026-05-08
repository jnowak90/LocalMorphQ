#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 23:06:02 2026

@author: Jacqueline Nowak, JNowak@mpimp-golm.mpg.de
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import scipy as sp
import skimage
from skimage.morphology import disk
import statsmodels.formula.api as smf
import sklearn
import joypy
import ast
import pingouin as pg
import matplotlib
import matplotlib.pyplot as plt

# add current script directory to path
pathScript = Path(__file__).resolve().parent
sys.path.append(str(pathScript))
import GraVisExtraction as GE

###############################################################################

# =========================
# Figure 4
#  Time-resolved analysis of pavement cell shape complexity in developing leaves of wild-type
#  and act2-1 act7-1 plants.
# =========================

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to dataset')
    args = parser.parse_args()
    
    pathRoot = Path(args.data)
    pathLeaves = pathRoot / 'Leaves'
    
    pathPlots = pathRoot / 'Plots'
    pathPlots.mkdir(parents=True, exist_ok=True)

    ###########################################################################
    
    ### create heatmaps of relative completeness in leaves for WT and act2-1act7-1
    cells = {'WT': {12: [13, 16, 18, 19, 20, 22, 23, 26, 27, 29, 30, 32, 33, 36, 37, 39, 43, 45], 
              24: [7, 13, 15, 17, 18, 20, 21, 22, 24, 25, 26, 27, 30, 31, 33, 35, 37, 39, 40, 43, 45, 46, 48, 49], 
              36: [1, 6, 8, 9, 10, 11, 12, 13, 15, (17, 18), 19, 20, 21, 23, 24, 25, 26, 28, 29, 30, 31, 33, 37, 38, 39, 40], 
              48: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, (16, 18), (17, 20, 19), 21, 22, (23, 25), 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]}, 
               'M': {0: [8, 10, 11, 13, 14, 15, 17, 19, 20, 24, 25, 26, 27, 28, 29, 30], 
              12: [7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], 
              24: [2, (3, 6), 7, 8, 10, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 28, 30, 31, 32, 33, 34, 35, (36, 37)], 
              36: [1, 3, 4, (5, 7), 9, 10, (11, 12), 13, 14, (15, 17), 16, 18, 19, 21, 22, 23, 25, 28, 29, (31, 32), 33, 34, 36, 37, 38, (39, 41)]}}
    
    topR = plt.get_cmap('Oranges', 256)
    newcolorsR = np.vstack((topR(np.linspace(0, 1, 256))))
    newcolorsR[0] = np.array([1.0, 1.0, 1.0, 0.0])
    cmapCellsR = matplotlib.colors.ListedColormap(newcolorsR, name='Oranges')
    cmapBinary = matplotlib.colors.ListedColormap([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    
    for gt in cells.keys():
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        for idx, tp in enumerate(cells[gt].keys()):
            if gt == 'WT':
                resolution = 1 / 10.9051
                name = 'Figure4A_Leaves_ComplexityHeatmap_WT.png'
            else:
                resolution = 1 / 6.5445
                name = 'Figure4B_Leaves_ComplexityHeatmap_Mutant.png'
            imageDilated, labeledImageSelected, labelsSelected = extract_selected_cells(cells, gt, tp, pathLeaves)    
            completeness = extract_selected_cell_properties(labeledImageSelected, labelsSelected, resolution, 'completeness')

            imageColored_C = labeledImageSelected.copy() * 0
            imageColored_C = imageColored_C.astype(float)
            imageColored_C[0, 0] = 1.0
            for label in range(len(completeness)):
                coord = np.transpose(np.where(labeledImageSelected == label + 2))
                for x, y in coord:
                    imageColored_C[x, y] = completeness[label]
            im = ax[idx].imshow(imageColored_C, cmap=cmapCellsR, zorder=0)
            ax[idx].imshow(imageDilated==1, cmap=cmapBinary, alpha=1, zorder=2)
            ax[idx].axis('off')
        cbar = fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
        cbar.set_label('PC shape complexity', rotation=90, labelpad=20)
        plt.tight_layout()
        fig.savefig(pathPlots / name, bbox_inches='tight', dpi=300) 
        plt.close()

    
    ###########################################################################
    trackedLabels = pd.read_csv(pathLeaves / 'TrackedLabels.csv')
    globalDensities = pd.read_csv(pathLeaves / 'globalDensities.csv')
    timepoints = ['0', '12', '24', '36', '48']

    dataTrackedCells = pd.DataFrame(columns=['Genotype', 'Filament', 'Replicate', 'Property', '0h', '12h', '24h', '36h'])

    # extract properties for tracked cells from global properties
    for idx in range(len(trackedLabels)):
        properties = globalDensities[(globalDensities['Filament'] == trackedLabels['Filament'][idx]) & (globalDensities['Genotype'] == trackedLabels['Genotype'][idx]) & (globalDensities['Replicate'] == trackedLabels['Replicate'][idx])]

        if trackedLabels['Genotype'][idx] == 'WT':
            selectedCells = [ast.literal_eval(n) if isinstance(n, str) else int(n) for n in trackedLabels.iloc[idx, 4:]]
            timepoints = [12, 24, 36, 48]
        else:
            selectedCells = [ast.literal_eval(n) if isinstance(n, str) else int(n) for n in trackedLabels.iloc[idx, 3:-1]]
            timepoints = [0, 12, 24, 36]
        
        dataArea, dataCompleteness, dataLobes = [], [], []
        for jdx, cell in enumerate(selectedCells):
            tp = timepoints[jdx]
            propertiesCell = properties[(properties['CellNumber'] == str(cell)) & (properties['Timepoint'] == tp)].reset_index(drop=True)
            if propertiesCell.shape[0] != 0:
                lobes, area, completeness = propertiesCell.iloc[0, [10, 11, 13]]
                dataArea.append(area)
                dataCompleteness.append(completeness)
                dataLobes.append(lobes)
                        
        dataAppend = [trackedLabels['Genotype'][idx], trackedLabels['Filament'][idx], trackedLabels['Replicate'][idx], 'Area']
        dataAppend.extend(dataArea)
        dataTrackedCells.loc[len(dataTrackedCells)] = dataAppend
        
        dataAppend = [trackedLabels['Genotype'][idx], trackedLabels['Filament'][idx], trackedLabels['Replicate'][idx], 'Completeness']
        dataAppend.extend(dataCompleteness)
        dataTrackedCells.loc[len(dataTrackedCells)] = dataAppend
                
        dataAppend = [trackedLabels['Genotype'][idx], trackedLabels['Filament'][idx], trackedLabels['Replicate'][idx], 'Lobes']
        dataAppend.extend(dataLobes)
        dataTrackedCells.loc[len(dataTrackedCells)] = dataAppend
    
    
    ###########################################################################
    ### relative completeness distributions for WT and mutant (tracked cells)
    # extract data from cells tracked over all time points
    completenessWT_tracked = dataTrackedCells[(dataTrackedCells['Property'] == 'Completeness') & (dataTrackedCells['Genotype'] == 'WT')].reset_index(drop=True)
    completenessM_tracked = dataTrackedCells[(dataTrackedCells['Property'] == 'Completeness') & (dataTrackedCells['Genotype'] == 'act2-1act7-1')].reset_index(drop=True)

    # use only data from matching time points between WT and mutant
    dataCompletenessWT = [completenessWT_tracked['0h'], completenessWT_tracked['12h'], completenessWT_tracked['24h'], completenessWT_tracked['36h']]
    dataCompletenessM = [completenessM_tracked['0h'], completenessM_tracked['12h'],  completenessM_tracked['24h'], completenessM_tracked['36h']]

    # create data frame for ridge plot
    df_t = create_df_for_ridgeplot(dataCompletenessWT, dataCompletenessM) 

    # plot relative completeness distributions as ridge plots
    fig, ax = joypy.joyplot(df_t, by='Timepoint', color=['cornflowerblue', 'gold'], alpha=0.5, figsize=(5, 5), overlap=0.25)
    plt.xlabel('Relative completeness')
    plt.ylabel('Timepoint')
    # plot means
    for i in range(4):
        ymin = 0
        meanWT = dataCompletenessWT[i].mean()
        meanM = dataCompletenessM[i].mean()
        yvalWT = ax[i].lines[1].get_ydata()[np.where(np.round(ax[i].lines[1].get_xdata(), 2) == np.around(meanWT, 2))[0][0]]
        yvalM = ax[i].lines[3].get_ydata()[np.where(np.round(ax[i].lines[3].get_xdata(), 2) == np.around(meanM, 2))[0][0]]
        ax[i].plot([meanWT, meanWT], [ymin, yvalWT], color='gray', ls='-', zorder=200)
        ax[i].plot([meanM, meanM], [ymin, yvalM], color='gray', ls='--', zorder=200)
    fig.savefig(pathPlots / 'Figure4C_Leaves_ShapeComplexityDistributions_WT_Mutant.png', bbox_inches='tight', dpi=300)               
    plt.close()
    
    # statistics
    print('Complexity (WT vs. M): \n...0h:', sp.stats.mannwhitneyu(dataCompletenessWT[0], dataCompletenessM[0])[1], '\n...12h:', sp.stats.mannwhitneyu(dataCompletenessWT[1], dataCompletenessM[1])[1], '\n...24h:', sp.stats.mannwhitneyu(dataCompletenessWT[2], dataCompletenessM[2])[1], '\n...36h:', sp.stats.mannwhitneyu(dataCompletenessWT[3], dataCompletenessM[3])[1])
    
    
    ### relative completeness vs. true lobes (tracked cells)
    lobesWT_tracked = dataTrackedCells[(dataTrackedCells['Property'] == 'Lobes') & (dataTrackedCells['Genotype'] == 'WT')].reset_index(drop=True)
    lobesM_tracked = dataTrackedCells[(dataTrackedCells['Property'] == 'Lobes') & (dataTrackedCells['Genotype'] == 'act2-1act7-1')].reset_index(drop=True)

    dataLobesWT = [lobesWT_tracked['0h'], lobesWT_tracked['12h'], lobesWT_tracked['24h'], lobesWT_tracked['36h']]
    dataLobesM = [lobesM_tracked['0h'], lobesM_tracked['12h'], lobesM_tracked['24h'], lobesM_tracked['36h']]
                
    # correlation plot completeness ~ lobes
    sizes = [30, 60, 90, 120]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.scatter([dataLobesWT[idx].mean() for idx in range(4)], [dataCompletenessWT[idx].mean() for idx in range(4)], c='cornflowerblue', s=sizes)
    plt.errorbar([dataLobesWT[idx].mean() for idx in range(4)], [dataCompletenessWT[idx].mean() for idx in range(4)], yerr=[[dataCompletenessWT[idx].mean() - dataCompletenessWT[idx].quantile(q=0.25) for idx in range(4)], [dataCompletenessWT[idx].quantile(q=0.75) - dataCompletenessWT[idx].mean() for idx in range(4)]], xerr=[[dataLobesWT[idx].mean() - dataLobesWT[idx].quantile(q=0.25) for idx in range(4)], [np.maximum(0, dataLobesWT[idx].quantile(q=0.75) - dataLobesWT[idx].mean()) for idx in range(4)]], fmt = 'o', color = 'cornflowerblue', linewidth=2, capsize=3, alpha=0.5)
    plt.scatter([dataLobesM[idx].mean() for idx in range(4)], [dataCompletenessM[idx].mean() for idx in range(4)], c='gold', s=sizes)
    plt.errorbar([dataLobesM[idx].mean() for idx in range(4)], [dataCompletenessM[idx].mean() for idx in range(4)], yerr=[[dataCompletenessM[idx].mean() - dataCompletenessM[idx].quantile(q=0.25) for idx in range(4)], [dataCompletenessM[idx].quantile(q=0.75) - dataCompletenessM[idx].mean() for idx in range(4)]], xerr=[[dataLobesM[idx].mean() - dataLobesM[idx].quantile(q=0.25) for idx in range(4)], [dataLobesM[idx].quantile(q=0.75) - dataLobesM[idx].mean() for idx in range(4)]], fmt = 'o', color = 'gold', linewidth=2, capsize=3, alpha=0.5)
    ax.set_ylabel('PC shape complexity')
    ax.set_xlabel('Number of true lobes')
    fig.savefig(pathPlots / 'Figure4D_Leaves_ShapeComplexityLobes_WT_Mutant.png', bbox_inches='tight', dpi=300)               
    plt.close()

    # statistical significance (Hotelling's T2 test)
    result0, result12, result24, result36 = get_Hotelling_pvalues(dataLobesWT, dataCompletenessWT, dataLobesM, dataCompletenessM)
    print("\nComplexity vs. lobes (WT vs. M): \n...Oh:", result0['pval'].iloc[0], '\n...12h:', result12['pval'].iloc[0], '\n...24h:', result24['pval'].iloc[0], '\n...36h', result36['pval'].iloc[0])  
    
    
    ### relative completeness vs. cell area (all cells)
    areaWT_tracked = dataTrackedCells[(dataTrackedCells['Property'] == 'Area') & (dataTrackedCells['Genotype'] == 'WT')].reset_index(drop=True)
    areaM_tracked = dataTrackedCells[(dataTrackedCells['Property'] == 'Area') & (dataTrackedCells['Genotype'] == 'act2-1act7-1')].reset_index(drop=True)
    
    xWT_reg, xM_reg, yWT_reg, yM_reg = extract_regression_data(areaWT_tracked, completenessWT_tracked, areaM_tracked, completenessM_tracked)
  
    slopeWT, slopeM, slopePvalue, slopeSE, rWT, rM = get_regression_data(xWT_reg, xM_reg, yWT_reg, yM_reg)
    print('\nComplexity vs. area (WT vs. M)')
    print('Slope_WT: ', slopeWT, ', Slope_M:', slopeM, ', P-value:', slopePvalue, ', SE:', slopeSE)
    print('...Pearson correlation coefficient: R_WT:', rWT, ', R_M:', rM)
    
    xWT_lin, xM_lin, yWT_lin, yM_lin, yWT_err, yM_err = create_linear_spaced_data(xWT_reg, xM_reg, slopeWT, slopeM)
    
    # correlation plot completeness ~ area
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(xWT_reg, yWT_reg, c='cornflowerblue', label='WT', s=25)
    ax.plot(xWT_lin, yWT_lin, c='cornflowerblue', linewidth=2)
    ax.fill_between(xWT_lin, yWT_lin - yWT_err, yWT_lin + yWT_err, color='cornflowerblue', alpha=0.2)
    ax.scatter(xM_reg, yM_reg, c='gold', label='act2-1act7-1', s=25)
    ax.plot(xM_lin, yM_lin, c='gold', linewidth=2)
    ax.fill_between(xM_lin, yM_lin - yM_err, yM_lin + yM_err, color='gold', alpha=0.2)
    ax.set_ylabel('PC shape complexity')
    ax.set_xlabel('log(Area_sc)')
    plt.legend()
    fig.savefig(pathPlots / 'Figure4E_Leaves_ShapeComplexityArea_WT_Mutant.png', bbox_inches='tight', dpi=300)               
    plt.close()    
 
    
###############################################################################
def merge_cells_from_labels(labelList, labeledImage):
    """
    given two cell labels, merge the cells into one
    """
    labeledCells = labeledImage.copy() * 0
    cellContours = []
    for label in labelList:
        coords = np.transpose(np.where(labeledImage == label + 1))
        for x, y in coords:
            labeledCells[x, y] = 1
        _, contourCell = GE.extract_cell_contour(label + 1, labeledImage)
        cellContours.append(contourCell)
    mergedContourImage = labeledImage.copy() * 0
    for xPos, yPos in np.concatenate((cellContours), axis=0):
        mergedContourImage[xPos, yPos] += 1
    mergedCellImage = (labeledCells * 2) + mergedContourImage
    mergedCellImage = mergedCellImage > 1
    mergedCellImageHoles = sp.ndimage.binary_fill_holes(mergedCellImage, disk(1))
    
    return(mergedCellImageHoles) 


def extract_selected_cells(cells, gt, tp, path):    
    if gt == 'WT':
        labeledImage = np.load(path / ('MTs/WT/' + str(tp) + '/R3/labeledCells.npy')) 
        resolution = 1 / 10.9051
    else:
        labeledImage = np.load(path / ('AFs/act2-1act7-1/' + str(tp) + '/R2/labeledCells.npy')) 
        resolution = 1 / 6.5445
        
    selectedCells = labeledImage.copy() * 0
    for label in cells[gt][tp]:
        if isinstance(label, tuple):
            cellImage = merge_cells_from_labels(label, labeledImage)
            _, cellContour = GE.extract_cell_contour(cellImage, 1)
        else:
            _, cellContour = GE.extract_cell_contour(labeledImage, label + 1)
        for x, y in cellContour:
            selectedCells[x, y] = 1

    imageDilated = skimage.morphology.binary_dilation(selectedCells == 1, disk(2))
    labeledImageSelected, labelsSelected = sp.ndimage.label(~(selectedCells == 1))
    return imageDilated, labeledImageSelected, labelsSelected


def extract_selected_cell_properties(labeledImageSelected, labelsSelected, resolution, prop):
    properties = []
    for label in range(2, labelsSelected + 1):
        visGraph, cellContour = GE.create_visibility_graph(labeledImageSelected, label, resolution)    
        if prop == 'lobeyness':
            pos = list(nx.get_node_attributes(visGraph, 'pos').values())
            hull = sp.spatial.ConvexHull(pos)
            lobeyness = len(cellContour) / hull.area 
            properties.append(lobeyness)
        elif prop == 'completeness':
            sigma = GE.compute_graph_complexity(visGraph)
            properties.append(1 - sigma)
    return properties


def create_df_for_ridgeplot(propertiesWT, propertiesM):
    tp = ['0h', '12h', '24h', '36h']
    df_t = pd.DataFrame(columns=['Timepoint', 'DistWT', 'DistM'])
    for idx in range(4):
        propWT = propertiesWT[idx].reset_index(drop=True)
        propM = propertiesM[idx].reset_index(drop=True)
        if len(propWT) < len(propM):
            for jdx in range(len(propM)):
                if jdx <= len(propWT) - 1:
                    dataAppend = [tp[idx], propWT[jdx], propM[jdx]]
                else:
                    dataAppend = [tp[idx], np.nan, propM[jdx]]
                df_t.loc[len(df_t)] = dataAppend
        elif len(propM) < len(propWT):
            for jdx in range(len(propWT)):
                if jdx <= len(propM) - 1:
                    dataAppend = [tp[idx], propWT[jdx], propM[jdx]]
                else:
                    dataAppend = [tp[idx], propWT[jdx], np.nan]
                df_t.loc[len(df_t)] = dataAppend   
        else:
            for jdx in range(len(propWT)):
                dataAppend = [tp[idx], propWT[jdx], propM[jdx]]
                df_t.loc[len(df_t)] = dataAppend
    return df_t


def get_Hotelling_pvalues(propWT1, propWT2, propM1, propM2):
    df0 = pd.DataFrame({'Genotype': len(propWT1[0]) * ['WT'] + len(propM1[0]) * ['M'], 
           'prop1': [x for n in (propWT1[0], propM1[0]) for x in n], 
           'prop2': [x for n in (propWT2[0], propM2[0]) for x in n]})
    df12 = pd.DataFrame({'Genotype': len(propWT1[1]) * ['WT'] + len(propM1[1]) * ['M'], 
           'prop1': [x for n in (propWT1[1], propM1[1]) for x in n], 
           'prop2': [x for n in (propWT2[1], propM2[1]) for x in n]})
    df24 = pd.DataFrame({'Genotype': len(propWT1[2]) * ['WT'] + len(propM1[2]) * ['M'], 
           'prop1': [x for n in (propWT1[2], propM1[2]) for x in n], 
           'prop2': [x for n in (propWT2[2], propM2[2]) for x in n]})
    df36 = pd.DataFrame({'Genotype': len(propWT1[3]) * ['WT'] + len(propM1[3]) * ['M'], 
           'prop1': [x for n in (propWT1[3], propM1[3]) for x in n], 
           'prop2': [x for n in (propWT2[3], propM2[3]) for x in n]})

    result0 = pg.multivariate_ttest(df0[df0['Genotype'] == 'WT'][['prop1', 'prop2']], df0[df0['Genotype'] == 'M'][['prop1', 'prop2']])
    result12 = pg.multivariate_ttest(df12[df12['Genotype'] == 'WT'][['prop1', 'prop2']], df12[df12['Genotype'] == 'M'][['prop1', 'prop2']])
    result24 = pg.multivariate_ttest(df24[df24['Genotype'] == 'WT'][['prop1', 'prop2']], df24[df24['Genotype'] == 'M'][['prop1', 'prop2']])
    result36 = pg.multivariate_ttest(df36[df36['Genotype'] == 'WT'][['prop1', 'prop2']], df36[df36['Genotype'] == 'M'][['prop1', 'prop2']])
    
    return result0, result12, result24, result36


def extract_regression_data(propWT1, propWT2, propM1, propM2):
    xWT, xM, yWT, yM = [], [], [], []
    for idx in range(len(propWT1)):
        xWT.extend([propWT1['0h'][idx], propWT1['12h'][idx], propWT1['24h'][idx], propWT1['36h'][idx]])
        yWT.extend([propWT2['0h'][idx], propWT2['12h'][idx], propWT2['24h'][idx], propWT2['36h'][idx]])
    for idx in range(len(propM1)):
        xM.extend([propM1['0h'][idx], propM1['12h'][idx], propM1['24h'][idx], propM1['36h'][idx]])
        yM.extend([propM2['0h'][idx], propM2['12h'][idx], propM2['24h'][idx], propM2['36h'][idx]])
        
    xWT_reg, xM_reg = prepare_regression_data(xWT, xM, True, 'unequal')
    yWT_reg, yM_reg = prepare_regression_data(yWT, yM, False, 'unequal')
    
    return xWT_reg, xM_reg, yWT_reg, yM_reg   
     

def get_regression_data(xWT_reg, xM_reg, yWT_reg, yM_reg):    
    # slopes   
    slopeWT, slopeM, slope_pval, slope_se = calculate_slope_difference(xWT_reg, xM_reg, yWT_reg, yM_reg)
    
    # correlation coefficients
    rWT, pWT = sp.stats.pearsonr(xWT_reg, yWT_reg)
    rM, pM = sp.stats.pearsonr(xM_reg, yM_reg)
    
    return slopeWT, slopeM, slope_pval, slope_se, rWT, rM


def prepare_regression_data(xWT, xM, log, samples):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    if samples == 'equal':
        nWT = np.sum([len(xWT[L]) for L in range(len(xWT))])
    else:
        nWT = len(xWT)
    xWT = np.array(xWT)
    xM = np.array(xM)
    xWT = xWT[:, np.newaxis]
    xM = xM[:, np.newaxis]
    if log == True:
        xWT = np.log10(xWT)
        xM = np.log10(xM)
    x = np.array(np.concatenate((xWT, xM))).reshape(-1, 1)
    x_scaled = min_max_scaler.fit_transform(x)
    xWT = x_scaled[:nWT]  
    xM = x_scaled[nWT:]  
    
    return(xWT.reshape(1, len(xWT))[0], xM.reshape(1, len(xM))[0]) 


def calculate_slope_difference(x1, x2, y1, y2):
    df = len(x1) + len(x2) - 4
    dfWT = pd.DataFrame({'y': y1.reshape(1, -1)[0], 'x': x1.reshape(1, -1)[0]})
    dfM = pd.DataFrame({'y': y2.reshape(1, -1)[0], 'x': x2.reshape(1, -1)[0]})
    modelWT = smf.ols('y ~ x -1', data=dfWT).fit()
    modelM = smf.ols('y ~ x -1', data=dfM).fit()
    slopeWT, slopeM = modelWT.params.iloc[0], modelM.params.iloc[0]
    slopeDifference = slopeWT - slopeM
    slopeSE = np.sqrt(modelWT.bse.iloc[0]**2 + modelM.bse.iloc[0]**2)
    slopeTstat = slopeDifference / slopeSE
    slopePvalue = 2 * (sp.stats.t.sf(np.abs(slopeTstat), df))
    
    return(slopeWT, slopeM, slopePvalue, slopeSE)
    

def create_linear_spaced_data(xWT, xM, slopeWT, slopeM):
    #linear spaced data
    xWT_lin = np.linspace(xWT.min(), xWT.max(), len(xWT))
    xM_lin = np.linspace(xM.min(), xM.max(), len(xM))  
    yWT_lin = np.array(slopeWT * xWT_lin)
    yM_lin = np.array(slopeM * xM_lin)
    # errors
    yWT_err = yWT_lin.std() * np.sqrt(1 / len(yWT_lin) + (yWT_lin - yWT_lin.mean())**2 / np.sum((yWT_lin - yWT_lin.mean())**2)) 
    yM_err = yM_lin.std() * np.sqrt(1 / len(yM_lin) + (yM_lin - yM_lin.mean())**2 / np.sum((yM_lin - yM_lin.mean())**2)) 
    
    return xWT_lin, xM_lin, yWT_lin, yM_lin, yWT_err, yM_err

###############################################################################

if __name__ == '__main__':
    main()
