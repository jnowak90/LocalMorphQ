#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 22:32:22 2026

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
import matplotlib
import matplotlib.pyplot as plt

# add current script directory to path
pathScript = Path(__file__).resolve().parent
sys.path.append(str(pathScript))
import GraVisExtraction as GE

###############################################################################

# =========================
# Figure 3
# Analysis of pavement cell shape complexity in cotyledons of wild-type and act2-1 act7-1
# plants.
# =========================

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to dataset')
    args = parser.parse_args()
    
    pathRoot = Path(args.data)
    pathCotyledon = pathRoot / 'Cotyledons'
    
    pathPlots = pathRoot / 'Plots'
    pathPlots.mkdir(parents=True, exist_ok=True)

    ###########################################################################
    ### create heatmaps of relative completeness  in cotyledons for WT and act2-1act7-1
    topO = plt.get_cmap('Oranges', 256)
    newcolorsO = np.vstack((topO(np.linspace(0, 0.1, 100)),
                           topO(np.linspace(0.1, 1, 156))))
    newcolorsO[0] = np.array([1.0, 1.0, 1.0, 0.0])
    cmapCellsO = matplotlib.colors.ListedColormap(newcolorsO, name='Orange')
    cmapBinary = matplotlib.colors.ListedColormap([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) # black
        
    
    ### plot cell outlines for selected cells
    cells = {'WT': [16, 18, 19, 20, 22, (23, 31, 33), 24, 26, (29, 38), 30, 32, 35, 40, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 62, 63, 64, 65, (66, 70, 71), 67, 68, 73, 74, (76, 78, 81, 82, 86), 79, 80, 84, 87, 88, 89, 92, 95, 99, 101, 104, 100, 108, 109], 
             'M': [4, 5, 6, 7, 8, 9, 11, 12, 13, 14, (16, 21), 17, 18, 19, 20, 22, 24, 25, 27, (28, 33), 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, (51, 54), 52, 53, 55, 56, (58, 60), 59, 61, (64, 65)]} 
    
    for gt in cells.keys():
        if gt == 'WT':
            labeledImage = np.load(pathCotyledon / 'MTs/WT/WT_4/labeledCells.npy')
            resolution = 1 / 6.9144
            name = 'Figure3A_Cotyledons_ComplexityHeatmap_WT.png'
        else:
            labeledImage = np.load(pathCotyledon / 'MTs/act2-1act7-1/act2act7_1/labeledCells.npy')
            resolution = 1 / 6.9476
            name = 'Figure3B_Cotyledons_ComplexityHeatmap_Mutant.png'
            
        # cell outlines
        selectedCells = labeledImage.copy() * 0
        for label in cells[gt]:
            if not isinstance(label, tuple):
               cellImage = labeledImage == label + 1
               visGraph, cellContour = GE.create_visibility_graph(labeledImage, label + 1, resolution)
            else:
                cellImage = merge_cells_from_labels(label, labeledImage)
                visGraph, cellContour = GE.create_visibility_graph(cellImage * 1, 1, resolution)
            for x, y in cellContour:
                selectedCells[x, y] = 1
        imageDilated = skimage.morphology.binary_dilation(selectedCells == 1, disk(2))
        
        # heatmaps
        labeledImageSelected, labelsSelected = sp.ndimage.label(~(selectedCells == 1))
        
        completeness = []
        for label in range(2, labelsSelected + 1):
            visGraph, cellContour = GE.create_visibility_graph(labeledImageSelected, label, 1/6.5445) ###
            sigma = GE.compute_graph_complexity(visGraph)
            if sigma == 1:
                completeness.append(0.05)
            else:
                completeness.append(1 - sigma)
            
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        imageColored_C = labeledImageSelected.copy() * 0
        imageColored_C = imageColored_C.astype(float)
        imageColored_C[0, 0] = 1.0
        for label in range(len(completeness)):
            coord = np.transpose(np.where(labeledImageSelected == label + 2))
            for x, y in coord:
                imageColored_C[x, y] = completeness[label]
        im = ax.imshow(imageColored_C, cmap=cmapCellsO, vmin=0, vmax=0.7, zorder=0)
        ax.imshow(imageDilated==1, cmap=cmapBinary, alpha=1, zorder=2)
        ax.axis('off')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('PC shape complexity', rotation=90, labelpad=20)
        fig.savefig(pathPlots / name, bbox_inches='tight', dpi=300) ###
        plt.close()
        
        
    ###########################################################################
    ### cell properties WT vs. mutant
    globalDensities = pd.read_csv(pathCotyledon / 'globalDensities.csv')
    cotyledonWT = globalDensities[globalDensities['Genotype'] == 'WT'].reset_index(drop=True)
    cotyledonM = globalDensities[(globalDensities['Genotype'] == 'act2-1act7-1') & (globalDensities['Area [µm2]'] > 11)].reset_index(drop=True)
    
    completenessWT = cotyledonWT['RelativeCompleteness']
    completenessM = cotyledonM['RelativeCompleteness']
    print('Two-sample t-test (complexity WT vs. M): ', sp.stats.mannwhitneyu(completenessWT, completenessM)[1])
    
    lobesWT = cotyledonWT['TrueLobes']
    lobesM = cotyledonM['TrueLobes']
    
    cotyledonWT_AF = cotyledonWT[cotyledonWT['Filament'] == 'AFs']
    cotyledonWT_MT = cotyledonWT[cotyledonWT['Filament'] == 'MTs']
    cotyledonM_AF = cotyledonM[cotyledonM['Filament'] == 'AFs']
    cotyledonM_MT = cotyledonM[cotyledonM['Filament'] == 'MTs']
    
    ### completeness vs. true lobes
    # prepare data for individual replicates
    completenessWT_AF_reps = [cotyledonWT_AF[cotyledonWT_AF['Replicate'] == rep]['RelativeCompleteness'].mean() for rep in cotyledonWT_AF['Replicate'].unique()]
    completenessWT_MT_reps = [cotyledonWT_MT[cotyledonWT_MT['Replicate'] == rep]['RelativeCompleteness'].mean() for rep in cotyledonWT_MT['Replicate'].unique()]
    completenessM_AF_reps = [cotyledonM_AF[cotyledonM_AF['Replicate'] == rep]['RelativeCompleteness'].mean() for rep in cotyledonM_AF['Replicate'].unique()]
    completenessM_MT_reps = [cotyledonM_MT[cotyledonM_MT['Replicate'] == rep]['RelativeCompleteness'].mean() for rep in cotyledonM_MT['Replicate'].unique()]
    
    lobesWT_AF_reps = [cotyledonWT_AF[cotyledonWT_AF['Replicate'] == rep]['TrueLobes'].mean() for rep in cotyledonWT_AF['Replicate'].unique()]
    lobesWT_MT_reps = [cotyledonWT_MT[cotyledonWT_MT['Replicate'] == rep]['TrueLobes'].mean() for rep in cotyledonWT_MT['Replicate'].unique()]
    lobesM_AF_reps = [cotyledonM_AF[cotyledonM_AF['Replicate'] == rep]['TrueLobes'].mean() for rep in cotyledonM_AF['Replicate'].unique()]
    lobesM_MT_reps = [cotyledonM_MT[cotyledonM_MT['Replicate'] == rep]['TrueLobes'].mean() for rep in cotyledonM_MT['Replicate'].unique()]
    
    completenessWT_reps = np.concatenate([completenessWT_AF_reps, completenessWT_MT_reps])
    completenessM_reps = np.concatenate([completenessM_AF_reps, completenessM_MT_reps])
    lobesWT_reps = np.concatenate([lobesWT_AF_reps, lobesWT_MT_reps])
    lobesM_reps = np.concatenate([lobesM_AF_reps, lobesM_MT_reps])
    
    # calculate regression slope
    print('Complexity vs. lobes, WT vs. M')
    lobesWT_reg, lobesM_reg = prepare_data(lobesWT_reps, lobesM_reps)
    completenessWT_reg, completenessM_reg = prepare_data(completenessWT_reps, completenessM_reps)
    slopeWT_LC, slopeM_LC, _, _ = calculate_slope_difference(lobesWT_reg, lobesM_reg, completenessWT_reg, completenessM_reg)
    
    # linear spaced data
    xWT_L_lin = np.linspace(lobesWT_reg.min(), lobesWT_reg.max(), len(lobesWT_reg))
    xM_L_lin = np.linspace(lobesM_reg.min(), lobesM_reg.max(), len(lobesM_reg))
    yWT_LC_lin = np.array(slopeWT_LC * xWT_L_lin)
    yM_LC_lin = np.array(slopeM_LC * xM_L_lin)
    
    rWT_LC, pWT_LC = sp.stats.spearmanr(lobesWT_reps, completenessWT_reps)
    rM_LC, pM_LC = sp.stats.spearmanr(lobesM_reps, completenessM_reps)
    print('Pearson correlation coefficient: R_WT:', rWT_LC, ', R_M:', rM_LC)
    
    ### completeness vs. area
    areaWT = cotyledonWT['Area [µm2]']
    areaM = cotyledonM['Area [µm2]']
    
    # regression data
    xWT_A_reg, xM_A_reg = prepare_regression_data(areaWT, areaM, True, 'unequal')
    yWT_C_reg, yM_C_reg = prepare_regression_data(completenessWT, completenessM, False, 'unequal')
    
    # slopes    
    print('Complexity vs. area, WT vs. M')
    slopeWT_AC, slopeM_AC, slope_pvalAC, slope_seAC = calculate_slope_difference(xWT_A_reg, xM_A_reg, yWT_C_reg, yM_C_reg)
    
    rWT_AC, pWT_AC = sp.stats.pearsonr(xWT_A_reg, yWT_C_reg)
    rM_AC, pM_AC = sp.stats.pearsonr(xM_A_reg, yM_C_reg)
    print('Pearson correlation coefficient: R_WT:', rWT_AC, ', R_M:', rM_AC)
    
    # linear spaced data
    xWT_A_lin = np.linspace(xWT_A_reg.min(), xWT_A_reg.max(), len(xWT_A_reg))
    xM_A_lin = np.linspace(xM_A_reg.min(), xM_A_reg.max(), len(xM_A_reg))
    yWT_AC_lin = np.array(slopeWT_AC * xWT_A_lin)
    yM_AC_lin = np.array(slopeM_AC * xM_A_lin)
    
    # errors
    yWT_AC_err = yWT_AC_lin.std() * np.sqrt(1 / len(yWT_AC_lin) + (yWT_AC_lin - yWT_AC_lin.mean())**2 / np.sum((yWT_AC_lin - yWT_AC_lin.mean())**2)) 
    yM_AC_err = yM_AC_lin.std() * np.sqrt(1 / len(yM_AC_lin) + (yM_AC_lin - yM_AC_lin.mean())**2 / np.sum((yM_AC_lin - yM_AC_lin.mean())**2)) 
    
    ### plot
    colors = ['cornflowerblue', 'gold']
    fig, ax = plt.subplots(1, 3, figsize=(11, 5), gridspec_kw={'width_ratios': [1, 1.4, 1.4]})
    bp = ax[0].violinplot([completenessWT, completenessM], positions=[1, 1.5], widths=0.4, showextrema=False, showmedians=True)
    bp['cmedians'].set_color('black')
    for idx, box in enumerate(bp['bodies']):
        box.set_facecolor(colors[idx])
        box.set_edgecolor('black')
    ax[0].set_xticks([1, 1.5])
    ax[0].set_xticklabels(['WT', 'act2-1 act7-1'])
    ax[0].set_ylabel('PC shape complexity')
    ax[0].set_ylim(-0.05, 0.8)
    ###
    ax[1].scatter(lobesWT_reg, completenessWT_reg, c='cornflowerblue', label='WT', s=40)
    ax[1].plot(xWT_L_lin, yWT_LC_lin, c='cornflowerblue', linewidth=2)
    ax[1].scatter(lobesM_reg, completenessM_reg, c='gold', label='act2-1act7-1', s=40)
    ax[1].plot(xM_L_lin, yM_LC_lin, c='gold', linewidth=2)
    ax[1].set_ylabel('PC shape complexity')
    ax[1].set_xlabel('Number of true lobes')
    ###
    ax[2].scatter(xWT_A_reg, yWT_C_reg, c='cornflowerblue', label='WT', s=25, alpha=0.5)
    ax[2].plot(xWT_A_lin, yWT_AC_lin, c='cornflowerblue', linewidth=2)
    ax[2].fill_between(xWT_A_lin, yWT_AC_lin - yWT_AC_err, yWT_AC_lin + yWT_AC_err, color='cornflowerblue', alpha=0.2)
    ax[2].scatter(xM_A_reg, yM_C_reg, c='gold', label='act2-1act7-1', s=25, alpha=0.5)
    ax[2].plot(xM_A_lin, yM_AC_lin, c='gold', linewidth=2)
    ax[2].fill_between(xM_A_lin, yM_AC_lin - yM_AC_err, yM_AC_lin + yM_AC_err, color='gold', alpha=0.2)
    ax[2].set_ylabel('PC shape complexity')
    ax[2].set_xlabel('log(Area_sc)')
    ax[2].legend()
    plt.tight_layout()
    fig.savefig(pathPlots / 'Figure3C-E_Cotyledons_ShapeComplexityLobesArea_WT_Mutant.png', bbox_inches='tight', dpi=300)               
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


def prepare_data(xWT, xM):
    xWT = np.array(xWT)
    xM = np.array(xM)
    xWT = xWT[:, np.newaxis]
    xM = xM[:, np.newaxis]
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
    print('Slope_WT: ', slopeWT, ', Slope_M:', slopeM, ', P-value:', slopePvalue, ', SE:', slopeSE)
    return(slopeWT, slopeM, slopePvalue, slopeSE)


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

###############################################################################

if __name__ == '__main__':
    main()