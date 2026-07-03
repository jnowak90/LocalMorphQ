#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:37:09 2026

@author: Jacqueline Nowak, JNowak@mpimp-golm.mpg.de
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy as sp
import skimage
import pingouin as pg
import networkx as nx
import matplotlib.pyplot as plt
import rich
from rich.table import Table
from rich.console import Console
    
# add current script directory to path
pathScript = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathScript))
import src.GraVisExtraction as GE
import src.Utils as ASU


###############################################################################
# =========================
# Figure 2
# Optimal mask sizes for perpendicular and circular masks
# =========================

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to dataset')
    args = parser.parse_args()
    
    pathRoot = Path(args.data)

    pathPlots = pathRoot / 'Plots'
    pathPlots.mkdir(parents=True, exist_ok=True)
    console = Console()
    
    ###########################################################################
    tableDensities = pd.DataFrame(columns=['Filament', 'Replicate', 'CellNumber', 'Mask', 'Density', 'DensityCell', 'Area [µm2]', 'Completeness', 'NodeIdx', 'NodeType', 'length=5', 'length=10', 'length=15', 'length=20', 'length=25', 'length=30', 'length=35','length=40', 'length=45', 'length=50'])
    maskSizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    cytoskeleton = ['MTs', 'AFs', 'AFs', 'MTs', 'MTs', 'AFs', 'MTs', 'AFs', 'AFs', 'AFs']
    replicates = ['R1', 'R2', 'R1', 'R1', 'R1', 'R1', 'R2', 'R1', 'R1', 'R2']
    cellnumbers = [19, 15, 58, 8, 17, 13, 19, 56, 29, 1]
    resolutions = [1/10.9051, 1/6.5445, 1/6.5445, 1/10.9051, 1/10.9051, 1/6.5445, 1/10.9051, 1/6.5445, 1/6.5445, 1/6.5445]

    for idx in range(10):
        densityEqual = skimage.io.imread(pathRoot / 'SyntheticPCs' / 'EqualDensity' / (str(idx + 1) + '_equalDensity.png'))
        densityLobe = skimage.io.imread(pathRoot / 'SyntheticPCs' / 'LobeDensity' /  (str(idx + 1) + '_lobeDensity.png'))
        densityNeck = skimage.io.imread(pathRoot / 'SyntheticPCs' / 'NeckDensity' / (str(idx + 1) + '_neckDensity.png'))
        resolution = resolutions[idx] 
            
        # create visGraph
        cellImage = (densityEqual == 127) * 1 
        coordsContour = np.transpose(np.where(cellImage == 1))
        cellImage = sp.ndimage.binary_fill_holes(cellImage)
        for x, y in coordsContour:
            cellImage[x, y] = 0
        visGraph, cellContour = GE.create_visibility_graph(cellImage * 1, 1, resolution)
        
        # prepare density images (int32, background = -2, interior = 2)
        densityEqualFiltered = (densityEqual.astype(np.int32).copy() * 0) - 2
        densityLobeFiltered = (densityLobe.astype(np.int32).copy() * 0) - 2
        densityNeckFiltered = (densityNeck.astype(np.int32).copy() * 0) - 2
        coordsCellInterior = np.transpose(np.where(cellImage == 1))
        for x, y in coordsCellInterior:
            if densityEqual[x, y] == 255:
                densityEqualFiltered[x, y] = 2
            else:
                densityEqualFiltered[x, y] = 0
            if densityLobe[x, y] == 255:
                densityLobeFiltered[x, y] = 2
            else:
                densityLobeFiltered[x, y] = 0
            if densityNeck[x, y] == 255:
                densityNeckFiltered[x, y] = 2
            else:
                densityNeckFiltered[x, y] = 0
        
        # calculate shape properties
        cellDensity = np.sum(densityEqual == 255) / np.sum(cellImage)
        completeness = GE.compute_graph_complexity(visGraph)
        area = np.sum(cellImage) * (resolution ** 2)      

        # calculate lobe and neck positions
        lobes, necks = GE.count_lobes_and_necks(visGraph)
        lobePos = get_lobe_neck_positions(visGraph)[0]
        neckPos = get_lobe_neck_positions(visGraph)[1]
        pos = nx.get_node_attributes(visGraph, 'pos')
        for node in pos.keys():
            if node in lobePos.keys():
                nodeType = 'lobe'
            elif node in neckPos.keys():
                nodeType = 'neck'
            else:
                nodeType = 'none'
                
            ### perpendicular mask
            # equal density
            densityPerpEqual = [ASU.calculate_density(densityEqualFiltered, node, pos, visGraph, length, 10, 0, 'perpendicular') for length in maskSizes]         
            dataAppendPerpEqual = [cytoskeleton[idx], replicates[idx], cellnumbers[idx], 'perpendicular', 'equal', cellDensity, area, completeness, node, nodeType]
            dataAppendPerpEqual.extend(densityPerpEqual)
            tableDensities.loc[len(tableDensities)] = dataAppendPerpEqual
            # lobe-centered density
            densityPerpLobe = [ASU.calculate_density(densityLobeFiltered, node, pos, visGraph, length, 10, 0, 'perpendicular') for length in maskSizes]
            dataAppendPerpLobe = [cytoskeleton[idx], replicates[idx], cellnumbers[idx], 'perpendicular', 'lobe', cellDensity, area, completeness, node, nodeType]
            dataAppendPerpLobe.extend(densityPerpLobe)
            tableDensities.loc[len(tableDensities)] = dataAppendPerpLobe
            # neck-centered density
            densityPerpNeck = [ASU.calculate_density(densityNeckFiltered, node, pos, visGraph, length, 10, 0, 'perpendicular') for length in maskSizes]
            dataAppendPerpNeck = [cytoskeleton[idx], replicates[idx], cellnumbers[idx], 'perpendicular', 'neck', cellDensity, area, completeness, node, nodeType]
            dataAppendPerpNeck.extend(densityPerpNeck)
            tableDensities.loc[len(tableDensities)] = dataAppendPerpNeck
            
            ### circular mask
            # equal density
            densityCircEqual = [ASU.calculate_density(densityEqualFiltered, node, pos, visGraph, length, 0, 0, 'circular') for length in maskSizes]
            dataAppendCircEqual = [cytoskeleton[idx], replicates[idx], cellnumbers[idx], 'circular', 'equal', cellDensity, area, completeness, node, nodeType]
            dataAppendCircEqual.extend(densityCircEqual)
            tableDensities.loc[len(tableDensities)] = dataAppendCircEqual
            # lobe-centered density
            densityCircLobe = [ASU.calculate_density(densityLobeFiltered, node, pos, visGraph, length, 0, 0, 'circular') for length in maskSizes]
            dataAppendCircLobe = [cytoskeleton[idx], replicates[idx], cellnumbers[idx], 'circular', 'lobe', cellDensity, area, completeness, node, nodeType]
            dataAppendCircLobe.extend(densityCircLobe)
            tableDensities.loc[len(tableDensities)] = dataAppendCircLobe
            # neck-centered density
            densityCircNeck = [ASU.calculate_density(densityNeckFiltered, node, pos, visGraph,  length, 0, 0, 'circular') for length in maskSizes]
            dataAppendCircNeck = [cytoskeleton[idx], replicates[idx], cellnumbers[idx], 'circular', 'neck', cellDensity, area, completeness, node, nodeType]
            dataAppendCircNeck.extend(densityCircNeck)
            tableDensities.loc[len(tableDensities)] = dataAppendCircNeck
    tableDensities.to_csv(pathPlots / 'OptimalMaskSizes.csv', index=None)

    ### plot mask densities
    densitiesLobesPerp = tableDensities[(tableDensities['NodeType'] == 'lobe') & (tableDensities['Mask'] == 'perpendicular')]  
    densitiesNecksPerp = tableDensities[(tableDensities['NodeType'] == 'neck') & (tableDensities['Mask'] == 'perpendicular')]  
    densitiesNonePerp = tableDensities[(tableDensities['NodeType'] == 'none') & (tableDensities['Mask'] == 'perpendicular')]   
    densitiesLobesCirc = tableDensities[(tableDensities['NodeType'] == 'lobe') & (tableDensities['Mask'] == 'circular')]  
    densitiesNecksCirc = tableDensities[(tableDensities['NodeType'] == 'neck') & (tableDensities['Mask'] == 'circular')]  
    densitiesNoneCirc = tableDensities[(tableDensities['NodeType'] == 'none') & (tableDensities['Mask'] == 'circular')] 
            
    fig, ax = plt.subplots(2, 3, figsize=(12, 12))
    ### perpendicular mask
    # equal distribution
    ax[0, 0].scatter(maskSizes, [densitiesLobesPerp[densitiesLobesPerp['Density'] == 'equal'].iloc[:, -10:].mean().values][0], c='darkorange')
    ax[0, 0].errorbar(maskSizes, [densitiesLobesPerp[densitiesLobesPerp['Density'] == 'equal'].iloc[:, -10:].mean().values][0], yerr=[densitiesLobesPerp[densitiesLobesPerp['Density'] == 'equal'].iloc[:, -10:].sem().values[0]], fmt = 'o', color = 'darkorange', linewidth=2, capsize=3, alpha=0.5, label='lobes')
    ax[0, 0].scatter(maskSizes, [densitiesNecksPerp[densitiesNecksPerp['Density'] == 'equal'].iloc[:, -10:].mean().values][0], c='cornflowerblue')
    ax[0, 0].errorbar(maskSizes, [densitiesNecksPerp[densitiesNecksPerp['Density'] == 'equal'].iloc[:, -10:].mean().values][0], yerr=[densitiesNecksPerp[densitiesNecksPerp['Density'] == 'equal'].iloc[:, -10:].sem().values[0]], fmt = 'o', color = 'cornflowerblue', linewidth=2, capsize=3, alpha=0.5, label='necks')
    ax[0, 0].scatter(maskSizes, [densitiesNonePerp[densitiesNonePerp['Density'] == 'equal'].iloc[:, -10:].mean().values][0], c='seagreen', alpha=0.25)
    ax[0, 0].errorbar(maskSizes, [densitiesNonePerp[densitiesNonePerp['Density'] == 'equal'].iloc[:, -10:].mean().values][0], yerr=[densitiesNonePerp[densitiesNonePerp['Density'] == 'equal'].iloc[:, -10:].sem().values[0]], fmt = 'o', color = 'seagreen', linewidth=2, capsize=3, alpha=0.25, label='none')
    ax[0, 0].set_ylabel('Density')
    ax[0, 0].set_xlabel('Mask length')
    ax[0, 0].set_xticks(maskSizes)
    ax[0, 0].set_xticklabels(maskSizes)
    # lobe-centered distribution
    ax[0, 1].scatter(maskSizes, [densitiesLobesPerp[densitiesLobesPerp['Density'] == 'lobe'].iloc[:, -10:].mean().values][0], c='darkorange')
    ax[0, 1].errorbar(maskSizes, [densitiesLobesPerp[densitiesLobesPerp['Density'] == 'lobe'].iloc[:, -10:].mean().values][0], yerr=[densitiesLobesPerp[densitiesLobesPerp['Density'] == 'lobe'].iloc[:, -10:].sem().values[0]], fmt = 'o', color = 'darkorange', linewidth=2, capsize=3, alpha=0.5, label='lobes')
    ax[0, 1].scatter(maskSizes, [densitiesNecksPerp[densitiesNecksPerp['Density'] == 'lobe'].iloc[:, -10:].mean().values][0], c='cornflowerblue')
    ax[0, 1].errorbar(maskSizes, [densitiesNecksPerp[densitiesNecksPerp['Density'] == 'lobe'].iloc[:, -10:].mean().values][0], yerr=[densitiesNecksPerp[densitiesNecksPerp['Density'] == 'lobe'].iloc[:, -10:].sem().values[0]], fmt = 'o', color = 'cornflowerblue', linewidth=2, capsize=3, alpha=0.5, label='necks')
    ax[0, 1].scatter(maskSizes, [densitiesNonePerp[densitiesNonePerp['Density'] == 'lobe'].iloc[:, -10:].mean().values][0], c='seagreen', alpha=0.25)
    ax[0, 1].errorbar(maskSizes, [densitiesNonePerp[densitiesNonePerp['Density'] == 'lobe'].iloc[:, -10:].mean().values][0], yerr=[densitiesNonePerp[densitiesNonePerp['Density'] == 'lobe'].iloc[:, -10:].sem().values[0]], fmt = 'o', color = 'seagreen', linewidth=2, capsize=3, alpha=0.25, label='none')
    ax[0, 1].set_ylabel('Density')
    ax[0, 1].set_xlabel('Mask length')
    ax[0, 1].set_xticks(maskSizes)
    ax[0, 1].set_xticklabels(maskSizes)
    # neck-centered distribution
    ax[0, 2].scatter(maskSizes, [densitiesLobesPerp[densitiesLobesPerp['Density'] == 'neck'].iloc[:, -10:].mean().values][0], c='darkorange')
    ax[0, 2].errorbar(maskSizes, [densitiesLobesPerp[densitiesLobesPerp['Density'] == 'neck'].iloc[:, -10:].mean().values][0], yerr=[densitiesLobesPerp[densitiesLobesPerp['Density'] == 'neck'].iloc[:, -10:].sem().values[0]], fmt = 'o', color = 'darkorange', linewidth=2, capsize=3, alpha=0.5, label='lobes')
    ax[0, 2].scatter(maskSizes, [densitiesNecksPerp[densitiesNecksPerp['Density'] == 'neck'].iloc[:, -10:].mean().values][0], c='cornflowerblue')
    ax[0, 2].errorbar(maskSizes, [densitiesNecksPerp[densitiesNecksPerp['Density'] == 'neck'].iloc[:, -10:].mean().values][0], yerr=[densitiesNecksPerp[densitiesNecksPerp['Density'] == 'neck'].iloc[:, -10:].sem().values[0]], fmt = 'o', color = 'cornflowerblue', linewidth=2, capsize=3, alpha=0.5, label='necks')
    ax[0, 2].scatter(maskSizes, [densitiesNonePerp[densitiesNonePerp['Density'] == 'neck'].iloc[:, -10:].mean().values][0], c='seagreen', alpha=0.25)
    ax[0, 2].errorbar(maskSizes, [densitiesNonePerp[densitiesNonePerp['Density'] == 'neck'].iloc[:, -10:].mean().values][0], yerr=[densitiesNonePerp[densitiesNonePerp['Density'] == 'neck'].iloc[:, -10:].sem().values[0]], fmt = 'o', color = 'seagreen', linewidth=2, capsize=3, alpha=0.25, label='none')
    ax[0, 2].set_ylabel('Density')
    ax[0, 2].set_xlabel('Mask length')
    ax[0, 2].set_xticks(maskSizes)
    ax[0, 2].set_xticklabels(maskSizes)
    ### circular mask
    # equal distribution
    ax[1, 0].scatter(maskSizes, [densitiesLobesCirc[densitiesLobesCirc['Density'] == 'equal'].iloc[:, -10:].mean().values][0], c='darkorange')
    ax[1, 0].errorbar(maskSizes, [densitiesLobesCirc[densitiesLobesCirc['Density'] == 'equal'].iloc[:, -10:].mean().values][0], yerr=[densitiesLobesCirc[densitiesLobesCirc['Density'] == 'equal'].iloc[:, -10:].sem().values[0]], fmt = 'o', color = 'darkorange', linewidth=2, capsize=3, alpha=0.5, label='lobes')
    ax[1, 0].scatter(maskSizes, [densitiesNecksCirc[densitiesNecksCirc['Density'] == 'equal'].iloc[:, -10:].mean().values][0], c='cornflowerblue')
    ax[1, 0].errorbar(maskSizes, [densitiesNecksCirc[densitiesNecksCirc['Density'] == 'equal'].iloc[:, -10:].mean().values][0], yerr=[densitiesNecksCirc[densitiesNecksCirc['Density'] == 'equal'].iloc[:, -10:].sem().values[0]], fmt = 'o', color = 'cornflowerblue', linewidth=2, capsize=3, alpha=0.5, label='necks')
    ax[1, 0].scatter(maskSizes, [densitiesNoneCirc[densitiesNoneCirc['Density'] == 'equal'].iloc[:, -10:].mean().values][0], c='seagreen', alpha=0.25)
    ax[1, 0].errorbar(maskSizes, [densitiesNoneCirc[densitiesNoneCirc['Density'] == 'equal'].iloc[:, -10:].mean().values][0], yerr=[densitiesNoneCirc[densitiesNoneCirc['Density'] == 'equal'].iloc[:, -10:].sem().values[0]], fmt = 'o', color = 'seagreen', linewidth=2, capsize=3, alpha=0.25, label='none')
    ax[1, 0].set_ylabel('Density')
    ax[1, 0].set_xlabel('Mask radius')
    ax[1, 0].set_xticks(maskSizes)
    ax[1, 0].set_xticklabels(maskSizes)
    # lobe-centered distribution
    ax[1, 1].scatter(maskSizes, [densitiesLobesCirc[densitiesLobesCirc['Density'] == 'lobe'].iloc[:, -10:].mean().values][0], c='darkorange')
    ax[1, 1].errorbar(maskSizes, [densitiesLobesCirc[densitiesLobesCirc['Density'] == 'lobe'].iloc[:, -10:].mean().values][0], yerr=[densitiesLobesCirc[densitiesLobesCirc['Density'] == 'lobe'].iloc[:, -10:].sem().values[0]], fmt = 'o', color = 'darkorange', linewidth=2, capsize=3, alpha=0.5, label='lobes')
    ax[1, 1].scatter(maskSizes, [densitiesNecksCirc[densitiesNecksCirc['Density'] == 'lobe'].iloc[:, -10:].mean().values][0], c='cornflowerblue')
    ax[1, 1].errorbar(maskSizes, [densitiesNecksCirc[densitiesNecksCirc['Density'] == 'lobe'].iloc[:, -10:].mean().values][0], yerr=[densitiesNecksCirc[densitiesNecksCirc['Density'] == 'lobe'].iloc[:, -10:].sem().values[0]], fmt = 'o', color = 'cornflowerblue', linewidth=2, capsize=3, alpha=0.5, label='necks')
    ax[1, 1].scatter(maskSizes, [densitiesNoneCirc[densitiesNoneCirc['Density'] == 'lobe'].iloc[:, -10:].mean().values][0], c='seagreen', alpha=0.25)
    ax[1, 1].errorbar(maskSizes, [densitiesNoneCirc[densitiesNoneCirc['Density'] == 'lobe'].iloc[:, -10:].mean().values][0], yerr=[densitiesNoneCirc[densitiesNoneCirc['Density'] == 'lobe'].iloc[:, -10:].sem().values[0]], fmt = 'o', color = 'seagreen', linewidth=2, capsize=3, alpha=0.25, label='none')
    ax[1, 1].set_ylabel('Density')
    ax[1, 1].set_xlabel('Mask radius')
    ax[1, 1].set_xticks(maskSizes)
    ax[1, 1].set_xticklabels(maskSizes)
    # neck-centered distribution
    ax[1, 2].scatter(maskSizes, [densitiesLobesCirc[densitiesLobesCirc['Density'] == 'neck'].iloc[:, -10:].mean().values][0], c='darkorange')
    ax[1, 2].errorbar(maskSizes, [densitiesLobesCirc[densitiesLobesCirc['Density'] == 'neck'].iloc[:, -10:].mean().values][0], yerr=[densitiesLobesCirc[densitiesLobesCirc['Density'] == 'neck'].iloc[:, -10:].sem().values[0]], fmt = 'o', color = 'darkorange', linewidth=2, capsize=3, alpha=0.5, label='lobes')
    ax[1, 2].scatter(maskSizes, [densitiesNecksCirc[densitiesNecksCirc['Density'] == 'neck'].iloc[:, -10:].mean().values][0], c='cornflowerblue')
    ax[1, 2].errorbar(maskSizes, [densitiesNecksCirc[densitiesNecksCirc['Density'] == 'neck'].iloc[:, -10:].mean().values][0], yerr=[densitiesNecksCirc[densitiesNecksCirc['Density'] == 'neck'].iloc[:, -10:].sem().values[0]], fmt = 'o', color = 'cornflowerblue', linewidth=2, capsize=3, alpha=0.5, label='necks')
    ax[1, 2].scatter(maskSizes, [densitiesNoneCirc[densitiesNoneCirc['Density'] == 'neck'].iloc[:, -10:].mean().values][0], c='seagreen', alpha=0.25)
    ax[1, 2].errorbar(maskSizes, [densitiesNoneCirc[densitiesNoneCirc['Density'] == 'neck'].iloc[:, -10:].mean().values][0], yerr=[densitiesNoneCirc[densitiesNoneCirc['Density'] == 'neck'].iloc[:, -10:].sem().values[0]], fmt = 'o', color = 'seagreen', linewidth=2, capsize=3, alpha=0.25, label='none')
    ax[1, 2].set_ylabel('Density')
    ax[1, 2].set_xlabel('Mask radius')
    ax[1, 2].set_xticks(maskSizes)
    ax[1, 2].set_xticklabels(maskSizes)
    ax[0, 0].set_ylim([0.3, 0.4])
    ax[0, 1].set_ylim([0.05, 0.7])
    ax[0, 2].set_ylim([0.05, 0.7])
    ax[1, 0].set_ylim([0.3, 0.4])
    ax[1, 1].set_ylim([0.05, 0.7])
    ax[1, 2].set_ylim([0.05, 0.7])
    ax[0, 0].legend()
    plt.tight_layout()
    fig.savefig(pathPlots / 'Figure2_SyntheticPCs_OptimalDensityMasks.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    ### statistics
    pValues = pd.DataFrame(columns=['Mask', 'Type', 'MaskSize', 'ANOVA', 'Lobe~Neck', 'Lobe~None', 'Neck~None'])

    masks = ['perpendicular', 'circular']
    datasetType = ['equal', 'lobe', 'neck']
    for mask in masks:
        for d in datasetType:
            for m in maskSizes:
                data = pd.DataFrame({'Values': pd.concat([tableDensities[(tableDensities['Mask'] == mask) & (tableDensities['Density'] == d) & (tableDensities['NodeType'] == 'lobe')]['length=' + str(m)], tableDensities[(tableDensities['Mask'] == mask) & (tableDensities['Density'] == d) & (tableDensities['NodeType'] == 'neck')]['length=' + str(m)], tableDensities[(tableDensities['Mask'] == mask) & (tableDensities['Density'] == d) & (tableDensities['NodeType'] == 'none')]['length=' + str(m)]]),
                                     'Group': (['lobe'] * len(tableDensities[(tableDensities['Mask'] == mask) & (tableDensities['Density'] == d) & (tableDensities['NodeType'] == 'lobe')]['length=' + str(m)]) + ['neck'] * len(tableDensities[(tableDensities['Mask'] == mask) & (tableDensities['Density'] == d) & (tableDensities['NodeType'] == 'neck')]['length=' + str(m)]) + ['none'] * len(tableDensities[(tableDensities['Mask'] == mask) & (tableDensities['Density'] == d) & (tableDensities['NodeType'] == 'none')]['length=' + str(m)]))})
               
                eval_var = pg.homoscedasticity(data=data, dv='Values', group='Group')['equal_var'].iloc[0]
                if eval_var:
                    aov = pg.anova(data=data, dv='Values', between='Group')['p-unc'].values[0]
                    posthoc = pg.pairwise_tukey(data=data, dv='Values', between='Group')['p-tukey'].values
                else:
                    aov = pg.welch_anova(data=data, dv='Values', between='Group')['p-unc'].values[0]
                    posthoc = pg.pairwise_gameshowell(data=data, dv='Values', between='Group')[ 'pval'].values

                dataAppend = [mask, d, m, format(aov, '.2e'), format(posthoc[0], '.2e'), format(posthoc[1], '.2e'), format(posthoc[2], '.2e')]
                pValues.loc[len(pValues)] = dataAppend
    
    
    tableFig2 = Table(title='P-values for different density mask sizes in lobe and neck regions')
    for col in pValues.columns:
        tableFig2.add_column(str(col))
    for _, row in pValues.iterrows():
        tableFig2.add_row(*[str(v) for v in row])

    console.print('[bold blue]Figure 2[/bold blue]')
    console.print(tableFig2)
 
###############################################################################
def get_lobe_neck_positions(graph):
    nodePos = nx.get_node_attributes(graph, 'pos')
    lobeIndices, neckIndices = GE.count_lobes_and_necks(graph)
    lobePos = {}
    neckPos = {}
    nodeIndices = list(graph.nodes)
    for nodeIdx in nodeIndices:
        if nodeIdx in lobeIndices:
            lobePos[nodeIdx] = nodePos[nodeIdx]
        if nodeIdx in neckIndices:
            neckPos[nodeIdx] = nodePos[nodeIdx]
    return(lobePos, neckPos)
    
    
if __name__ == '__main__':
    main()