#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 21:53:30 2026

@author: Jacqueline Nowak, JNowak@mpimp-golm.mpg.de
"""

import argparse
from pathlib import Path
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
    
###############################################################################
# =========================
# Figure 1
# Visualization of cytoskeleton organization in wild-type cotyledon pavement cells.
# =========================

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to dataset')
    args = parser.parse_args()
    
    pathRoot = Path(args.data)
    pathCotyledon = pathRoot / 'Cotyledons'

    pathPlots = pathRoot / 'Plots'
    pathPlots.mkdir(parents=True, exist_ok=True)
    
    
    ### import densities from tables
    globalDensities = pd.read_csv(pathCotyledon / 'globalDensities.csv')
    dataPerpPC = pd.read_csv(pathCotyledon / 'perpendicularDensities.csv')
    dataCircPC = pd.read_csv(pathCotyledon / 'circularDensities.csv')
    dataPolyPC = pd.read_csv(pathCotyledon / 'polygonalDensities.csv') 
    
    
    ### remove cells with no density in polygonal mask
    dataPolyPC = dataPolyPC[(dataPolyPC['densityInsidePolygon'] != 0) & (dataPolyPC['densityOutsidePolygon'] != 0)]

    
    ### plot densities of AFs and MTs in lobes and necks for each mask
    # variables for boxplot
    colors = ['#2ca02c', '#2ca02c', '#e377c2', '#e377c2']
    hatches = ['', '/', '', '/']
    
    fig, ax = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    # perpendicular mask
    bp1 = ax[0].boxplot([dataPerpPC[(dataPerpPC['Filament'] == 'AFs') & (dataPerpPC['Genotype'] == 'WT') & (dataPerpPC['NodeType'] == 'lobe')]['Density'],
                      dataPerpPC[(dataPerpPC['Filament'] == 'AFs') & (dataPerpPC['Genotype'] == 'WT') & (dataPerpPC['NodeType'] == 'neck')]['Density'], 
                      dataPerpPC[(dataPerpPC['Filament'] == 'MTs') & (dataPerpPC['Genotype'] == 'WT') & (dataPerpPC['NodeType'] == 'lobe')]['Density'], 
                      dataPerpPC[(dataPerpPC['Filament'] == 'MTs') & (dataPerpPC['Genotype'] == 'WT') & (dataPerpPC['NodeType'] == 'neck')]['Density']], positions=[0, 0.75, 2, 2.75], widths=0.6, patch_artist=True, showfliers=False)
    for element in ['boxes', 'whiskers', 'medians', 'means', 'fliers', 'caps']:
        plt.setp(bp1[element], color='black')
    for idx, box in enumerate(bp1['boxes']):
        box.set(facecolor=colors[idx])
        box.set(hatch=hatches[idx])
    ax[0].set_xticks([0.5, 2.5])
    ax[0].set_xticklabels(['Actin filaments', 'Microtubules'])
    # circular mask
    bp2 = ax[1].boxplot([dataCircPC[(dataCircPC['Filament'] == 'AFs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'lobe')]['Density'],
                      dataCircPC[(dataCircPC['Filament'] == 'AFs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'neck')]['Density'], 
                      dataCircPC[(dataCircPC['Filament'] == 'MTs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'lobe')]['Density'], 
                      dataCircPC[(dataCircPC['Filament'] == 'MTs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'neck')]['Density']], positions=[0, 0.75, 2, 2.75], widths=0.6, patch_artist=True, showfliers=False)
    for element in ['boxes', 'whiskers', 'medians', 'means', 'fliers', 'caps']:
        plt.setp(bp2[element], color='black')
    for idx, box in enumerate(bp2['boxes']):
        box.set(facecolor=colors[idx])
        box.set(hatch=hatches[idx])
    ax[1].set_xticks([0.5, 2.5])
    ax[1].set_xticklabels(['Actin filaments', 'Microtubules'])
    # polygonal mask
    bp3 = ax[2].boxplot([dataPolyPC[(dataPolyPC['Filament'] == 'AFs') & (dataPolyPC['Genotype'] == 'WT')]['densityOutsidePolygon'],
                      dataPolyPC[(dataPolyPC['Filament'] == 'AFs') & (dataPolyPC['Genotype'] == 'WT')]['densityInsidePolygon'], 
                      dataPolyPC[(dataPolyPC['Filament'] == 'MTs') & (dataPolyPC['Genotype'] == 'WT')]['densityOutsidePolygon'], 
                      dataPolyPC[(dataPolyPC['Filament'] == 'MTs') & (dataPolyPC['Genotype'] == 'WT')]['densityInsidePolygon']], positions=[0, 0.75, 2, 2.75], widths=0.6, patch_artist=True, showfliers=False)
    for element in ['boxes', 'whiskers', 'medians', 'means', 'fliers', 'caps']:
        plt.setp(bp3[element], color='black')
    for idx, box in enumerate(bp3['boxes']):
        box.set(facecolor=colors[idx])
        box.set(hatch=hatches[idx])
    ax[2].set_xticks([0.5, 2.5])
    ax[2].set_xticklabels(['Actin filaments', 'Microtubules'])
    # global mask
    bp4 = ax[3].boxplot([globalDensities[(globalDensities['Filament'] == 'AFs') & (globalDensities['Genotype'] == 'WT')]['Density'], globalDensities[(globalDensities['Filament'] == 'MTs') & (globalDensities['Genotype'] == 'WT')]['Density']], positions=[0, 1], widths=0.6, patch_artist=True, showfliers=False)
    for element in ['boxes', 'whiskers', 'medians', 'means', 'fliers', 'caps']:
        plt.setp(bp4[element], color='black')
    for idx, box in enumerate(bp4['boxes']):
        box.set(facecolor=colors[idx + 1])
    ax[3].set_xticks([0, 1])
    ax[3].set_xticklabels(['Actin filaments', 'Microtubules'])
    ax[0].set_ylim([-0.01, 0.2])
    ax[0].set_ylabel('Cytoskeleton density')
    plt.subplots_adjust(wspace=0.05)
    fig.savefig(pathPlots / 'Figure1C-F_Cotyledons_CytoskeletonDensityMasks_WT.png', bbox_inches='tight', dpi=300)               
    plt.close()
      
    
    ### statistical significance
    print('Numbers of samples:', 
          '\n...Perpencidular mask:',
          '\n......AFs:', dataPerpPC[(dataPerpPC['Filament'] == 'AFs') & (dataPerpPC['Genotype'] == 'WT') & (dataPerpPC['NodeType'] == 'lobe')].shape[0], 
          '\n......MTs:', dataPerpPC[(dataPerpPC['Filament'] == 'MTs') & (dataPerpPC['Genotype'] == 'WT') & (dataPerpPC['NodeType'] == 'lobe')].shape[0], 
          '\n...Circular mask:',
          '\n......AFs:', dataCircPC[(dataCircPC['Filament'] == 'AFs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'lobe')].shape[0], 
          '\n......MTs:', dataCircPC[(dataCircPC['Filament'] == 'MTs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'lobe')].shape[0], 
          '\n...Polygonal mask:',
          '\n......AFs:', dataPolyPC[(dataPolyPC['Filament'] == 'AFs') & (dataPolyPC['Genotype'] == 'WT')].shape[0], 
          '\n......MTs:', dataPolyPC[(dataPolyPC['Filament'] == 'MTs') & (dataPolyPC['Genotype'] == 'WT')].shape[0], 
          '\n...Global mask:',
          '\n......AFs:', globalDensities[(globalDensities['Filament'] == 'AFs') & (globalDensities['Genotype'] == 'WT')].shape[0], 
          '\n......MTs:', globalDensities[(globalDensities['Filament'] == 'MTs') & (globalDensities['Genotype'] == 'WT')].shape[0])
    
    print('Perpendicular mask (AFs):', sp.stats.mannwhitneyu(dataPerpPC[(dataPerpPC['Filament'] == 'AFs') & (dataPerpPC['Genotype'] == 'WT') & (dataPerpPC['NodeType'] == 'lobe')]['Density'], dataPerpPC[(dataPerpPC['Filament'] == 'AFs') & (dataPerpPC['Genotype'] == 'WT') & (dataPerpPC['NodeType'] == 'neck')]['Density'])[1])
    print('Perpendicular mask (MTs):', sp.stats.mannwhitneyu(dataPerpPC[(dataPerpPC['Filament'] == 'MTs') & (dataPerpPC['Genotype'] == 'WT') & (dataPerpPC['NodeType'] == 'lobe')]['Density'], dataPerpPC[(dataPerpPC['Filament'] == 'MTs') & (dataPerpPC['Genotype'] == 'WT') & (dataPerpPC['NodeType'] == 'neck')]['Density'])[1])
      
    print('Circular mask (AFs):', sp.stats.mannwhitneyu(dataCircPC[(dataCircPC['Filament'] == 'AFs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'lobe')]['Density'], dataCircPC[(dataCircPC['Filament'] == 'AFs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'neck')]['Density'])[1])
    print('Circular mask (MTs):', sp.stats.mannwhitneyu(dataCircPC[(dataCircPC['Filament'] == 'MTs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'lobe')]['Density'], dataCircPC[(dataCircPC['Filament'] == 'MTs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'neck')]['Density'])[1])
      
    print('Polygonal mask (AFs):', sp.stats.mannwhitneyu(dataPolyPC[(dataPolyPC['Filament'] == 'AFs') & (dataPolyPC['Genotype'] == 'WT')]['densityOutsidePolygon'], dataPolyPC[(dataPolyPC['Filament'] == 'AFs') & (dataPolyPC['Genotype'] == 'WT')]['densityInsidePolygon'])[1])
    print('Polygonal mask (MTs):', sp.stats.mannwhitneyu(dataPolyPC[(dataPolyPC['Filament'] == 'MTs') & (dataPolyPC['Genotype'] == 'WT')]['densityOutsidePolygon'], dataPolyPC[(dataPolyPC['Filament'] == 'MTs') & (dataPolyPC['Genotype'] == 'WT')]['densityInsidePolygon'])[1])

if __name__ == '__main__':
    main()