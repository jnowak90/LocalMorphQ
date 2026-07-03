#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:14:46 2026

@author: Jacqueline Nowak, JNowak@mpimp-golm.mpg.de
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import scipy as sp
import networkx as nx
import skimage
from skimage.morphology import disk
import statsmodels.formula.api as smf
import sklearn
import joypy
import ast
import pingouin as pg
import matplotlib
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
# Supplememtary Figures
#   Figure S1 - Different mask approaches for cytoskeleton analysis (cotyledon)
#   Figure S2 - Segmentation of wild-type leaves
#   Figure S3 - Segmentation of act2-1act7-1 leaves
#   Figure S4 - Time-series analysis of lobeyness (leaves)
#   Figure S5 - Growth rate analysis (leaves)
#   Table S1 - Cytoskeletal denisties of different masks (cotyledon)
#   Table S2 - Sample numbers in cotyledon
#   Table S3 - Comparison of ime-series data in leaves
#   Table S4 - Sample numbers in leaves
# =========================

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to dataset')
    args = parser.parse_args()
    
    pathRoot = Path(args.data)
    pathCotyledons = pathRoot / 'Cotyledons'
    pathLeaves = pathRoot / 'Leaves'
    
    pathPlots = pathRoot / 'Plots'
    pathPlots.mkdir(parents=True, exist_ok=True)
    
    pathPlotsSupplementary = pathPlots / 'Supplementary'
    pathPlotsSupplementary.mkdir(parents=True, exist_ok=True)
    
    
    ###########################################################################
    localDensitiesR = pd.read_csv(pathCotyledons / 'perpendicularDensities.csv')
    localDensitiesC = pd.read_csv(pathCotyledons / 'circularDensities.csv')
    localDensitiesP = pd.read_csv(pathCotyledons / 'polygonalDensities.csv')
    globalDensitiesC = pd.read_csv(pathCotyledons / 'globalDensities.csv')
    
    trackedLabels = pd.read_csv(pathLeaves / 'TrackedLabels.csv')
    trackedLabelsGR = pd.read_csv(pathLeaves / 'TrackedLabels_GR.csv')
    globalDensities = pd.read_csv(pathLeaves / 'globalDensities.csv')
    timepoints = [0, 12, 24, 36, 48]
    
    topO = plt.get_cmap('Oranges', 256)
    newcolorsO = np.vstack((topO(np.linspace(0, 1, 256))))
    newcolorsO[0] = np.array([1.0, 1.0, 1.0, 0.0])
    cmapCellsO = matplotlib.colors.ListedColormap(newcolorsO, name='Oranges')
    
    topR = plt.get_cmap('Spectral_r', 256)
    newcolorsR = np.vstack((topR(np.linspace(0, 1, 256))))
    newcolorsR[0] = np.array([1.0, 1.0, 1.0, 0.0])
    cmapCellsR = matplotlib.colors.ListedColormap(newcolorsR, name='Rainbow')
    
    cmapBinary = matplotlib.colors.ListedColormap([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    cmapBinaryRed = matplotlib.colors.ListedColormap([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]]) # red
    console = Console()
    
   
    ###############################
    ########## Figure S1 ##########
    ###############################   
    colors = ['#2ca02c', '#2ca02c', '#e377c2', '#e377c2']
    hatches = ['', '/', '', '/']
    
    fig, ax = plt.subplots(3, 3, sharey='row', figsize=(15, 15))
    ##### ACTIN DENSITY #####
    # AF density perpendicular mask
    bp11 = ax[0, 0].violinplot([extract_cell_data(localDensitiesR, 'WT', 'AFs', 'lobe'), 
                                extract_cell_data(localDensitiesR, 'WT', 'AFs', 'neck'),
                                extract_cell_data(localDensitiesR, 'act2-1act7-1', 'AFs', 'lobe'),
                                extract_cell_data(localDensitiesR, 'act2-1act7-1', 'AFs', 'neck')], positions=[0, 0.75, 2, 2.75], widths=0.55, showextrema=False, showmedians=True)
    ax[0, 0].set_xticks([0.375, 2.375])
    ax[0, 0].set_xticklabels(['WT', 'act2-1act7-1'])
    ax[0, 0].set_ylabel('Actin fiilament density')
    ax[0, 0].set_ylim([-0.01, 0.21])
    bp11['cmedians'].set_color('black')
    for idx, box in enumerate(bp11['bodies']):
        box.set_facecolor(colors[0])
        box.set_edgecolor('black')
        box.set(hatch=hatches[idx])
        box.set_alpha(1)
    # AF density circular mask    
    bp12 = ax[0, 1].violinplot([extract_cell_data(localDensitiesC, 'WT', 'AFs', 'lobe'), 
                                extract_cell_data(localDensitiesC, 'WT', 'AFs', 'neck'),
                                extract_cell_data(localDensitiesC, 'act2-1act7-1', 'AFs', 'lobe'),
                                extract_cell_data(localDensitiesC, 'act2-1act7-1', 'AFs', 'neck')], positions=[0, 0.75, 2, 2.75], widths=0.55, showextrema=False, showmedians=True)
    ax[0, 1].set_xticks([0.375, 2.375])
    ax[0, 1].set_xticklabels(['WT', 'act2-1act7-1'])
    ax[0, 1].set_ylim([-0.01, 0.21])
    bp12['cmedians'].set_color('black')
    for idx, box in enumerate(bp12['bodies']):
        box.set_facecolor(colors[0])
        box.set_edgecolor('black')
        box.set(hatch=hatches[idx])       
        box.set_alpha(1)
    # AF density polygonal mask    
    bp13 = ax[0, 2].violinplot([extract_cell_data(localDensitiesP, 'WT', 'AFs', 'outside'), 
                                extract_cell_data(localDensitiesP, 'WT', 'AFs', 'inside'),
                                extract_cell_data(localDensitiesP, 'act2-1act7-1', 'AFs', 'outside'),
                                extract_cell_data(localDensitiesP, 'act2-1act7-1', 'AFs', 'inside')], positions=[0, 0.75, 2, 2.75], widths=0.55, showextrema=False, showmedians=True)
    ax[0, 2].set_xticks([0.375, 2.375])
    ax[0, 2].set_xticklabels(['WT', 'act2-1act7-1'])
    ax[0, 2].set_ylim([-0.01, 0.21])
    bp13['cmedians'].set_color('black')
    for idx, box in enumerate(bp13['bodies']):
        box.set_facecolor(colors[0])
        box.set_edgecolor('black')
        box.set(hatch=hatches[idx])       
        box.set_alpha(1)        
    ###########################################################################
    ##### MICROTUBULE DENSITY #####
    # MT density perpendicular mask
    bp21 = ax[1, 0].violinplot([extract_cell_data(localDensitiesR, 'WT', 'MTs', 'lobe'), 
                                extract_cell_data(localDensitiesR, 'WT', 'MTs', 'neck'),
                                extract_cell_data(localDensitiesR, 'act2-1act7-1', 'AFs', 'lobe'),
                                extract_cell_data(localDensitiesR, 'act2-1act7-1', 'AFs', 'neck')], positions=[0, 0.75, 2, 2.75], widths=0.55, showextrema=False, showmedians=True)
    ax[1, 0].set_xticks([0.375, 2.375])
    ax[1, 0].set_xticklabels(['WT', 'act2-1act7-1'])
    ax[1, 0].set_ylabel('Microtubule density')
    ax[1, 0].set_ylim([-0.01, 0.21])
    bp21['cmedians'].set_color('black')
    for idx, box in enumerate(bp21['bodies']):
        box.set_facecolor(colors[2])
        box.set_edgecolor('black')
        box.set(hatch=hatches[idx])
        box.set_alpha(1)
    # MT density circular mask    
    bp22 = ax[1, 1].violinplot([extract_cell_data(localDensitiesC, 'WT', 'MTs', 'lobe'), 
                                extract_cell_data(localDensitiesC, 'WT', 'MTs', 'neck'),
                                extract_cell_data(localDensitiesC, 'act2-1act7-1', 'MTs', 'lobe'),
                                extract_cell_data(localDensitiesC, 'act2-1act7-1', 'MTs', 'neck')], positions=[0, 0.75, 2, 2.75], widths=0.55, showextrema=False, showmedians=True)
    ax[1, 1].set_xticks([0.375, 2.375])
    ax[1, 1].set_xticklabels(['WT', 'act2-1act7-1'])
    ax[1, 1].set_ylim([-0.01, 0.21])
    bp22['cmedians'].set_color('black')
    for idx, box in enumerate(bp22['bodies']):
        box.set_facecolor(colors[2])
        box.set_edgecolor('black')
        box.set(hatch=hatches[idx])       
        box.set_alpha(1)
    # MT density polygonal mask    
    bp23 = ax[1, 2].violinplot([extract_cell_data(localDensitiesP, 'WT', 'MTs', 'outside'), 
                                extract_cell_data(localDensitiesP, 'WT', 'MTs', 'inside'),
                                extract_cell_data(localDensitiesP, 'act2-1act7-1', 'MTs', 'outside'),
                                extract_cell_data(localDensitiesP, 'act2-1act7-1', 'MTs', 'inside')], positions=[0, 0.75, 2, 2.75], widths=0.55, showextrema=False, showmedians=True)
    ax[1, 2].set_xticks([0.375, 2.375])
    ax[1, 2].set_xticklabels(['WT', 'act2-1act7-1'])
    ax[1, 2].set_ylim([-0.01, 0.21])
    bp23['cmedians'].set_color('black')
    for idx, box in enumerate(bp23['bodies']):
        box.set_facecolor(colors[2])
        box.set_edgecolor('black')
        box.set(hatch=hatches[idx])       
        box.set_alpha(1)         
    ###########################################################################
    ##### DENSITY RATIO LOBES/NECKS #####        
    # MT density perpendicular mask
    bp31 = ax[2, 0].violinplot([extract_ratio_from_df(localDensitiesR[(localDensitiesR['Genotype'] =='WT') & (localDensitiesR['Filament'] == 'AFs')], 'perp')[0], 
                                extract_ratio_from_df(localDensitiesR[(localDensitiesR['Genotype'] =='act2-1act7-1') & (localDensitiesR['Filament'] == 'AFs')], 'perp')[0],
                                extract_ratio_from_df(localDensitiesR[(localDensitiesR['Genotype'] =='WT') & (localDensitiesR['Filament'] == 'MTs')], 'perp')[0],
                                extract_ratio_from_df(localDensitiesR[(localDensitiesR['Genotype'] =='act2-1act7-1') & (localDensitiesR['Filament'] == 'MTs')], 'perp')[0]], positions=[0, 0.75, 2, 2.75], widths=0.55, showextrema=False, showmedians=True)
    ax[2, 0].set_xticks([0, 0.75, 2, 2.75])
    ax[2, 0].set_xticklabels(2 * ['WT', 'act2-1act7-1'])
    ax[2, 0].set_ylabel('Cytoskeleton density lobes / necks')
    ax[2, 0].set_ylim([0, 2.1])
    ax[2, 0].set_xlim([-0.5, 3.25])
    ax[2, 0].plot([-0.5, 3.25], [1, 1], linestyle='dashed', color='gray')
    bp31['cmedians'].set_color('black')
    for idx, box in enumerate(bp31['bodies']):
        box.set_facecolor(colors[idx])
        box.set_edgecolor('black')
        box.set_alpha(1)
    # MT density circular mask    
    bp32 = ax[2, 1].violinplot([extract_ratio_from_df(localDensitiesC[(localDensitiesC['Genotype'] =='WT') & (localDensitiesC['Filament'] == 'AFs')], 'circ')[0], 
                                extract_ratio_from_df(localDensitiesC[(localDensitiesC['Genotype'] =='act2-1act7-1') & (localDensitiesC['Filament'] == 'AFs')], 'circ')[0],
                                extract_ratio_from_df(localDensitiesC[(localDensitiesC['Genotype'] =='WT') & (localDensitiesC['Filament'] == 'MTs')], 'circ')[0],
                                extract_ratio_from_df(localDensitiesC[(localDensitiesC['Genotype'] =='act2-1act7-1') & (localDensitiesC['Filament'] == 'MTs')], 'circ')[0]], positions=[0, 0.75, 2, 2.75], widths=0.55, showextrema=False, showmedians=True)
    ax[2, 1].set_xticks([0, 0.75, 2, 2.75])
    ax[2, 1].set_xticklabels(2 * ['WT', 'act2-1act7-1'])
    ax[2, 1].set_ylim([0, 2.1])
    ax[2, 1].set_xlim([-0.5, 3.25])
    ax[2, 1].plot([-0.5, 3.25], [1, 1], linestyle='dashed', color='gray')
    bp32['cmedians'].set_color('black')
    for idx, box in enumerate(bp32['bodies']):
        box.set_facecolor(colors[idx])
        box.set_edgecolor('black')      
        box.set_alpha(1)
    # MT density polygonal mask    
    bp33 = ax[2, 2].violinplot([extract_ratio_from_df(localDensitiesP[(localDensitiesP['Genotype'] =='WT') & (localDensitiesP['Filament'] == 'AFs')], 'poly')[0], 
                                extract_ratio_from_df(localDensitiesP[(localDensitiesP['Genotype'] =='act2-1act7-1') & (localDensitiesP['Filament'] == 'AFs')], 'poly')[0],
                                extract_ratio_from_df(localDensitiesP[(localDensitiesP['Genotype'] =='WT') & (localDensitiesP['Filament'] == 'MTs')], 'poly')[0],
                                extract_ratio_from_df(localDensitiesP[(localDensitiesP['Genotype'] =='act2-1act7-1') & (localDensitiesP['Filament'] == 'MTs')], 'poly')[0]], positions=[0, 0.75, 2, 2.75], widths=0.55, showextrema=False, showmedians=True)
    ax[2, 2].set_xticks([0, 0.75, 2, 2.75])
    ax[2, 2].set_xticklabels(2 * ['WT', 'act2-1act7-1'])
    ax[2, 2].set_ylim([0, 2.1])
    ax[2, 2].set_xlim([-0.5, 3.25])
    ax[2, 2].plot([-0.5, 3.25], [1, 1], linestyle='dashed', color='gray')
    bp33['cmedians'].set_color('black')
    for idx, box in enumerate(bp33['bodies']):
        box.set_facecolor(colors[idx])
        box.set_edgecolor('black')    
        box.set_alpha(1)                 
    plt.tight_layout()
    fig.savefig(pathPlotsSupplementary / 'FigureS1_Cotyledons_CytoskeletonDensityMasks_WT_Mutant.png', bbox_inches='tight', dpi=300)
    plt.close()    
    
    tableFigS1 = Table(title='P-values for different mask approaches of AF and MT in lobes and necks')
    tableFigS1.add_column('', justify='left')
    tableFigS1.add_column('Perpendicular mask', justify='center')
    tableFigS1.add_column('', justify='center')
    tableFigS1.add_column('Circular mask', justify='center')
    tableFigS1.add_column('', justify='center')
    tableFigS1.add_column('Polygonal mask', justify='center')
    tableFigS1.add_column('', justify='center')
    
    tableFigS1.add_row('', '[blue]WT[/blue]', '[blue]act2-1act7-1[/blue]', '[blue]WT[/blue]', '[blue]act2-1act7-1[/blue]', '[blue]WT[/blue]', '[blue]act2-1act7-1[/blue]')
    
    tableFigS1.add_row('AF density', 
                       str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesR, 'WT', 'AFs', 'lobe'), extract_cell_data(localDensitiesC, 'WT', 'AFs', 'neck'))[1], '.2e')), 
                       str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesR, 'act2-1act7-1', 'AFs', 'lobe'), extract_cell_data(localDensitiesC, 'act2-1act7-1', 'AFs', 'neck'))[1], '.2e')), 
                       str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesC, 'WT', 'AFs', 'lobe'), extract_cell_data(localDensitiesC, 'WT', 'AFs', 'neck'))[1], '.2e')), 
                       str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesR, 'act2-1act7-1', 'AFs', 'lobe'), extract_cell_data(localDensitiesC, 'act2-1act7-1', 'AFs', 'neck'))[1], '.2e')), 
                       str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesP, 'WT', 'AFs', 'outside'), extract_cell_data(localDensitiesP, 'WT', 'AFs', 'inside'))[1], '.2e')), 
                       str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesP, 'act2-1act7-1', 'AFs', 'outside'), extract_cell_data(localDensitiesP, 'act2-1act7-1', 'AFs', 'inside'))[1], '.2e')))
    
    tableFigS1.add_row('MT density', 
                       str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesR, 'WT', 'MTs', 'lobe'), extract_cell_data(localDensitiesC, 'WT', 'MTs', 'neck'))[1], '.2e')), 
                       str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesR, 'act2-1act7-1', 'MTs', 'lobe'), extract_cell_data(localDensitiesC, 'act2-1act7-1', 'MTs', 'neck'))[1], '.2e')), 
                       str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesC, 'WT', 'MTs', 'lobe'), extract_cell_data(localDensitiesC, 'WT', 'MTs', 'neck'))[1], '.2e')), 
                       str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesR, 'act2-1act7-1', 'MTs', 'lobe'), extract_cell_data(localDensitiesC, 'act2-1act7-1', 'MTs', 'neck'))[1], '.2e')), 
                       str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesP, 'WT', 'MTs', 'outside'), extract_cell_data(localDensitiesP, 'WT', 'MTs', 'inside'))[1], '.2e')), 
                       str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesP, 'act2-1act7-1', 'MTs', 'outside'), extract_cell_data(localDensitiesP, 'act2-1act7-1', 'MTs', 'inside'))[1], '.2e')))
    
    tableFigS1.add_row('', '', '', '', '', '', '')
    tableFigS1.add_row('', '[blue]AF[/blue]', '[blue]MT[/blue]', '[blue]AF[/blue]', '[blue]MT[/blue]', '[blue]AF[/blue]', '[blue]MT[/blue]')
    
    tableFigS1.add_row('Cytoskeleton density (lobes/necks)', 
                       str(format(sp.stats.mannwhitneyu(extract_ratio_from_df(localDensitiesR[(localDensitiesR['Genotype'] =='WT') & (localDensitiesR['Filament'] == 'AFs')], 'perp')[0], extract_ratio_from_df(localDensitiesR[(localDensitiesR['Genotype'] =='act2-1act7-1') & (localDensitiesR['Filament'] == 'AFs')], 'perp')[0])[1], '.2e')), 
                       str(format(sp.stats.mannwhitneyu(extract_ratio_from_df(localDensitiesR[(localDensitiesR['Genotype'] =='WT') & (localDensitiesR['Filament'] == 'MTs')], 'perp')[0], extract_ratio_from_df(localDensitiesR[(localDensitiesR['Genotype'] =='act2-1act7-1') & (localDensitiesR['Filament'] == 'MTs')], 'perp')[0])[1], '.2e')), 
                       str(format(sp.stats.mannwhitneyu(extract_ratio_from_df(localDensitiesC[(localDensitiesC['Genotype'] =='WT') & (localDensitiesC['Filament'] == 'AFs')], 'circ')[0], extract_ratio_from_df(localDensitiesC[(localDensitiesC['Genotype'] =='act2-1act7-1') & (localDensitiesC['Filament'] == 'AFs')], 'circ')[0])[1], '.2e')), 
                       str(format(sp.stats.mannwhitneyu(extract_ratio_from_df(localDensitiesC[(localDensitiesC['Genotype'] =='WT') & (localDensitiesC['Filament'] == 'MTs')], 'circ')[0], extract_ratio_from_df(localDensitiesC[(localDensitiesC['Genotype'] =='act2-1act7-1') & (localDensitiesC['Filament'] == 'MTs')], 'circ')[0])[1], '.2e')), 
                       str(format(sp.stats.mannwhitneyu(extract_ratio_from_df(localDensitiesP[(localDensitiesP['Genotype'] =='WT') & (localDensitiesP['Filament'] == 'AFs')], 'poly')[0], extract_ratio_from_df(localDensitiesP[(localDensitiesP['Genotype'] =='act2-1act7-1') & (localDensitiesP['Filament'] == 'AFs')], 'poly')[0])[1], '.2e')), 
                       str(format(sp.stats.mannwhitneyu(extract_ratio_from_df(localDensitiesP[(localDensitiesP['Genotype'] =='WT') & (localDensitiesP['Filament'] == 'MTs')], 'poly')[0], extract_ratio_from_df(localDensitiesP[(localDensitiesP['Genotype'] =='act2-1act7-1') & (localDensitiesP['Filament'] == 'MTs')], 'poly')[0])[1], '.2e')))
    
    console.print('[bold blue]Supplementary Figure 1[/bold blue]')
    console.print(tableFigS1)

    
    ###############################
    ########## Figure S2 ##########
    ###############################  

    filaments = ['AFs', 'MTs']
    replicates = ['R1', 'R2', 'R3']
    
    ### WT
    fig, ax = plt.subplots(6, 5, figsize=(18, 24))
    axes_flat = ax.flatten()
    idx = 0
    for fil in filaments:
        for rep in replicates:
            for tp in timepoints: 
                labeledImage = np.load(pathLeaves / fil / 'WT' / str(tp) / rep / 'labeledCells.npy')
                membraneImage = skimage.io.imread(pathLeaves / fil / 'WT' / str(tp) / rep / 'membrane.tif')
                dilatedImage = skimage.morphology.binary_dilation(labeledImage == 0, disk(2))
                
                ax = axes_flat[idx]
                ax.imshow(membraneImage, cmap='gray_r', zorder=0)
                ax.imshow(dilatedImage==1, cmap=cmapBinaryRed, alpha=1, zorder=2)
                ax.axis('off')
                if idx <= 4:
                    ax.set_title(str(timepoints[idx]))
                if idx in [0, 5, 10, 15, 20, 25]:
                    ax.text(-0.05, 0.5, fil + ' - ' + rep, transform=ax.transAxes, rotation=90, va='center', ha='right')
                idx += 1
    plt.tight_layout()               
    fig.savefig(pathPlotsSupplementary / 'FigureS2_ExtractedCells_WT.png', bbox_inches='tight', dpi=300)               
    plt.close()
    
    
    ###############################
    ########## Figure S3 ##########
    ###############################  

    ### act2-1act7-1
    fig, ax = plt.subplots(6, 5, figsize=(18, 24))
    axes_flat = ax.flatten()
    idx = 0
    for fil in filaments:
        for rep in replicates:
            for tp in timepoints: 
                labeledImage = np.load(pathLeaves / fil / 'act2-1act7-1' / str(tp) / rep / 'labeledCells.npy')
                membraneImage = skimage.io.imread(pathLeaves / fil / 'act2-1act7-1' / str(tp) / rep / 'membrane.tif')
                dilatedImage = skimage.morphology.binary_dilation(labeledImage == 0, disk(2))
                
                ax = axes_flat[idx]
                ax.imshow(membraneImage, cmap='gray_r', zorder=0)
                ax.imshow(dilatedImage==1, cmap=cmapBinaryRed, alpha=1, zorder=2)
                ax.axis('off')
                if idx <= 4:
                    ax.set_title(str(timepoints[idx]))
                if idx in [0, 5, 10, 15, 20, 25]:
                    ax.text(-0.05, 0.5, fil + ' - ' + rep, transform=ax.transAxes, rotation=90, va='center', ha='right')
                idx += 1
    plt.tight_layout()               
    fig.savefig(pathPlotsSupplementary / 'FigureS3_ExtractedCells_act2-1act7-1.png', bbox_inches='tight', dpi=300)               
    plt.close()
    
    
    ###############################
    ########## Figure S4 ##########
    ###############################  
    
    ### create heatmaps of relative completeness in leaves for WT and act2-1act7-1
    cells = {'WT': {12: [13, 16, 18, 19, 20, 22, 23, 26, 27, 29, 30, 32, 33, 36, 37, 39, 43, 45], 
              24: [7, 13, 15, 17, 18, 20, 21, 22, 24, 25, 26, 27, 30, 31, 33, 35, 37, 39, 40, 43, 45, 46, 48, 49], 
              36: [1, 6, 8, 9, 10, 11, 12, 13, 15, (17, 18), 19, 20, 21, 23, 24, 25, 26, 28, 29, 30, 31, 33, 37, 38, 39, 40], 
              48: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, (16, 18), (17, 20, 19), 21, 22, (23, 25), 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]}, 
               'M': {0: [8, 10, 11, 13, 14, 15, 17, 19, 20, 24, 25, 26, 27, 28, 29, 30], 
              12: [7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], 
              24: [2, (3, 6), 7, 8, 10, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 28, 30, 31, 32, 33, 34, 35, (36, 37)], 
              36: [1, 3, 4, (5, 7), 9, 10, (11, 12), 13, 14, (15, 17), 16, 18, 19, 21, 22, 23, 25, 28, 29, (31, 32), 33, 34, 36, 37, 38, (39, 41)]}}
        
    for gt in cells.keys():
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        for idx, tp in enumerate(cells[gt].keys()):
            if gt == 'WT':
                resolution = 1 / 10.9051
                name = 'FigureS4A_Leaves_LobeynessHeatmap_WT.png'
            else:
                resolution = 1 / 6.5445
                name = 'FigureS4B_Leaves_LobeynessHeatmap_Mutant.png'
            imageDilated, labeledImageSelected, labelsSelected = extract_selected_cells(cells, gt, tp, pathLeaves)    
            lobeyness = extract_selected_cell_properties(labeledImageSelected, labelsSelected, resolution, 'lobeyness')
             
            imageColored_L = labeledImageSelected.copy() * 0
            imageColored_L = imageColored_L.astype(float)
            imageColored_L[0, 0] = 1.0
            for label in range(len(lobeyness)):
                coord = np.transpose(np.where(labeledImageSelected == label + 2))
                l = (lobeyness[label] - 0.8) / 0.3           
                for x, y in coord:
                    imageColored_L[x, y] = l
            im = ax[idx].imshow(imageColored_L, cmap=cmapCellsO, zorder=0)
            ax[idx].imshow(imageDilated==1, cmap=cmapBinary, alpha=1, zorder=2)
            ax[idx].axis('off')
        cbar = fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
        cbar.set_label('Lobeyness', rotation=90, labelpad=20)
        plt.tight_layout()
        fig.savefig(pathPlotsSupplementary / name, bbox_inches='tight', dpi=300) 
        plt.close()

    
    ###########################################################################
    dataTrackedCells = pd.DataFrame(columns=['Genotype', 'Filament', 'Replicate', 'Property', '0h', '12h', '24h', '36h'])
    tps = ['0', '12', '24', '36', '48']

    # extract properties for tracked cells from global properties
    for idx in range(len(trackedLabels)):
        properties = globalDensities[(globalDensities['Filament'] == trackedLabels['Filament'][idx]) & (globalDensities['Genotype'] == trackedLabels['Genotype'][idx]) & (globalDensities['Replicate'] == trackedLabels['Replicate'][idx])]

        if trackedLabels['Genotype'][idx] == 'WT':
            selectedCells = [ast.literal_eval(n) if isinstance(n, str) else int(n) for n in trackedLabels.iloc[idx, 4:]]
            tps = [12, 24, 36, 48]
        else:
            selectedCells = [ast.literal_eval(n) if isinstance(n, str) else int(n) for n in trackedLabels.iloc[idx, 3:-1]]
            tps = [0, 12, 24, 36]
        
        dataArea, dataLobeyness, dataLobes = [], [], []
        for jdx, cell in enumerate(selectedCells):
            tp = tps[jdx]
            propertiesCell = properties[(properties['CellNumber'] == str(cell)) & (properties['Timepoint'] == tp)].reset_index(drop=True)
            if propertiesCell.shape[0] != 0:
                lobes, area, lobeyness = propertiesCell.iloc[0, [10, 11, 14]]
                dataArea.append(area)
                dataLobeyness.append(lobeyness)
                dataLobes.append(lobes)
                        
        dataAppend = [trackedLabels['Genotype'][idx], trackedLabels['Filament'][idx], trackedLabels['Replicate'][idx], 'Area']
        dataAppend.extend(dataArea)
        dataTrackedCells.loc[len(dataTrackedCells)] = dataAppend
        
        dataAppend = [trackedLabels['Genotype'][idx], trackedLabels['Filament'][idx], trackedLabels['Replicate'][idx], 'Lobeyness']
        dataAppend.extend(dataLobeyness)
        dataTrackedCells.loc[len(dataTrackedCells)] = dataAppend
                
        dataAppend = [trackedLabels['Genotype'][idx], trackedLabels['Filament'][idx], trackedLabels['Replicate'][idx], 'Lobes']
        dataAppend.extend(dataLobes)
        dataTrackedCells.loc[len(dataTrackedCells)] = dataAppend
    
    
    ### relative completeness distributions for WT and mutant (tracked cells)
    # extract data from cells tracked over all time points
    lobeynessWT_tracked = dataTrackedCells[(dataTrackedCells['Property'] == 'Lobeyness') & (dataTrackedCells['Genotype'] == 'WT')].reset_index(drop=True)
    lobeynessM_tracked = dataTrackedCells[(dataTrackedCells['Property'] == 'Lobeyness') & (dataTrackedCells['Genotype'] == 'act2-1act7-1')].reset_index(drop=True)

    # use only data from matching time points between WT and mutant
    dataLobeynessWT = [lobeynessWT_tracked['0h'], lobeynessWT_tracked['12h'], lobeynessWT_tracked['24h'], lobeynessWT_tracked['36h']]
    dataLobeynessM = [lobeynessM_tracked['0h'], lobeynessM_tracked['12h'],  lobeynessM_tracked['24h'], lobeynessM_tracked['36h']]

    # create data frame for ridge plot
    df_t = create_df_for_ridgeplot(dataLobeynessWT, dataLobeynessM) 

    # plot relative completeness distributions as ridge plots
    fig, ax = joypy.joyplot(df_t, by='Timepoint', color=['cornflowerblue', 'gold'], alpha=0.5, figsize=(5, 5), overlap=0.25)
    plt.xlabel('Lobeyness')
    plt.ylabel('Timepoint')
    # plot means
    for i in range(4):
        ymin = 0
        meanWT = dataLobeynessWT[i].mean()
        meanM = dataLobeynessM[i].mean()
        yvalWT = ax[i].lines[1].get_ydata()[np.where(np.round(ax[i].lines[1].get_xdata(), 2) == np.around(meanWT, 2))[0][0]]
        yvalM = ax[i].lines[3].get_ydata()[np.where(np.round(ax[i].lines[3].get_xdata(), 2) == np.around(meanM, 2))[0][0]]
        ax[i].plot([meanWT, meanWT], [ymin, yvalWT], color='gray', ls='-', zorder=200)
        ax[i].plot([meanM, meanM], [ymin, yvalM], color='gray', ls='--', zorder=200)
    fig.savefig(pathPlotsSupplementary / 'FigureS4C_Leaves_LobeynessDistributions_WT_Mutant.png', bbox_inches='tight', dpi=300)               
    plt.close()
    
    # statistics
    console.print('[bold blue]Supplementary Figure 4[/bold blue]')
    print('Lobeyness (WT vs. M): \n...0h:', sp.stats.mannwhitneyu(dataLobeynessWT[0], dataLobeynessM[0])[1], '\n...12h:', sp.stats.mannwhitneyu(dataLobeynessWT[1], dataLobeynessM[1])[1], '\n...24h:', sp.stats.mannwhitneyu(dataLobeynessWT[2], dataLobeynessM[2])[1], '\n...36h:', sp.stats.mannwhitneyu(dataLobeynessWT[3], dataLobeynessM[3])[1])
    
    
    ### lobeyness vs. true lobes (tracked cells)
    lobesWT_tracked = dataTrackedCells[(dataTrackedCells['Property'] == 'Lobes') & (dataTrackedCells['Genotype'] == 'WT')].reset_index(drop=True)
    lobesM_tracked = dataTrackedCells[(dataTrackedCells['Property'] == 'Lobes') & (dataTrackedCells['Genotype'] == 'act2-1act7-1')].reset_index(drop=True)

    dataLobesWT = [lobesWT_tracked['0h'], lobesWT_tracked['12h'], lobesWT_tracked['24h'], lobesWT_tracked['36h']]
    dataLobesM = [lobesM_tracked['0h'], lobesM_tracked['12h'], lobesM_tracked['24h'], lobesM_tracked['36h']]
                
    # correlation plot completeness ~ lobes
    sizes = [30, 60, 90, 120]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.scatter([dataLobesWT[idx].mean() for idx in range(4)], [dataLobeynessWT[idx].mean() for idx in range(4)], c='cornflowerblue', s=sizes)
    plt.errorbar([dataLobesWT[idx].mean() for idx in range(4)], [dataLobeynessWT[idx].mean() for idx in range(4)], yerr=[[dataLobeynessWT[idx].mean() - dataLobeynessWT[idx].quantile(q=0.25) for idx in range(4)], [dataLobeynessWT[idx].quantile(q=0.75) - dataLobeynessWT[idx].mean() for idx in range(4)]], xerr=[[dataLobesWT[idx].mean() - dataLobesWT[idx].quantile(q=0.25) for idx in range(4)], [np.maximum(0, dataLobesWT[idx].quantile(q=0.75) - dataLobesWT[idx].mean()) for idx in range(4)]], fmt = 'o', color = 'cornflowerblue', linewidth=2, capsize=3, alpha=0.5)
    plt.scatter([dataLobesM[idx].mean() for idx in range(4)], [dataLobeynessM[idx].mean() for idx in range(4)], c='gold', s=sizes)
    plt.errorbar([dataLobesM[idx].mean() for idx in range(4)], [dataLobeynessM[idx].mean() for idx in range(4)], yerr=[[dataLobeynessM[idx].mean() - dataLobeynessM[idx].quantile(q=0.25) for idx in range(4)], [dataLobeynessM[idx].quantile(q=0.75) - dataLobeynessM[idx].mean() for idx in range(4)]], xerr=[[dataLobesM[idx].mean() - dataLobesM[idx].quantile(q=0.25) for idx in range(4)], [dataLobesM[idx].quantile(q=0.75) - dataLobesM[idx].mean() for idx in range(4)]], fmt = 'o', color = 'gold', linewidth=2, capsize=3, alpha=0.5)
    ax.set_ylabel('Lobeyness')
    ax.set_xlabel('Number of true lobes')
    fig.savefig(pathPlotsSupplementary / 'FigureS4D_Leaves_LobeynessLobes_WT_Mutant.png', bbox_inches='tight', dpi=300)               
    plt.close()

    # statistical significance (Hotelling's T2 test)
    result0, result12, result24, result36 = get_Hotelling_pvalues(dataLobesWT, dataLobeynessWT, dataLobesM, dataLobeynessM)
    print("\nLobeyness vs. lobes (WT vs. M): \n...Oh:", result0['pval'].iloc[0], '\n...12h:', result12['pval'].iloc[0], '\n...24h:', result24['pval'].iloc[0], '\n...36h', result36['pval'].iloc[0])  
    
    
    ### lobeyness vs. cell area (all cells)
    areaWT_tracked = dataTrackedCells[(dataTrackedCells['Property'] == 'Area') & (dataTrackedCells['Genotype'] == 'WT')].reset_index(drop=True)
    areaM_tracked = dataTrackedCells[(dataTrackedCells['Property'] == 'Area') & (dataTrackedCells['Genotype'] == 'act2-1act7-1')].reset_index(drop=True)
    
    xWT_reg, xM_reg, yWT_reg, yM_reg = extract_regression_data(areaWT_tracked, lobeynessWT_tracked, areaM_tracked, lobeynessM_tracked)
  
    slopeWT, slopeM, slopePvalue, slopeSE, rWT, rM = get_regression_data(xWT_reg, xM_reg, yWT_reg, yM_reg)
    print('\nLobeyness vs. area (WT vs. M)')
    print('Slope_WT: ', slopeWT, ', Slope_M:', slopeM, ', P-value:', slopePvalue, ', SE:', slopeSE)
    print('...Pearson correlation coefficient: R_WT:', rWT, ', R_M:', rM, '\n')
    
    xWT_lin, xM_lin, yWT_lin, yM_lin, yWT_err, yM_err = create_linear_spaced_data(xWT_reg, xM_reg, slopeWT, slopeM)
    
    # correlation plot completeness ~ area
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(xWT_reg, yWT_reg, c='cornflowerblue', label='WT', s=25)
    ax.plot(xWT_lin, yWT_lin, c='cornflowerblue', linewidth=2)
    ax.fill_between(xWT_lin, yWT_lin - yWT_err, yWT_lin + yWT_err, color='cornflowerblue', alpha=0.2)
    ax.scatter(xM_reg, yM_reg, c='gold', label='act2-1act7-1', s=25)
    ax.plot(xM_lin, yM_lin, c='gold', linewidth=2)
    ax.fill_between(xM_lin, yM_lin - yM_err, yM_lin + yM_err, color='gold', alpha=0.2)
    ax.set_ylabel('Lobeyness')
    ax.set_xlabel('log(Area_sc)')
    plt.legend()
    fig.savefig(pathPlotsSupplementary / 'FigureS4E_Leaves_LobeynessArea_WT_Mutant.png', bbox_inches='tight', dpi=300)               
    plt.close()   
    
    
    ###############################
    ########## Figure S5 ##########
    ###############################

    ### calculate AGR and RGR from selected cell areas
    resWT, resM = 1 / 10.9051, 1 / 6.5445
    tableAreas = pd.DataFrame(columns=['Genotype', 'Area0h', 'Area12h', 'Area24h', 'Area36h', 'AGR0-24h', 'AGR12-36h', 'RGR0-24h', 'RGR12-36h'])
    
    for idx in range(len(trackedLabelsGR)):
        if trackedLabelsGR['Genotype'][idx] == 'WT':
            dataAppend = ['WT']
            for jdx, tp in enumerate([12, 24, 36, 48]):
                labeledImageWT = np.load(pathLeaves / 'MTs' / 'WT' / str(tp) / 'R3' / 'labeledCells.npy')
                cell = trackedLabelsGR.iloc[idx, jdx + 1]
                if isinstance(cell, str):
                    if '(' in cell:
                        cellTuple = tuple([int(x) for x in cell.replace('(', '').replace(')', '').split(',')])      
                        cellImage = merge_cells_from_labels(cellTuple, labeledImageWT)
                        area = np.sum(cellImage == 1) * (resWT ** 2)
                    else:
                        area = np.sum(labeledImageWT == int(cell) + 1) * (resWT ** 2)
                else:
                    area = np.sum(labeledImageWT == int(cell) + 1) * (resWT ** 2)
                dataAppend.append(area)
                            
            agr0_24 = (dataAppend[3] - dataAppend[1]) / 24
            agr12_36 = (dataAppend[4] - dataAppend[2]) / 24
            rgr0_24 = (np.log(dataAppend[3]) - np.log(dataAppend[1])) / 24
            rgr12_36 = (np.log(dataAppend[4]) - np.log(dataAppend[2])) / 24
            dataAppend.extend([agr0_24, agr12_36, rgr0_24, rgr12_36])
            tableAreas.loc[len(tableAreas)] = dataAppend
        else:
            dataAppend = ['act2-1act7-1']
            for jdx, tp in enumerate([0, 12, 24, 36]):
                labeledImageM = np.load(pathLeaves / 'AFs' / 'act2-1act7-1' / str(tp) / 'R2' / 'labeledCells.npy')
                cell = trackedLabelsGR.iloc[idx, jdx + 1]
                if isinstance(cell, str):
                    if '(' in cell:
                        cellTuple = tuple([int(x) for x in cell.replace('(', '').replace(')', '').split(',')])      
                        cellImage = merge_cells_from_labels(cellTuple, labeledImageM)
                        area = np.sum(cellImage == 1) * (resM ** 2)
                    else:
                        area = np.sum(labeledImageM == int(cell) + 1) * (resM ** 2)
                else:
                    area = np.sum(labeledImageM == int(cell) + 1) * (resM ** 2)
                dataAppend.append(area)
                            
            agr0_24 = (dataAppend[3] - dataAppend[1]) / 24
            agr12_36 = (dataAppend[4] - dataAppend[2]) / 24
            rgr0_24 = (np.log(dataAppend[3]) - np.log(dataAppend[1])) / 24
            rgr12_36 = (np.log(dataAppend[4]) - np.log(dataAppend[2])) / 24
            dataAppend.extend([agr0_24, agr12_36, rgr0_24, rgr12_36])
            tableAreas.loc[len(tableAreas)] = dataAppend
            

    ### cell outlines 
    # WT > MTs > R3, act2-1act7-1 > AFs > R2
    cellsGR = {'WT':
              {24: [7, 13, 15, 17, 18, 20, 21, 22, 24, 25, 26, 27, 30, 31, 33, 35, 37, 40, 43], 
              48: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, (16, 18), (17, 20, 19), 21, 22, (23, 25), 24, 26, 27, 28, 29, 30, 31, 32, 33, 37, 38, 42]},  
               'act2-1act7-1': 
                   {12: [9, 10, 11, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32], 
              36: [1, 3, 4, (5, 7), 9, 10, (11, 12), 13, 14, (15, 17), 16, 21, 22, 23, 25, 28, 29, (31, 32), 33, 34, 36, 37, 38, (39, 41)]}}


    fig, ax = plt.subplots(2, 4, figsize=(8, 5))
    for gt in cellsGR.keys():
        for idx, tp in enumerate(cellsGR[gt].keys()):
            if gt == 'WT':
                labeledImage = np.load(pathLeaves / 'MTs' / 'WT' / str(tp) / 'R3' / 'labeledCells.npy')
            else:
                labeledImage = np.load(pathLeaves / 'AFs' / 'act2-1act7-1' / str(tp) / 'R2' / 'labeledCells.npy')
            imageDilated, labeledImageSelected, labelsSelected = extract_selected_cells(cellsGR, gt, tp, pathLeaves)    
            dataArea = tableAreas[tableAreas['Genotype'] == gt].reset_index(drop=True)
            dataLabels = trackedLabelsGR[trackedLabelsGR['Genotype'] == gt].reset_index(drop=True)
            if tp in [12, 24]:
                agr = list(dataArea['AGR0-24h'])
                rgr = list(dataArea['RGR0-24h'])
                labelsTracked = list(dataLabels['12h'])
            else:
                agr = list(dataArea['AGR12-36h'])
                rgr = list(dataArea['RGR12-36h'])
                labelsTracked = list(dataLabels['36h'])
            
            # absolute growth rate
            imageColored_A = labeledImageSelected.copy() * 0
            imageColored_A = imageColored_A.astype(float)
            imageColored_A[0, 0] = 1.0
            for label in range(len(agr)):
                if isinstance(labelsTracked[label], str):
                    if '(' in cell:
                        cellTuple = tuple([int(x) for x in labelsTracked[label].replace('(', '').replace(')', '').split(',')])
                        cellImage = merge_cells_from_labels(cellTuple, labeledImage)
                        coord = np.transpose(np.where(cellImage == 1))
                    else:
                        coord = np.transpose(np.where(labeledImage == int(labelsTracked[label]) + 1))              
                else:
                    coord = np.transpose(np.where(labeledImage == label + 1))
                a = (agr[label] - 1) / 15           
                for x, y in coord:
                    imageColored_A[x, y] = a
            if gt == 'WT':
                imA = ax[0, idx].imshow(imageColored_A, cmap=cmapCellsR, zorder=0)
                ax[0, idx].imshow(imageDilated==1, cmap=cmapBinary, alpha=1, zorder=2)
                ax[0, idx].axis('off')
            else:
                imA = ax[0, idx + 2].imshow(imageColored_A, cmap=cmapCellsR, zorder=0)
                ax[0, idx + 2].imshow(imageDilated==1, cmap=cmapBinary, alpha=1, zorder=2)
                ax[0, idx + 2].axis('off')                
            
            # relative growth rate
            imageColored_R = labeledImageSelected.copy() * 0
            imageColored_R = imageColored_R.astype(float)
            imageColored_R[0, 0] = 1.0
            for label in range(len(rgr)):
                if isinstance(labelsTracked[label], str):
                    if '(' in cell:
                        cellTuple = tuple([int(x) for x in labelsTracked[label].replace('(', '').replace(')', '').split(',')])
                        cellImage = merge_cells_from_labels(cellTuple, labeledImage)
                        coord = np.transpose(np.where(cellImage == 1))
                    else:
                        coord = np.transpose(np.where(labeledImage == int(labelsTracked[label]) + 1))              
                else:
                    coord = np.transpose(np.where(labeledImage == label + 1))
                r = (rgr[label] - 0.01) / 0.05      
                for x, y in coord:
                    imageColored_R[x, y] = r
            if gt == 'WT':
                imR = ax[1, idx].imshow(imageColored_R, cmap=cmapCellsR, zorder=0)
                ax[1, idx].imshow(imageDilated==1, cmap=cmapBinary, alpha=1, zorder=2)
                ax[1, idx].axis('off')
            else:
                imR = ax[1, idx + 2].imshow(imageColored_R, cmap=cmapCellsR, zorder=0)
                ax[1, idx + 2].imshow(imageDilated==1, cmap=cmapBinary, alpha=1, zorder=2)
                ax[1, idx + 2].axis('off')
                        
    cbarA = fig.colorbar(imA, ax=ax[0, 3], fraction=0.046, pad=0.04)
    cbarA.set_label('Absolute growth rate [h^-1]', rotation=90, labelpad=20)
    cbarR = fig.colorbar(imR, ax=ax[1, 3], fraction=0.046, pad=0.04)
    cbarR.set_label('Relative growth rate [h^-1]', rotation=90, labelpad=20)    
    plt.tight_layout()
    fig.savefig(pathPlotsSupplementary / 'FigureS5AB_Leaves_AGR_RGR_Heatmaps_WT_Mutant.png', bbox_inches='tight', dpi=300) 
    plt.close()
        
    #######
    agr0_24 = {'WT': [], 'act2-1act7-1': []}     
    rgr0_24 = {'WT': [], 'act2-1act7-1': []}  
    agr12_36 = {'WT': [], 'act2-1act7-1': []}     
    rgr12_36 = {'WT': [], 'act2-1act7-1': []}  

    dataArea = dataTrackedCells[dataTrackedCells['Property'] == 'Area'].reset_index(drop=True)

    # AGR
    for idx in range(len(dataArea)):
        gr0_24 = (dataArea['24h'][idx] - dataArea['0h'][idx]) / 24
        gr12_36 = (dataArea['36h'][idx] - dataArea['12h'][idx]) / 24
        if dataArea['Genotype'][idx] == 'WT':
            agr0_24['WT'].append(gr0_24)    
            agr12_36['WT'].append(gr12_36)
        else:
            agr0_24['act2-1act7-1'].append(gr0_24)    
            agr12_36['act2-1act7-1'].append(gr12_36)

    # RGR
    for idx in range(len(dataArea)):
        gr0_24 = (np.log(dataArea['24h'][idx]) - np.log(dataArea['0h'][idx])) / 24
        gr12_36 = (np.log(dataArea['36h'][idx]) - np.log(dataArea['12h'][idx])) / 24
        if dataArea['Genotype'][idx] == 'WT':
            rgr0_24['WT'].append(gr0_24)
            rgr12_36['WT'].append(gr12_36)
        else:
            rgr0_24['act2-1act7-1'].append(gr0_24)    
            rgr12_36['act2-1act7-1'].append(gr12_36)              
 
    colors = ['cornflowerblue', 'gold', 'cornflowerblue', 'gold']
    fig, ax = plt.subplots(1, 2, figsize=(10, 5)) 
    bp1 = ax[0].violinplot([agr0_24['WT'], agr0_24['act2-1act7-1'], agr12_36['WT'], agr12_36['act2-1act7-1']], positions=[0, 0.8, 2, 2.8], widths=0.6, showextrema=False, showmedians=True)
    ax[0].set_xticks([0.4, 2.4])
    ax[0].set_xticklabels(['0-24h', '12-36h'])
    ax[0].set_ylabel('Absolute growth rate [h^-1]')
    bp1['cmedians'].set_color('black')
    for idx, box in enumerate(bp1['bodies']):
        box.set_facecolor(colors[idx])
        box.set_edgecolor('black')
        box.set_alpha(1)
    ax[0].set_ylim(1.5, 20)
    bp2 = ax[1].violinplot([rgr0_24['WT'], rgr0_24['act2-1act7-1'], rgr12_36['WT'], rgr12_36['act2-1act7-1']], positions=[0, 0.8, 2, 2.8], widths=0.6, showextrema=False, showmedians=True)
    ax[1].set_xticks([0.4, 2.4])
    ax[1].set_xticklabels(['0-24h', '12-36h'])
    bp2['cmedians'].set_color('black')
    for idx, box in enumerate(bp2['bodies']):
        box.set_facecolor(colors[idx])
        box.set_edgecolor('black')
        box.set_alpha(1)
    ax[1].set_ylabel('Relative growth rate [h^-1]')
    ax[1].set_ylim(0.01, 0.06)
    fig.savefig(pathPlotsSupplementary / 'FigureS5C_Leaves_AGR_RGR_Boxplots_WT_Mutant.png', bbox_inches='tight', dpi=300)
    plt.close()

    # statistical test
    console.print('[bold blue]Supplementary Figure 5[/bold blue]')
    print('\nAbsolute growth rate')
    print('...P-value (0-24h):', sp.stats.ttest_ind(agr0_24['WT'], agr0_24['act2-1act7-1'])[1])
    print('...P-value (12-36h):', sp.stats.ttest_ind(agr12_36['WT'], agr12_36['act2-1act7-1'])[1])

    print('\nRelative growth rate')
    print('...P-value (0-24h):', sp.stats.ttest_ind(rgr0_24['WT'], rgr0_24['act2-1act7-1'])[1])
    print('...P-value (12-36h):', sp.stats.ttest_ind(rgr12_36['WT'], rgr12_36['act2-1act7-1'])[1], '\n')
    
    
    ##############################
    ########## Table S1 ##########
    ##############################
    tableS1 = Table(title='Statistical comparison of cytoskeletal densities in lobe and neck regions.')
    tableS1.add_column('', justify='left')
    tableS1.add_column('AFs (perp)', justify='center')
    tableS1.add_column('MTs (perp)', justify='center')
    tableS1.add_column('AFs (circ)', justify='center')
    tableS1.add_column('MTs (circ)', justify='center')
    tableS1.add_column('AFs (poly)', justify='center')
    tableS1.add_column('MTs (poly)', justify='center')
    tableS1.add_column('AFs (global)', justify='center')
    tableS1.add_column('MTs (global)', justify='center')
    
    tableS1.add_row('Wild-type', 
                    str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesR, 'WT', 'AFs', 'lobe'), extract_cell_data(localDensitiesR, 'WT', 'AFs', 'neck'))[1], '.2e')), 
                    str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesR, 'WT', 'MTs', 'lobe'), extract_cell_data(localDensitiesR, 'WT', 'MTs', 'neck'))[1], '.2e')),
                    str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesC, 'WT', 'AFs', 'lobe'), extract_cell_data(localDensitiesC, 'WT', 'AFs', 'neck'))[1], '.2e')),
                    str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesC, 'WT', 'MTs', 'lobe'), extract_cell_data(localDensitiesC, 'WT', 'MTs', 'neck'))[1], '.2e')),
                    str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesP, 'WT', 'AFs', 'inside'), extract_cell_data(localDensitiesP, 'WT', 'AFs', 'outside'))[1], '.2e')),
                    str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesP, 'WT', 'MTs', 'inside'), extract_cell_data(localDensitiesP, 'WT', 'MTs', 'outside'))[1], '.2e')),
                    str(format(sp.stats.mannwhitneyu(extract_cell_data(globalDensitiesC, 'WT', 'AFs', None), extract_cell_data(globalDensitiesC, 'act2-1act7-1', 'AFs', None))[1], '.2e')),
                    str(format(sp.stats.mannwhitneyu(extract_cell_data(globalDensitiesC, 'WT', 'MTs', None), extract_cell_data(globalDensitiesC, 'act2-1act7-1', 'MTs', None))[1], '.2e')))
    
    tableS1.add_row('[italic]act2-1act7-1[/italic]',
                    str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesR, 'act2-1act7-1', 'AFs', 'lobe'), extract_cell_data(localDensitiesR, 'act2-1act7-1', 'AFs', 'neck'))[1], '.2e')), 
                    str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesR, 'act2-1act7-1', 'MTs', 'lobe'), extract_cell_data(localDensitiesR, 'act2-1act7-1', 'MTs', 'neck'))[1], '.2e')),
                    str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesC, 'act2-1act7-1', 'AFs', 'lobe'), extract_cell_data(localDensitiesC, 'act2-1act7-1', 'AFs', 'neck'))[1], '.2e')),
                    str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesC, 'act2-1act7-1', 'MTs', 'lobe'), extract_cell_data(localDensitiesC, 'act2-1act7-1', 'MTs', 'neck'))[1], '.2e')),
                    str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesP, 'act2-1act7-1', 'AFs', 'inside'), extract_cell_data(localDensitiesP, 'act2-1act7-1', 'AFs', 'outside'))[1], '.2e')),
                    str(format(sp.stats.mannwhitneyu(extract_cell_data(localDensitiesP, 'act2-1act7-1', 'MTs', 'inside'), extract_cell_data(localDensitiesP, 'act2-1act7-1', 'MTs', 'outside'))[1], '.2e')), '', '')
    
    tableS1.add_row('Ratio (lobe/neck)',
                    str(format(sp.stats.mannwhitneyu(extract_ratio_from_df(localDensitiesR[(localDensitiesR['Genotype'] =='WT') & (localDensitiesR['Filament'] == 'AFs')], 'perp')[0], extract_ratio_from_df(localDensitiesR[(localDensitiesR['Genotype'] =='act2-1act7-1') & (localDensitiesR['Filament'] == 'AFs')], 'perp')[0])[1], '.2e')), 
                    str(format(sp.stats.mannwhitneyu(extract_ratio_from_df(localDensitiesR[(localDensitiesR['Genotype'] =='WT') & (localDensitiesR['Filament'] == 'MTs')], 'perp')[0], extract_ratio_from_df(localDensitiesR[(localDensitiesR['Genotype'] =='act2-1act7-1') & (localDensitiesR['Filament'] == 'MTs')], 'perp')[0])[1], '.2e')),
                    str(format(sp.stats.mannwhitneyu(extract_ratio_from_df(localDensitiesC[(localDensitiesC['Genotype'] =='WT') & (localDensitiesC['Filament'] == 'AFs')], 'circ')[0], extract_ratio_from_df(localDensitiesC[(localDensitiesC['Genotype'] =='act2-1act7-1') & (localDensitiesC['Filament'] == 'AFs')], 'circ')[0])[1], '.2e')),
                    str(format(sp.stats.mannwhitneyu(extract_ratio_from_df(localDensitiesC[(localDensitiesC['Genotype'] =='WT') & (localDensitiesC['Filament'] == 'MTs')], 'circ')[0], extract_ratio_from_df(localDensitiesC[(localDensitiesC['Genotype'] =='act2-1act7-1') & (localDensitiesC['Filament'] == 'MTs')], 'circ')[0])[1], '.2e')),
                    str(format(sp.stats.mannwhitneyu(extract_ratio_from_df(localDensitiesP[(localDensitiesP['Genotype'] =='WT') & (localDensitiesP['Filament'] == 'AFs')], 'poly')[0], extract_ratio_from_df(localDensitiesP[(localDensitiesP['Genotype'] =='act2-1act7-1') & (localDensitiesP['Filament'] == 'AFs')], 'poly')[0])[1], '.2e')),
                    str(format(sp.stats.mannwhitneyu(extract_ratio_from_df(localDensitiesP[(localDensitiesP['Genotype'] =='WT') & (localDensitiesP['Filament'] == 'MTs')], 'poly')[0], extract_ratio_from_df(localDensitiesP[(localDensitiesP['Genotype'] =='act2-1act7-1') & (localDensitiesP['Filament'] == 'MTs')], 'poly')[0])[1], '.2e')), '-', '-')
    
    console.print('[bold blue]Supplementary Table 1[/bold blue]')
    console.print(tableS1)
    
    ##############################
    ########## Table S2 ##########
    ##############################
    tableS2 = Table(title='Summary of pavement cell, lobe and neck number in wild-type and act2-1act7-1 samples in cotyledons.')
    tableS2.add_column('Genotype', justify='left')
    tableS2.add_column('Replicate', justify='center')
    tableS2.add_column('AFs #Cells', justify='center')
    tableS2.add_column('AFs #Lobes', justify='center')
    tableS2.add_column('AFs #Necks', justify='center')
    tableS2.add_column('MTs #Cells', justify='center')
    tableS2.add_column('MTs #Lobes', justify='center')
    tableS2.add_column('MTs #Necks', justify='center')
    
    tableS2.add_row('Wild-type', '1', 
                    extract_cell_number(globalDensitiesC, 'WT', 'AFs', 0, 'WT_1', None),  
                    extract_cell_number(localDensitiesC, 'WT', 'AFs', 0, 'WT_1', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'WT', 'AFs', 0, 'WT_1', 'neck'),  
                    extract_cell_number(globalDensitiesC, 'WT', 'MTs', 0, 'WT_1', None),  
                    extract_cell_number(localDensitiesC, 'WT', 'MTs', 0, 'WT_1', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'WT', 'MTs', 0, 'WT_1', 'neck'))
    tableS2.add_row('', '2',  
                    extract_cell_number(globalDensitiesC, 'WT', 'AFs', 0, 'WT_2', None),  
                    extract_cell_number(localDensitiesC, 'WT', 'AFs', 0, 'WT_2', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'WT', 'AFs', 0, 'WT_2', 'neck'),  
                    extract_cell_number(globalDensitiesC, 'WT', 'MTs', 0, 'WT_2', None),  
                    extract_cell_number(localDensitiesC, 'WT', 'MTs', 0, 'WT_2', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'WT', 'MTs', 0, 'WT_2', 'neck'))
    tableS2.add_row('', '3', 
                    extract_cell_number(globalDensitiesC, 'WT', 'AFs', 0, 'WT_3', None),  
                    extract_cell_number(localDensitiesC, 'WT', 'AFs', 0, 'WT_3', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'WT', 'AFs', 0, 'WT_3', 'neck'),  
                    extract_cell_number(globalDensitiesC, 'WT', 'MTs', 0, 'WT_3', None),  
                    extract_cell_number(localDensitiesC, 'WT', 'MTs', 0, 'WT_3', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'WT', 'MTs', 0, 'WT_3', 'neck'))
    tableS2.add_row('', '4', 
                    extract_cell_number(globalDensitiesC, 'WT', 'AFs', 0, 'WT_4', None),  
                    extract_cell_number(localDensitiesC, 'WT', 'AFs', 0, 'WT_4', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'WT', 'AFs', 0, 'WT_4', 'neck'),  
                    extract_cell_number(globalDensitiesC, 'WT', 'MTs', 0, 'WT_4', None),  
                    extract_cell_number(localDensitiesC, 'WT', 'MTs', 0, 'WT_4', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'WT', 'MTs', 0, 'WT_4', 'neck'))
    tableS2.add_row('', '5', 
                    extract_cell_number(globalDensitiesC, 'WT', 'AFs', 0, 'WT_5', None),  
                    extract_cell_number(localDensitiesC, 'WT', 'AFs', 0, 'WT_5', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'WT', 'AFs', 0, 'WT_5', 'neck'),  
                    extract_cell_number(globalDensitiesC, 'WT', 'MTs', 0, 'WT_5', None),  
                    extract_cell_number(localDensitiesC, 'WT', 'MTs', 0, 'WT_5', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'WT', 'MTs', 0, 'WT_5', 'neck'))
    tableS2.add_row('', '6', 
                    extract_cell_number(globalDensitiesC, 'WT', 'AFs', 0, 'WT_6', None),  
                    extract_cell_number(localDensitiesC, 'WT', 'AFs', 0, 'WT_6', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'WT', 'AFs', 0, 'WT_6', 'neck'),  
                    extract_cell_number(globalDensitiesC, 'WT', 'MTs', 0, 'WT_6', None),  
                    extract_cell_number(localDensitiesC, 'WT', 'MTs', 0, 'WT_6', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'WT', 'MTs', 0, 'WT_6', 'neck'))
    tableS2.add_row('', '7', '-', '-', '-',  
    extract_cell_number(globalDensitiesC, 'WT', 'MTs', 0, 'WT_7', None),  
    extract_cell_number(localDensitiesC, 'WT', 'MTs', 0, 'WT_7', 'lobe'),  
    extract_cell_number(localDensitiesC, 'WT', 'MTs', 0, 'WT_7', 'neck'))
    tableS2.add_row('', '8', '-', '-', '-', 
    extract_cell_number(globalDensitiesC, 'WT', 'MTs', 0, 'WT_8', None),  
    extract_cell_number(localDensitiesC, 'WT', 'MTs', 0, 'WT_8', 'lobe'),  
    extract_cell_number(localDensitiesC, 'WT', 'MTs', 0, 'WT_8', 'neck'))
    tableS2.add_row('', '[italic]Total[/italic]', 
                    f"[italic]{extract_cell_number(globalDensitiesC, 'WT', 'AFs', 0, None, None)}[/italic]",        
                    f"[italic]{extract_cell_number(localDensitiesC, 'WT', 'AFs', 0, None, 'lobe')}[/italic]", 
                    f"[italic]{extract_cell_number(localDensitiesC, 'WT', 'AFs', 0, None, 'neck')}[/italic]", 
                    f"[italic]{extract_cell_number(globalDensitiesC, 'WT', 'MTs', 0, None, None)}[/italic]", 
                    f"[italic]{extract_cell_number(localDensitiesC, 'WT', 'MTs', 0, None, 'lobe')}[/italic]", 
                    f"[italic]{extract_cell_number(localDensitiesC, 'WT', 'MTs', 0, None, 'neck')}[/italic]", end_section=True)    
    
    tableS2.add_row('[italic]act2-1act7-1[/italic]', '1', 
                    extract_cell_number(globalDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_1', None),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_1', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_1', 'neck'),  
                    extract_cell_number(globalDensitiesC, 'act2-1act7-1', 'MTs', 0, 'act2act7_1', None),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'MTs', 0, 'act2act7_1', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'MTs', 0, 'act2act7_1', 'neck'))
    tableS2.add_row('', '2', 
                    extract_cell_number(globalDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_2', None),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_2', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_2', 'neck'),  
                    extract_cell_number(globalDensitiesC, 'act2-1act7-1', 'MTs', 0, 'act2act7_2', None),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'MTs', 0, 'act2act7_2', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'MTs', 0, 'act2act7_2', 'neck'))
    tableS2.add_row('', '3', 
                    extract_cell_number(globalDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_3', None),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_3', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_3', 'neck'),  
                    extract_cell_number(globalDensitiesC, 'act2-1act7-1', 'MTs', 0, 'act2act7_3', None),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'MTs', 0, 'act2act7_3', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'MTs', 0, 'act2act7_3', 'neck'))
    tableS2.add_row('', '4', 
                    extract_cell_number(globalDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_4', None),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_4', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_4', 'neck'),  
                    extract_cell_number(globalDensitiesC, 'act2-1act7-1', 'MTs', 0, 'act2act7_4', None),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'MTs', 0, 'act2act7_4', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'MTs', 0, 'act2act7_4', 'neck'))
    tableS2.add_row('', '5', 
                    extract_cell_number(globalDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_5', None),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_5', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_5', 'neck'),               
                    '-', '-', '-')
    tableS2.add_row('', '6', 
                    extract_cell_number(globalDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_6', None),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_6', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_6', 'neck'),               
                    '-', '-', '-')
    tableS2.add_row('', '7', 
                    extract_cell_number(globalDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_7', None),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_7', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_7', 'neck'),               
                    '-', '-', '-')
    tableS2.add_row('', '8', 
                    extract_cell_number(globalDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_8', None),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_8', 'lobe'),  
                    extract_cell_number(localDensitiesC, 'act2-1act7-1', 'AFs', 0, 'act2act7_8', 'neck'),               
                    '-', '-', '-')
    tableS2.add_row('', '[italic]Total[/italic]', 
        f"[italic]{extract_cell_number(globalDensitiesC, 'act2-1act7-1', 'AFs', 0, None, None)}[/italic]",        
        f"[italic]{extract_cell_number(localDensitiesC, 'act2-1act7-1', 'AFs', 0, None, 'lobe')}[/italic]", 
        f"[italic]{extract_cell_number(localDensitiesC, 'act2-1act7-1', 'AFs', 0, None, 'neck')}[/italic]", 
        f"[italic]{extract_cell_number(globalDensitiesC, 'act2-1act7-1', 'MTs', 0, None, None)}[/italic]", 
        f"[italic]{extract_cell_number(localDensitiesC, 'act2-1act7-1', 'MTs', 0, None, 'lobe')}[/italic]", 
        f"[italic]{extract_cell_number(localDensitiesC, 'act2-1act7-1', 'MTs', 0, None, 'neck')}[/italic]")

    console.print('[bold blue]Supplementary Table 2[/bold blue]')
    console.print(tableS2)
    
    ##############################
    ########## Table S3 ##########
    ##############################
    grouped = globalDensities.groupby(['Genotype', 'Timepoint'])

    completenessWT = [grouped.get_group(('WT', t))['RelativeCompleteness'] for t in timepoints]
    completenessM  = [grouped.get_group(('act2-1act7-1', t))['RelativeCompleteness'] for t in timepoints]
  
    tableS3 = Table(title='Statistical comparison of pavement cell shape complexity across developmental time points in wild-type and act2-1act7-1 leaves.')
    tableS3.add_column('', justify='left')
    tableS3.add_column('', justify='left')
    tableS3.add_column('[italic]act2-1act7-1[/italic]', justify='center')
    tableS3.add_column('', justify='center')
    tableS3.add_column('', justify='center')
    tableS3.add_column('', justify='center')
    tableS3.add_column('', justify='center')
    
    tableS3.add_row('[bold]Mann-Whitney U test[/bold]', '', '0h', '12h', '24h', '36h', '48h')
    tableS3.add_row('Wild-type', '0h', 
                    str(format(pg.mwu(completenessWT[0], completenessM[0])['p-val'].iloc[0], '.2e')),           
                    str(format(pg.mwu(completenessWT[0], completenessM[1])['p-val'].iloc[0], '.2e')), 
                    str(format(pg.mwu(completenessWT[0], completenessM[2])['p-val'].iloc[0], '.2e')), 
                    str(format(pg.mwu(completenessWT[0], completenessM[3])['p-val'].iloc[0], '.2e')), 
                    str(format(pg.mwu(completenessWT[0], completenessM[4])['p-val'].iloc[0], '.2e')))
    tableS3.add_row('', '12h', 
                    f"[white on cyan]{str(format(pg.mwu(completenessWT[1], completenessM[0])['p-val'].iloc[0], '.2e'))}[/]",           
                    str(format(pg.mwu(completenessWT[1], completenessM[1])['p-val'].iloc[0], '.2e')), 
                    str(format(pg.mwu(completenessWT[1], completenessM[2])['p-val'].iloc[0], '.2e')), 
                    str(format(pg.mwu(completenessWT[1], completenessM[3])['p-val'].iloc[0], '.2e')), 
                    str(format(pg.mwu(completenessWT[1], completenessM[4])['p-val'].iloc[0], '.2e')))
    tableS3.add_row('', '24h', 
                    str(format(pg.mwu(completenessWT[2], completenessM[0])['p-val'].iloc[0], '.2e')),           
                    f"[white on cyan]{str(format(pg.mwu(completenessWT[2], completenessM[1])['p-val'].iloc[0], '.2e'))}[/]", 
                    str(format(pg.mwu(completenessWT[2], completenessM[2])['p-val'].iloc[0], '.2e')), 
                    str(format(pg.mwu(completenessWT[2], completenessM[3])['p-val'].iloc[0], '.2e')), 
                    str(format(pg.mwu(completenessWT[2], completenessM[4])['p-val'].iloc[0], '.2e')))
    tableS3.add_row('', '36h', 
                    str(format(pg.mwu(completenessWT[3], completenessM[0])['p-val'].iloc[0], '.2e')),           
                    f"[white on #808080]{str(format(pg.mwu(completenessWT[3], completenessM[1])['p-val'].iloc[0], '.2e'))}[/]", 
                    f"[white on cyan]{str(format(pg.mwu(completenessWT[3], completenessM[2])['p-val'].iloc[0], '.2e'))}[/]", 
                    str(format(pg.mwu(completenessWT[3], completenessM[3])['p-val'].iloc[0], '.2e')), 
                    str(format(pg.mwu(completenessWT[3], completenessM[4])['p-val'].iloc[0], '.2e')))
    tableS3.add_row('', '48h', 
                    str(format(pg.mwu(completenessWT[4], completenessM[0])['p-val'].iloc[0], '.2e')),           
                    str(format(pg.mwu(completenessWT[4], completenessM[1])['p-val'].iloc[0], '.2e')), 
                    f"[white on #808080]{str(format(pg.mwu(completenessWT[4], completenessM[2])['p-val'].iloc[0], '.2e'))}[/]", 
                    str(format(pg.mwu(completenessWT[4], completenessM[3])['p-val'].iloc[0], '.2e')), 
                    str(format(pg.mwu(completenessWT[4], completenessM[4])['p-val'].iloc[0], '.2e')), end_section=True)
    
    tableS3.add_row('[bold]Kolmoogorov-Smirnov test[/bold]', '', '0h', '12h', '24h', '36h', '48h')
    tableS3.add_row('Wild-type', '0h', 
                    str(format(sp.stats.ks_2samp(completenessWT[0], completenessM[0])[1], '.2e')),           
                    str(format(sp.stats.ks_2samp(completenessWT[0], completenessM[1])[1], '.2e')), 
                    str(format(sp.stats.ks_2samp(completenessWT[0], completenessM[2])[1], '.2e')), 
                    str(format(sp.stats.ks_2samp(completenessWT[0], completenessM[3])[1], '.2e')), 
                    str(format(sp.stats.ks_2samp(completenessWT[0], completenessM[4])[1], '.2e')))
    tableS3.add_row('', '12h', 
                    f"[white on cyan]{str(format(sp.stats.ks_2samp(completenessWT[1], completenessM[0])[1], '.2e'))}[/]",           
                    str(format(sp.stats.ks_2samp(completenessWT[1], completenessM[1])[1], '.2e')), 
                    str(format(sp.stats.ks_2samp(completenessWT[1], completenessM[2])[1], '.2e')), 
                    str(format(sp.stats.ks_2samp(completenessWT[1], completenessM[3])[1], '.2e')), 
                    str(format(sp.stats.ks_2samp(completenessWT[1], completenessM[4])[1], '.2e')))
    tableS3.add_row('', '24h', 
                    str(format(sp.stats.ks_2samp(completenessWT[2], completenessM[0])[1], '.2e')),           
                    f"[white on cyan]{str(format(sp.stats.ks_2samp(completenessWT[2], completenessM[1])[1], '.2e'))}[/]", 
                    str(format(sp.stats.ks_2samp(completenessWT[2], completenessM[2])[1], '.2e')), 
                    str(format(sp.stats.ks_2samp(completenessWT[2], completenessM[3])[1], '.2e')), 
                    str(format(sp.stats.ks_2samp(completenessWT[2], completenessM[4])[1], '.2e')))
    tableS3.add_row('', '36h', 
                    str(format(sp.stats.ks_2samp(completenessWT[3], completenessM[0])[1], '.2e')),           
                    f"[white on #808080]{str(format(sp.stats.ks_2samp(completenessWT[3], completenessM[1])[1], '.2e'))}[/]", 
                    f"[white on cyan]{str(format(sp.stats.ks_2samp(completenessWT[3], completenessM[2])[1], '.2e'))}[/]", 
                    str(format(sp.stats.ks_2samp(completenessWT[3], completenessM[3])[1], '.2e')), 
                    str(format(sp.stats.ks_2samp(completenessWT[3], completenessM[4])[1], '.2e')))
    tableS3.add_row('', '48h', 
                    str(format(sp.stats.ks_2samp(completenessWT[4], completenessM[0])[1], '.2e')),           
                    str(format(sp.stats.ks_2samp(completenessWT[4], completenessM[1])[1], '.2e')), 
                    str(format(sp.stats.ks_2samp(completenessWT[4], completenessM[2])[1], '.2e')), 
                    str(format(sp.stats.ks_2samp(completenessWT[4], completenessM[3])[1], '.2e')), 
                    str(format(sp.stats.ks_2samp(completenessWT[4], completenessM[4])[1], '.2e')), end_section=True)

    tableS3.add_row("[bold]Levene's test[/bold]", '', '0h', '12h', '24h', '36h', '48h')
    tableS3.add_row('Wild-type', '0h', 
                    str(format(sp.stats.levene(completenessWT[0], completenessM[0])[1], '.2e')),           
                    str(format(sp.stats.levene(completenessWT[0], completenessM[1])[1], '.2e')), 
                    str(format(sp.stats.levene(completenessWT[0], completenessM[2])[1], '.2e')), 
                    str(format(sp.stats.levene(completenessWT[0], completenessM[3])[1], '.2e')), 
                    str(format(sp.stats.levene(completenessWT[0], completenessM[4])[1], '.2e')))
    tableS3.add_row('', '12h', 
                    f"[white on cyan]{str(format(sp.stats.levene(completenessWT[1], completenessM[0])[1], '.2e'))}[/]",           
                    str(format(sp.stats.levene(completenessWT[1], completenessM[1])[1], '.2e')), 
                    str(format(sp.stats.levene(completenessWT[1], completenessM[2])[1], '.2e')), 
                    str(format(sp.stats.levene(completenessWT[1], completenessM[3])[1], '.2e')), 
                    str(format(sp.stats.levene(completenessWT[1], completenessM[4])[1], '.2e')))
    tableS3.add_row('', '24h', 
                    str(format(sp.stats.levene(completenessWT[2], completenessM[0])[1], '.2e')),           
                    f"[white on cyan]{str(format(sp.stats.levene(completenessWT[2], completenessM[1])[1], '.2e'))}[/]", 
                    f"[white on #808080]{str(format(sp.stats.levene(completenessWT[2], completenessM[2])[1], '.2e'))}[/]", 
                    str(format(sp.stats.levene(completenessWT[2], completenessM[3])[1], '.2e')), 
                    str(format(sp.stats.levene(completenessWT[2], completenessM[4])[1], '.2e')))
    tableS3.add_row('', '36h', 
                    str(format(sp.stats.levene(completenessWT[3], completenessM[0])[1], '.2e')),           
                    str(format(sp.stats.levene(completenessWT[3], completenessM[1])[1], '.2e')), 
                    f"[white on cyan]{str(format(sp.stats.levene(completenessWT[3], completenessM[2])[1], '.2e'))}[/]", 
                    f"[white on #808080]{str(format(sp.stats.levene(completenessWT[3], completenessM[3])[1], '.2e'))}[/]", 
                    str(format(sp.stats.levene(completenessWT[3], completenessM[4])[1], '.2e')))
    tableS3.add_row('', '48h', 
                    str(format(sp.stats.levene(completenessWT[4], completenessM[0])[1], '.2e')),           
                    str(format(sp.stats.levene(completenessWT[4], completenessM[1])[1], '.2e')), 
                    str(format(sp.stats.levene(completenessWT[4], completenessM[2])[1], '.2e')), 
                    f"[white on cyan]{str(format(sp.stats.levene(completenessWT[4], completenessM[3])[1], '.2e'))}[/]", 
                    f"[white on #808080]{str(format(sp.stats.levene(completenessWT[4], completenessM[4])[1], '.2e'))}[/]", end_section=True)
    
    console.print('[bold blue]Supplementary Table 3[/bold blue]')
    console.print(tableS3)
        
    
    ##############################
    ########## Table S4 ##########
    ##############################
   
    tableS4 = Table(title='Summary of pavement cell numbers and tracked cells in time-series leaf analysis.')
    tableS4.add_column('Genotype', justify='left')
    tableS4.add_column('Filament', justify='left')
    tableS4.add_column('Replicate', justify='center')
    tableS4.add_column('#Cells (0h)', justify='center')
    tableS4.add_column('#Cells (12h)', justify='center')
    tableS4.add_column('#Cells (24h)', justify='center')
    tableS4.add_column('#Cells (36h)', justify='center')
    tableS4.add_column('#Cells (48h)', justify='center')
    tableS4.add_column('#Cells (tracked)', justify='center')
    
    tableS4.add_row('Wild-type', 'AF', '1', 
                  extract_cell_number(globalDensities, 'WT', 'AFs', 0, 'R1', None), 
                  extract_cell_number(globalDensities, 'WT', 'AFs', 12, 'R1', None), 
                  extract_cell_number(globalDensities, 'WT', 'AFs', 24, 'R1', None), 
                  extract_cell_number(globalDensities, 'WT', 'AFs', 36, 'R1', None), 
                  extract_cell_number(globalDensities, 'WT', 'AFs', 48, 'R1', None), 
                  extract_cell_number(trackedLabels, 'WT', 'AFs', None, 'R1', None))
  
    tableS4.add_row('', '', '2', 
                  extract_cell_number(globalDensities, 'WT', 'AFs', 0, 'R2', None), 
                  extract_cell_number(globalDensities, 'WT', 'AFs', 12, 'R2', None), 
                  extract_cell_number(globalDensities, 'WT', 'AFs', 24, 'R2', None), 
                  extract_cell_number(globalDensities, 'WT', 'AFs', 36, 'R2', None), 
                  extract_cell_number(globalDensities, 'WT', 'AFs', 48, 'R2', None), 
                  extract_cell_number(trackedLabels, 'WT', 'AFs', None, 'R2', None))
    
    tableS4.add_row('', '', '3', 
                  extract_cell_number(globalDensities, 'WT', 'AFs', 0, 'R4', None), 
                  extract_cell_number(globalDensities, 'WT', 'AFs', 12, 'R4', None), 
                  extract_cell_number(globalDensities, 'WT', 'AFs', 24, 'R4', None), 
                  extract_cell_number(globalDensities, 'WT', 'AFs', 36, 'R4', None), 
                  extract_cell_number(globalDensities, 'WT', 'AFs', 48, 'R4', None), 
                  extract_cell_number(trackedLabels, 'WT', 'AFs', None, 'R4', None))
    
    tableS4.add_row('', 'MT', '1', 
                  extract_cell_number(globalDensities, 'WT', 'MTs', 0, 'R1', None), 
                  extract_cell_number(globalDensities, 'WT', 'MTs', 12, 'R1', None), 
                  extract_cell_number(globalDensities, 'WT', 'MTs', 24, 'R1', None), 
                  extract_cell_number(globalDensities, 'WT', 'MTs', 36, 'R1', None), 
                  extract_cell_number(globalDensities, 'WT', 'MTs', 48, 'R1', None), 
                  extract_cell_number(trackedLabels, 'WT', 'MTs', None, 'R1', None))
  
    tableS4.add_row('', '', '2', 
                  extract_cell_number(globalDensities, 'WT', 'MTs', 0, 'R2', None), 
                  extract_cell_number(globalDensities, 'WT', 'MTs', 12, 'R2', None), 
                  extract_cell_number(globalDensities, 'WT', 'MTs', 24, 'R2', None), 
                  extract_cell_number(globalDensities, 'WT', 'MTs', 36, 'R2', None), 
                  extract_cell_number(globalDensities, 'WT', 'MTs', 48, 'R2', None), 
                  extract_cell_number(trackedLabels, 'WT', 'MTs', None, 'R2', None))
    
    tableS4.add_row('', '', '3', 
                  extract_cell_number(globalDensities, 'WT', 'MTs', 0, 'R3', None), 
                  extract_cell_number(globalDensities, 'WT', 'MTs', 12, 'R3', None), 
                  extract_cell_number(globalDensities, 'WT', 'MTs', 24, 'R3', None), 
                  extract_cell_number(globalDensities, 'WT', 'MTs', 36, 'R3', None), 
                  extract_cell_number(globalDensities, 'WT', 'MTs', 48, 'R3', None), 
                  extract_cell_number(trackedLabels, 'WT', 'MTs', None, 'R3', None))
    
    tableS4.add_row('', '[italic]Total[/italic]', '',  
                  f"[italic]{extract_cell_number(globalDensities, 'WT', None, 0, None, None)}[/italic]", 
                  f"[italic]{extract_cell_number(globalDensities, 'WT', None, 12, None, None)}[/italic]", 
                  f"[italic]{extract_cell_number(globalDensities, 'WT', None, 24, None, None)}[/italic]", 
                  f"[italic]{extract_cell_number(globalDensities, 'WT', None, 36, None, None)}[/italic]", 
                  f"[italic]{extract_cell_number(globalDensities, 'WT', None, 48, None, None)}[/italic]",
                  f"[italic]{extract_cell_number(trackedLabels, 'WT', None, None, None, None)}[/italic]", end_section=True)
                                                                                                                                                                                                                              
    tableS4.add_row('act2-1act7-1', 'AF', '1', 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'AFs', 0, 'R1', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'AFs', 12, 'R1', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'AFs', 24, 'R1', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'AFs', 36, 'R1', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'AFs', 48, 'R1', None), 
                  extract_cell_number(trackedLabels, 'act2-1act7-1', 'AFs', None, 'R1', None))
                                    
    tableS4.add_row('', '', '2', 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'AFs', 0, 'R2', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'AFs', 12, 'R2', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'AFs', 24, 'R2', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'AFs', 36, 'R2', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'AFs', 48, 'R2', None), 
                  extract_cell_number(trackedLabels, 'act2-1act7-1', 'AFs', None, 'R2', None))
                                    
    tableS4.add_row('', '', '3', 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'AFs', 0, 'R3', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'AFs', 12, 'R3', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'AFs', 24, 'R3', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'AFs', 36, 'R3', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'AFs', 48, 'R3', None), 
                  extract_cell_number(trackedLabels, 'act2-1act7-1', 'AFs', None, 'R3', None))
    
    tableS4.add_row('', 'MT', '1', 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'MTs', 0, 'R1', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'MTs', 12, 'R1', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'MTs', 24, 'R1', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'MTs', 36, 'R1', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'MTs', 48, 'R1', None), 
                  extract_cell_number(trackedLabels, 'act2-1act7-1', 'MTs', None, 'R1', None))
                                    
    tableS4.add_row('', '', '2', 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'MTs', 0, 'R2', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'MTs', 12, 'R2', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'MTs', 24, 'R2', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'MTs', 36, 'R2', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'MTs', 48, 'R2', None), 
                  extract_cell_number(trackedLabels, 'act2-1act7-1', 'MTs', None, 'R2', None))
                                    
    tableS4.add_row('', '', '3', 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'MTs', 0, 'R3', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'MTs', 12, 'R3', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'MTs', 24, 'R3', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'MTs', 36, 'R3', None), 
                  extract_cell_number(globalDensities, 'act2-1act7-1', 'MTs', 48, 'R3', None), 
                  extract_cell_number(trackedLabels, 'act2-1act7-1', 'MTs', None, 'R3', None))
    
    tableS4.add_row('', '[italic]Total[/italic]', '', 
                  f"[italic]{extract_cell_number(globalDensities, 'act2-1act7-1', None, 0, None, None)}[/italic]", 
                  f"[italic]{extract_cell_number(globalDensities, 'act2-1act7-1', None, 12, None, None)}[/italic]", 
                  f"[italic]{extract_cell_number(globalDensities, 'act2-1act7-1', None, 24, None, None)}[/italic]", 
                  f"[italic]{extract_cell_number(globalDensities, 'act2-1act7-1', None, 36, None, None)}[/italic]", 
                  f"[italic]{extract_cell_number(globalDensities, 'act2-1act7-1', None, 48, None, None)}[/italic]",
                  f"[italic]{extract_cell_number(trackedLabels, 'act2-1act7-1', None, None, None, None)}[/italic]",)
    
    console.print('[bold blue]Supplementary Table 4[/bold blue]')
    console.print(tableS4)
    
    
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


def extract_cell_number(data, genotype, filament, timepoint, replicate, nodetype):
    if nodetype == None:
        if filament != None:
            if timepoint != None:
                if replicate != None:
                    cellNr = data[(data['Genotype'] == genotype) & (data['Filament'] == filament) & (data['Replicate'] == replicate) & (data['Timepoint'] == timepoint)].shape[0]
                else:
                    cellNr = data[(data['Genotype'] == genotype) & (data['Filament'] == filament) & (data['Timepoint'] == timepoint)].shape[0]
            else:
                cellNr = data[(data['Genotype'] == genotype) & (data['Filament'] == filament) & (data['Replicate'] == replicate)].shape[0]
        else:
            if timepoint != None:
                cellNr = data[(data['Genotype'] == genotype) & (data['Timepoint'] == timepoint)].shape[0]
            else:
                cellNr = data[data['Genotype'] == genotype].shape[0]
    else:
        if replicate != None:
            cellNr = data[(data['Genotype'] == genotype) & (data['Filament'] == filament) & (data['Replicate'] == replicate) & (data['Timepoint'] == timepoint) & (data['NodeType'] == nodetype)].shape[0]
        else:
            cellNr = data[(data['Genotype'] == genotype) & (data['Filament'] == filament) & (data['Timepoint'] == timepoint) & (data['NodeType'] == nodetype)].shape[0]
    return str(cellNr)

    
def extract_cell_data(data, genotype, filament, nodetype):
    if nodetype != None:
        if nodetype == 'lobe' or nodetype == 'neck':
            cellData = data[(data['Genotype'] == genotype) & (data['Filament'] == filament) & (data['NodeType'] == nodetype)]
            return cellData['Density']
        else:
            cellData = data[(data['Genotype'] == genotype) & (data['Filament'] == filament)]
            if nodetype == 'inside':
                return cellData['densityInsidePolygon']
            else:
                return cellData['densityOutsidePolygon']
                
    else:
        cellData = data[(data['Genotype'] == genotype) & (data['Filament'] == filament)]
        return cellData['Density']

 
def extract_ratio_from_df(df, prop):
    dataRatios = []
    if prop == 'perp' or prop == 'circ':
        cellRatios = []
        for rep in np.unique(df['Replicate']):
            dfRep = df[df['Replicate'] == rep]
            cellNumbers = np.unique(dfRep['CellNumber'])
            for cell in cellNumbers:
                intensityLobes = np.array(dfRep[(dfRep['CellNumber'] == cell) & (dfRep['NodeType'] == 'lobe')]['Density'])
                intensityNecks = np.array(dfRep[(dfRep['CellNumber'] == cell) & (dfRep['NodeType'] == 'neck')]['Density'])
                if np.mean(intensityNecks) > 0.001:
                    cellRatio = np.mean(intensityLobes) / np.mean(intensityNecks)
                    if cellRatio <= 2:
                        cellRatios.append(cellRatio)
        dataRatios.append(cellRatios)
    elif prop == 'poly':
        cellRatios = []
        for rep in np.unique(df['Replicate']):
            dfRep = df[df['Replicate'] == rep]
            cellNumbers = np.unique(dfRep['CellNumber'])
            for cell in cellNumbers:
                intensityLobes = np.array(dfRep[dfRep['CellNumber'] == cell]['densityOutsidePolygon'])
                intensityNecks = np.array(dfRep[dfRep['CellNumber'] == cell]['densityInsidePolygon'])
                if np.mean(intensityNecks) > 0.005:
                    cellRatio = np.mean(intensityLobes) / np.mean(intensityNecks)
                    if cellRatio <= 2:
                        cellRatios.append(cellRatio)
        dataRatios.append(cellRatios)
    return(dataRatios) 
  
    
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

###############################################################################

if __name__ == '__main__':
    main()
