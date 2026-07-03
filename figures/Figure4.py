#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 22:12:14 2026

@author: Jacqueline Nowak, JNowak@mpimp-golm.mpg.de
"""

import argparse
from pathlib import Path
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

###############################################################################
# =========================
# Figure 3
# Visualization of actin cytoskeleton organization in cotyledon pavement cells in wild-type and
# the act2-1 act7-1 mutant.
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
    dataCircPC = pd.read_csv(pathCotyledon / 'circularDensities.csv')
        

    ### plot global densities for AFs and MTs in WT and act2-1act7-1
    # variables for boxplot
    hatches = ['', '/', '', '/']
    
    fig, ax = plt.subplots(1, 4, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 2, 2]})
    bp1 = ax[0].violinplot([globalDensities[(globalDensities['Filament'] == 'AFs') & (globalDensities['Genotype'] == 'WT')]['Density'], globalDensities[(globalDensities['Filament'] == 'AFs') & (globalDensities['Genotype'] == 'act2-1act7-1')]['Density']], positions=[0, 1], widths=0.8, showextrema=False, showmedians=True)
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(['WT', 'act2-1act7-1'])
    ax[0].set_ylabel('AF global density')
    ax[0].set_ylim([0, 0.15])
    bp1['cmedians'].set_color('black')
    for idx, box in enumerate(bp1['bodies']):
        box.set_facecolor('#2ca02c')
        box.set_edgecolor('black')
        box.set_alpha(0.75)
    bp2 = ax[1].violinplot([globalDensities[(globalDensities['Filament'] == 'MTs') & (globalDensities['Genotype'] == 'WT')]['Density'], globalDensities[(globalDensities['Filament'] == 'MTs') & (globalDensities['Genotype'] == 'act2-1act7-1')]['Density']], positions=[0, 1], widths=0.8, showextrema=False, showmedians=True)
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(['WT', 'act2-1act7-1'])
    ax[1].set_ylabel('MT global density')
    ax[1].set_ylim([0, 0.15])
    bp2['cmedians'].set_color('black')
    for idx, box in enumerate(bp2['bodies']):
        box.set_facecolor('#e377c2')
        box.set_edgecolor('black')
        box.set_alpha(0.75)   
    # AF density circular mask
    bp3 = ax[2].violinplot([dataCircPC[(dataCircPC['Filament'] == 'AFs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'lobe')]['Density'], 
            dataCircPC[(dataCircPC['Filament'] == 'AFs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'neck')]['Density'],
            dataCircPC[(dataCircPC['Filament'] == 'AFs') & (dataCircPC['Genotype'] == 'act2-1act7-1') & (dataCircPC['NodeType'] == 'lobe')]['Density'],
            dataCircPC[(dataCircPC['Filament'] == 'AFs') & (dataCircPC['Genotype'] == 'act2-1act7-1') & (dataCircPC['NodeType'] == 'neck')]['Density']], positions=[0, 0.75, 2, 2.75], widths=0.55, showextrema=False, showmedians=True)
    ax[2].set_xticks([0.375, 2.375])
    ax[2].set_xticklabels(['WT', 'act2-1act7-1'])
    ax[2].set_ylabel('AF density (circular mask)')
    ax[2].set_ylim([-0.01, 0.2])
    bp3['cmedians'].set_color('black')
    for idx, box in enumerate(bp3['bodies']):
        box.set_facecolor('#2ca02c')
        box.set_edgecolor('black')
        box.set(hatch=hatches[idx])
        box.set_alpha(0.75)
    # MT density circular mask
    bp4 = ax[3].violinplot([dataCircPC[(dataCircPC['Filament'] == 'MTs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'lobe')]['Density'], 
            dataCircPC[(dataCircPC['Filament'] == 'MTs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'neck')]['Density'],
            dataCircPC[(dataCircPC['Filament'] == 'MTs') & (dataCircPC['Genotype'] == 'act2-1act7-1') & (dataCircPC['NodeType'] == 'lobe')]['Density'],
            dataCircPC[(dataCircPC['Filament'] == 'MTs') & (dataCircPC['Genotype'] == 'act2-1act7-1') & (dataCircPC['NodeType'] == 'neck')]['Density']], positions=[0, 0.75, 2, 2.75], widths=0.55, showextrema=False, showmedians=True)
    ax[3].set_xticks([0.375, 2.375])
    ax[3].set_xticklabels(['WT', 'act2-1act7-1'])
    ax[3].set_ylabel('MT density (circular mask)')
    ax[3].set_ylim([-0.01, 0.2])
    bp4['cmedians'].set_color('black')
    for idx, box in enumerate(bp4['bodies']):
        box.set_facecolor('#e377c2')
        box.set_edgecolor('black')
        box.set(hatch=hatches[idx])
        box.set_alpha(0.75)
    fig.tight_layout()
    fig.savefig(pathPlots / 'Figure3C-F_Cotyledons_CytoskeletonDensity_WT_Mutant.png', bbox_inches='tight', dpi=300)               
    plt.close()
    
    
    ### statistical significance
    print('Numbers of samples:', 
          '\n...AF global density:',
          '\n......WT:', globalDensities[(globalDensities['Filament'] == 'AFs') & (globalDensities['Genotype'] == 'WT')].shape[0], 
          '\n......act2-1act7-1:', globalDensities[(globalDensities['Filament'] == 'AFs') & (globalDensities['Genotype'] == 'act2-1act7-1')].shape[0], 
          '\n...MT global density:',
          '\n......WT:', globalDensities[(globalDensities['Filament'] == 'MTs') & (globalDensities['Genotype'] == 'WT')].shape[0], 
          '\n......act2-1act7-1:', globalDensities[(globalDensities['Filament'] == 'MTs') & (globalDensities['Genotype'] == 'act2-1act7-1')].shape[0],           
          '\n...Circular mask:',
          '\n......AFs (WT):', dataCircPC[(dataCircPC['Filament'] == 'AFs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'lobe')].shape[0], 
          '\n......AFs (act2-1act7-1):', dataCircPC[(dataCircPC['Filament'] == 'AFs') & (dataCircPC['Genotype'] == 'act2-1act7-1') & (dataCircPC['NodeType'] == 'lobe')].shape[0],  
          '\n......MTs (WT):', dataCircPC[(dataCircPC['Filament'] == 'MTs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'lobe')].shape[0], 
          '\n......MTs (act2-1act7-1):', dataCircPC[(dataCircPC['Filament'] == 'MTs') & (dataCircPC['Genotype'] == 'act2-1act7-1') & (dataCircPC['NodeType'] == 'lobe')].shape[0])
    
    print('AF global density:', sp.stats.mannwhitneyu(globalDensities[(globalDensities['Filament'] == 'AFs') & (globalDensities['Genotype'] == 'WT')]['Density'], globalDensities[(globalDensities['Filament'] == 'AFs') & (globalDensities['Genotype'] == 'act2-1act7-1')]['Density'])[1])
    
    print('MT global density:', sp.stats.mannwhitneyu(globalDensities[(globalDensities['Filament'] == 'MTs') & (globalDensities['Genotype'] == 'WT')]['Density'], globalDensities[(globalDensities['Filament'] == 'MTs') & (globalDensities['Genotype'] == 'act2-1act7-1')]['Density'])[1])
    
    print('\nCircular Mask:')
    print('AF density in WT:', sp.stats.mannwhitneyu(dataCircPC[(dataCircPC['Filament'] == 'AFs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'lobe')]['Density'], dataCircPC[(dataCircPC['Filament'] == 'AFs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'neck')]['Density'])[1])
    
    print('AF density in act2-1act7-1:', sp.stats.mannwhitneyu(dataCircPC[(dataCircPC['Filament'] == 'AFs') & (dataCircPC['Genotype'] == 'act2-1act7-1') & (dataCircPC['NodeType'] == 'lobe')]['Density'],
            dataCircPC[(dataCircPC['Filament'] == 'AFs') & (dataCircPC['Genotype'] == 'act2-1act7-1') & (dataCircPC['NodeType'] == 'neck')]['Density'])[1])
    
    print('MT density in WT:', sp.stats.mannwhitneyu(dataCircPC[(dataCircPC['Filament'] == 'MTs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'lobe')]['Density'], 
            dataCircPC[(dataCircPC['Filament'] == 'MTs') & (dataCircPC['Genotype'] == 'WT') & (dataCircPC['NodeType'] == 'neck')]['Density'])[1])
    
    print('MT density in act2-1act7-1:', sp.stats.mannwhitneyu(dataCircPC[(dataCircPC['Filament'] == 'MTs') & (dataCircPC['Genotype'] == 'act2-1act7-1') & (dataCircPC['NodeType'] == 'lobe')]['Density'],
            dataCircPC[(dataCircPC['Filament'] == 'MTs') & (dataCircPC['Genotype'] == 'act2-1act7-1') & (dataCircPC['NodeType'] == 'neck')]['Density'])[1])
    
if __name__ == '__main__':
    main()
