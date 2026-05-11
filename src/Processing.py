#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 16:25:43 2026

@author: Jacqueline Nowak, JNowak@mpimp-golm.mpg.de
"""
from pathlib import Path
import sys
import glob
import pickle
import numpy as np
import skimage
import re
import ast 

# add current script directory to path
pathScript = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathScript))
import Utils as ASU

###############################################################################
def process_images(config, pathData):
    
    if config['output']['globalDensities'] == True:
        initialize_table(pathData, 'globalDensities.csv')
    if config['output']['perpendicularDensities'] == True:
        initialize_table(pathData, 'perpendicularDensities.csv')
    if config['output']['circularDensities'] == True:
        initialize_table(pathData, 'circularDensities.csv')
    if config['output']['polygonalDensities'] == True:
        initialize_table(pathData, 'polygonalDensities.csv')
        
        
    # collect all image folder of the project
    for folder, meta in iterate_structure_from_yaml(config, pathData):
        
        # find all images with defined pattern
        pattern = config['data']['file_pattern']
        imageFiles = sorted(folder.glob(pattern))

        # get segmentation parameters from config file
        resolution = get_image_resolution(config, meta)
        inputChannelMembrane = get_input_channel(config, meta, 'membrane')
        inputChannelCytoskeleton = get_input_channel(config, meta, 'filament')
        selectedCells = get_selected_cells(config, meta)
        manualChanges = get_manual_changes(config, meta)
        timepoint = get_timepoint(config, meta)
        
        # get cytoskeleton parameters
        gauss, sigma, block, small, factr = get_cytoskeleton_parameters(config, meta)
         
        print(folder)
        fileMembrane = [f for f in imageFiles if inputChannelMembrane in f.name]
        fileCytoskeleton = [f for f in imageFiles if inputChannelCytoskeleton in f.name]
        
        
        ### membrane image processing
        if len(fileMembrane) != 0:
            print('...Process membrane image:', fileMembrane[0].name)
            # segment cells from membrane images
            skeletonImageCells, branchlessSkeletonCells, labeledCells, labelsCells = segment_pavement_cells(fileMembrane[0], manualChanges, folder)
            
            if config['output']['labeledCells'] == True:
                np.save(folder / 'labeledCells.npy', labeledCells)
            if config['output']['skeletonImageCells'] == True:
                np.save(folder / 'skeletonImageCells.npy', skeletonImageCells)
                
            # create visibility graphs for segmented cells
            visibilityGraphs, cellContours, junctions, shapeResultsTable = ASU.create_visibility_graphs(labeledCells, labelsCells, branchlessSkeletonCells, skeletonImageCells, resolution, selectedCells, folder)
            if config['output']['visibilityGraphs'] == True:
                with open(folder / 'visibilityGraphs.gpickle', 'wb') as f:
                    pickle.dump(visibilityGraphs, f, pickle.HIGHEST_PROTOCOL)
            if config['output']['cellContours'] == True:
                with open(folder / 'cellContours.gpickle', 'wb') as f:
                    pickle.dump(cellContours, f, pickle.HIGHEST_PROTOCOL)
            if config['output']['junctions'] == True:
                np.save(folder / 'junctions.npy', junctions)
            if config['output']['shapeResultsTable'] == True:
                shapeResultsTable.to_csv(folder / 'shapeResultsTable.csv', index=False)    
        else:
            print('...No membrane image was found')
    

        ### filament image processing
        if len(fileCytoskeleton) != 0:
            print('...Process filament image:', fileCytoskeleton[0].name)
            # extract sructures from cytoskeleton images
            filteredCytoskeleton = extract_cytoskeleton(fileCytoskeleton[0], gauss, sigma, block, small, factr)
            if config['output']['filteredCytoskeleton'] == True:
                np.save(folder / 'filteredCytoskeleton.npy', filteredCytoskeleton)
        else:
            print('...No filament image was found.')

        
        ### merge membrane and cytoskeleton analysis
        if len(fileMembrane) != 0 and len(fileCytoskeleton) != 0:
            print('...Calculate cell densities.')
            if config['output']['membraneSegmentationOverlay'] == True:
                ASU.create_cell_cytoskeleton_overlay(labeledCells, filteredCytoskeleton, folder / 'Overlay_Cells_Filaments.png', selectedCells)   
                
            # quantify cytoskeleton density in local and global cell regions
            if config['output']['globalDensities'] == True:
                dfG = ASU.calculate_mask_densities(visibilityGraphs, cellContours, filteredCytoskeleton, labeledCells, meta['replicates'], timepoint, junctions, resolution, meta['targets'], meta['conditions'], 'global')
                append_dataframe(dfG, pathData / 'globalDensities.csv')
            if config['output']['perpendicularDensities'] == True:
                dfP = ASU.calculate_mask_densities(visibilityGraphs, cellContours, filteredCytoskeleton, labeledCells, meta['replicates'], timepoint, junctions, resolution, meta['targets'], meta['conditions'], 'perpendicular')
                append_dataframe(dfP, pathData / 'perpendicularDensities.csv')
            if config['output']['circularDensities'] == True:
                dfC = ASU.calculate_mask_densities(visibilityGraphs, cellContours, filteredCytoskeleton, labeledCells, meta['replicates'], timepoint, junctions, resolution, meta['targets'], meta['conditions'], 'circular')
                append_dataframe(dfC, pathData / 'circularDensities.csv')
            if config['output']['polygonalDensities'] == True:
                dfPo = ASU.calculate_mask_densities(visibilityGraphs, cellContours, filteredCytoskeleton, labeledCells,meta['replicates'], timepoint, junctions, resolution, meta['targets'], meta['conditions'], 'polygonal')
                append_dataframe(dfPo, pathData / 'polygonalDensities.csv')              

            
        elif len(fileMembrane) != 0 and len(fileCytoskeleton) == 0:
            print('...Calculate global cell properties.')
            # plot overlay of extracted cells and extracted cytoskeleton
            if config['output']['membraneSegmentationOverlay'] == True:
                ASU.create_membrane_segmentation_overlay(fileMembrane[0], labeledCells, folder / 'Overlay_Cells_Membrane.png', selectedCells)   
               
            # quantify global shape properties
            if config['output']['globalDensities'] == True:
                dfG = ASU.calculate_global_properties(visibilityGraphs, cellContours, None, labeledCells, meta['replicates'], timepoint, junctions, resolution, meta['targets'], meta['conditions'], 'global')
                append_dataframe(dfG, pathData / 'globalDensities.csv')

      
###############################################################################   
def iterate_structure_from_yaml(config, pathData):
    structure = config['structure']
    keys = list(structure.keys())
    values = list(structure.values())
    
    def recurse(level, currentPath, meta):
        if level == len(keys):
            yield currentPath, meta.copy()
            return
        key = keys[level]
        vals = values[level]
        if not vals:
            if not currentPath.exists():
                return
            for sub in currentPath.iterdir():
                if not sub.is_dir():
                    continue
                meta[key] = sub.name
                yield from recurse(level + 1, sub, meta)
        else:
            for val in vals:
                nextPath = currentPath / str(val)
                if not nextPath.exists():
                    continue
                meta[key] = val
                yield from recurse(level + 1, nextPath, meta)
    yield from recurse(0, pathData, {})
    
        
def get_image_resolution(config, meta):
    if 'resolutions' in config and config['resolutions'] is not None:
        res = config['resolutions']
        values = list(meta.values())
        for val in values:
            if isinstance(res, dict) and val in res:
                res = res[val]
            else: 
                break
    else:
        res = 1
    res = float(res)
    return res


def get_input_channel(config, meta, channel):
    if 'inputChannel' in config and config['inputChannel'] is not None:
        inp = config['inputChannel']
        values = list(meta.values())
        for val in values:
            if isinstance(inp, dict) and val in inp:
                inp = inp[val]
            else:
                break
        inp = inp[channel]
    else:
        inp = None
    return inp 


def get_selected_cells(config, meta):
    if 'cells' in config and config['cells'] is not None:
        cell = config['cells']
        values = list(meta.values())
        for val in values:
            cell = cell[val]
        cell = parse_cell_labels(cell)
    else:
        cell = None
    return cell 

def parse_cell_labels(cell):
    raw = cell.strip().lstrip('[').rstrip(']')
    tokens = re.split(r',\s*(?![^(]*\))', raw)
    result = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token.startswith('('):
            result.append(ast.literal_eval(token))
        elif re.match(r'^\d+[-:]\d+$', token):
            sep = '-' if '-' in token else ':'
            start, end = token.split(sep)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(token))
    return result
   
 
def get_manual_changes(config, meta):
    if 'manualChanges' in config  and config['manualChanges'] is not None:
        man = config['manualChanges']
        values = list(meta.values())
        for val in values:
            man = man[val]
    else:
        man = []
    return man 
 
    
def get_cytoskeleton_parameters(config, meta):
    if 'parameters' in config and config['parameters'] is not None:
        params = config['parameters']
        values = list(meta.values())
        for val in values:
            if isinstance(params, dict) and val in params:
                params = params.get(val, {})     
            else:
                break
    else:
        params = [2.0, 2.0, 111.0, 27.0, 0.5]
    return params


def get_timepoint(config, meta):
    timepoints = config['data']['timepoints']
    if timepoints == 'single':
        tp = 0
    else:
        tp = int(meta['timepoints'])
    return tp


def segment_pavement_cells(imagePath, manualChanges, folder):
    rawImage = skimage.io.imread(imagePath)
    rawImage = np.pad(rawImage, 5)
    skeletonImage, branchlessSkeleton, labeledImage, labels = ASU.process_membrane_image(rawImage, manualChanges, folder)
    return skeletonImage, branchlessSkeleton, labeledImage, labels   
    
    
def extract_cytoskeleton(imagePath, gauss, sigma, block, small, factr):
    cytoskeletonImage = skimage.io.imread(imagePath)
    cytoskeletonImage = np.pad(cytoskeletonImage, 5)
    filteredImage = ASU.process_cytoskeleton_image(cytoskeletonImage, gauss, sigma, block , small ,factr)
    return filteredImage


def initialize_table(pathData, tableName):
    tablePath = pathData / tableName
    if tablePath.exists():
        tablePath.unlink()  # delete old table


def append_dataframe(df, pathTable):
    df.to_csv(pathTable, mode='a', header=not pathTable.exists(), index=False)