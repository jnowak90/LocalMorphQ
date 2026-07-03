#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 16:25:43 2026

@author: Jacqueline Nowak, JNowak@mpimp-golm.mpg.de
"""
import yaml
import argparse
import sys
from pathlib import Path

pathScript = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathScript))
import src.Processing as Processing

###############################################################################
def load_config(path):
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
        
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config

###############################################################################

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--data', required=True, help='Path to image data')
    args = parser.parse_args()
    
    config = load_config(args.config)
    print('Running project:',  config['project']['name'])
    
    pathData = Path(args.data)

    # -------------------------------------------------------------------------
    # Run pipeline
    Processing.process_images(config, pathData)
    


if __name__ == '__main__':
    main()
    
