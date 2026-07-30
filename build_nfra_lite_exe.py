"""
Build Script for NFRA Lite Executable

This script creates a standalone .exe for NFRA Lite.
PyTorch must already be installed on the target machine.

Usage:
    python build_nfra_lite_exe.py

Created by Saurav Bhandari
"""

import PyInstaller.__main__
import os
import sys

def build_exe():
    print("Building NFRA Lite executable...")
    
    # Get the absolute path of the project
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, "src")
    
    PyInstaller.__main__.run([
        'src/nfra/models/nfra_lite.py',           # Main entry point
        '--name=NFRA_Lite',
        '--onefile',                              # Single executable file
        '--console',                              # Show console window
        '--add-data', f'{src_path};nfra',         # Include nfra package
        '--paths', src_path,
        '--hidden-import=nfra',
        '--hidden-import=nfra.core',
        '--hidden-import=nfra.models',
        '--hidden-import=nfra.training',
        '--hidden-import=nfra.evaluation',
        '--hidden-import=nfra.utils',
        '--collect-all=nfra',
        '--exclude-module=torch',                 # Don't bundle PyTorch
        '--exclude-module=torchvision',
        '--exclude-module=torchaudio',
    ])
    
    print("\nBuild complete!")
    print("Executable location: dist/NFRA_Lite.exe")

if __name__ == "__main__":
    build_exe()