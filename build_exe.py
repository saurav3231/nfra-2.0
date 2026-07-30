"""
NFRA Lite Executable Builder

This script creates a lightweight .exe file.
PyTorch must already be installed on the target system.

Usage:
    python build_exe.py

Created by Saurav Bhandari
"""

import PyInstaller.__main__
import os

def build():
    print("Building NFRA Lite executable...\n")
    
    PyInstaller.__main__.run([
        'main.py',
        '--name=NFRA_Lite',
        '--onefile',
        '--console',
        '--paths=src',
        '--add-data=src;nfra',
        '--hidden-import=nfra',
        '--hidden-import=nfra.core',
        '--hidden-import=nfra.models',
        '--hidden-import=nfra.utils',
        '--collect-submodules=nfra',
        '--exclude-module=torch',
        '--exclude-module=torchvision',
        '--exclude-module=torchaudio',
        '--exclude-module=numpy',
    ])
    
    print("\n" + "="*50)
    print("Build finished!")
    print("Your executable is located at: dist/NFRA_Lite.exe")
    print("="*50)

if __name__ == "__main__":
    build()