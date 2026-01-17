#!/usr/bin/env python
"""
Launcher script for gui2 - works from any directory
Run this script with: python launch_gui2.py
"""

import os
import sys

def main():
    # Get the directory where this script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Change to script directory (ensures relative paths work)
    os.chdir(script_directory)
    
    # Add script directory to Python path
    if script_directory not in sys.path:
        sys.path.insert(0, script_directory)
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.executable}")
    
    # Check if models exist
    if not os.path.exists('models'):
        print("ERROR: models/ directory not found!")
        print("Please run: python train_gui2_models.py")
        sys.exit(1)
    
    models_dir = 'models'
    model_count = len([f for f in os.listdir(models_dir) if f.endswith('_model.pkl')])
    print(f"Found {model_count} trained models")
    
    # Import and run gui2
    try:
        from gui2 import SimpleLicenseClassifier
        import tkinter as tk
        
        print("Launching GUI...")
        root = tk.Tk()
        app = SimpleLicenseClassifier(root)
        
        # Make window visible and focused
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)
        
        root.mainloop()
        
    except ImportError as e:
        print(f"ERROR: Could not import gui2: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
