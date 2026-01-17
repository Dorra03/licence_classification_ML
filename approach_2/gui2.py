"""
Simple License Classifier GUI - gui2.py
Input raw license text and classify using multiple trained models.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pickle
import os
import sys
from collections import Counter


class SimpleLicenseClassifier:
    """Simple GUI for license classification with multiple models."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("License Classifier - gui2")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Models and vectorizer
        self.models = {}
        self.vectorizer = None
        
        # Load models
        self.load_models()
        
        # Create GUI
        self.create_gui()
    
    def load_models(self):
        """Load trained models from disk."""
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, 'models')
        
        if not os.path.exists(models_dir):
            messagebox.showerror(
                "Error",
                f"Models directory not found at:\n{models_dir}\n\n"
                f"Run: python train_gui2_models.py"
            )
            self.root.destroy()
            return False
        
        try:
            # Load vectorizer
            with open(f'{models_dir}/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load models
            model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
            for model_file in sorted(model_files):
                model_name = model_file.replace('_model.pkl', '')
                with open(f'{models_dir}/{model_file}', 'rb') as f:
                    self.models[model_name] = pickle.load(f)
            
            if not self.models:
                messagebox.showerror("Error", "No models found!")
                self.root.destroy()
                return False
            
            print(f"✓ Loaded {len(self.models)} models")
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models:\n{e}")
            self.root.destroy()
            return False
    
    def create_gui(self):
        """Create the GUI layout."""
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Title
        title = ttk.Label(
            main_frame,
            text="License Classifier - gui2",
            font=("Arial", 16, "bold")
        )
        title.pack()
        
        subtitle = ttk.Label(
            main_frame,
            text=f"Using {len(self.models)} trained models: {', '.join(sorted(self.models.keys()))}",
            font=("Arial", 10),
            foreground="gray"
        )
        subtitle.pack(pady=(0, 15))
        
        # Split frame for input and output
        split_frame = ttk.Frame(main_frame)
        split_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Input
        left_frame = ttk.LabelFrame(split_frame, text="License Text Input", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.text_input = scrolledtext.ScrolledText(
            left_frame,
            height=25,
            width=50,
            wrap=tk.WORD,
            font=("Courier", 10)
        )
        self.text_input.pack(fill=tk.BOTH, expand=True)
        
        # Right side - Output
        right_frame = ttk.LabelFrame(split_frame, text="Classification Results", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.results_text = scrolledtext.ScrolledText(
            right_frame,
            height=25,
            width=50,
            wrap=tk.WORD,
            font=("Courier", 10),
            state=tk.DISABLED
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Buttons frame at bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(15, 0))
        
        classify_btn = ttk.Button(
            button_frame,
            text="CLASSIFY",
            command=self.classify_text
        )
        classify_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(
            button_frame,
            text="Clear Input",
            command=lambda: self.text_input.delete(1.0, tk.END)
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        clear_results_btn = ttk.Button(
            button_frame,
            text="Clear Results",
            command=self.clear_results
        )
        clear_results_btn.pack(side=tk.LEFT, padx=5)
    
    def classify_text(self):
        """Classify the input text using all loaded models."""
        
        # Get input text
        text = self.text_input.get(1.0, tk.END).strip()
        
        if not text:
            messagebox.showwarning("Warning", "Please enter license text to classify.")
            return
        
        if not self.vectorizer or not self.models:
            messagebox.showerror("Error", "Models not loaded properly.")
            return
        
        try:
            # Vectorize text
            X = self.vectorizer.transform([text])
            
            # Clear results
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            
            # Add header
            results = "="*50 + "\n"
            results += "CLASSIFICATION RESULTS\n"
            results += "="*50 + "\n\n"
            
            # Get predictions from all models
            predictions = {}
            for model_name in sorted(self.models.keys()):
                model = self.models[model_name]
                
                # Get prediction
                pred = model.predict(X)[0]
                
                # Get confidence (if available)
                if hasattr(model, 'predict_proba'):
                    conf = max(model.predict_proba(X)[0])
                else:
                    conf = 0.0
                
                predictions[model_name] = (pred, conf)
            
            # Display results
            results += "Model Predictions:\n"
            results += "-"*50 + "\n\n"
            
            for model_name, (pred, conf) in sorted(predictions.items()):
                results += f"{model_name.upper()}\n"
                results += f"  Category:   {pred}\n"
                results += f"  Confidence: {conf*100:.2f}%\n"
                results += "\n"
            
            # Consensus
            results += "="*50 + "\n"
            results += "CONSENSUS\n"
            results += "="*50 + "\n\n"
            
            categories = [p[0] for p in predictions.values()]
            category_counts = Counter(categories)
            
            for category, count in category_counts.most_common():
                results += f"{category}: {count}/{len(self.models)} models\n"
            
            most_common = category_counts.most_common(1)[0]
            results += f"\nMost Likely: {most_common[0]}\n"
            results += f"Agreement: {most_common[1]}/{len(self.models)} models\n"
            
            # Display results
            self.results_text.insert(tk.END, results)
            self.results_text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed:\n{e}")
    
    def clear_results(self):
        """Clear the results text area."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)


def main():
    """Launch the GUI."""
    try:
        root = tk.Tk()
        app = SimpleLicenseClassifier(root)
        
        # Ensure window is visible and focused
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)
        
        root.mainloop()
    except tk.TclError as e:
        print(f"ERROR: Cannot display GUI (headless environment): {e}")
        print("Try running from PowerShell/CMD instead of VS Code terminal")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
