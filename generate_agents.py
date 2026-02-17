
import argparse
import sys
import os
import pandas as pd
import torch

# Add CTAB-GAN-Plus to path
ctab_gan_plus_path = r"./CTAB-GAN-Plus"
sys.path.append(ctab_gan_plus_path)

try:
    from model.ctabgan import CTABGAN
except ImportError:
    print(f"Error: Could not find CTAB-GAN-Plus code or CTABGAN class.")
    print(f"Please check if '{ctab_gan_plus_path}' path and 'model/ctabgan.py' file are correct.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic agents using a trained CTABGAN model.")
    parser.add_argument("--n", type=int, required=True, help="Number of agents to generate.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved CTABGAN model (.pth file).")
    parser.add_argument("--output", type=str, required=True, help="Path to save the generated CSV.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu).")

    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)

    try:
        model = CTABGAN.load(args.model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"Generating {args.n} agents...")
    try:
        generated_data = model.generate_samples(args.n)
        print(f"Generated {len(generated_data)} samples.")
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"Saving to {args.output}...")
    try:
        generated_data.to_csv(args.output, index=False, encoding='utf-8-sig')
        print("Done.")
    except Exception as e:
        print(f"Error saving output: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
