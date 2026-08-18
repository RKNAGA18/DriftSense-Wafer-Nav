import os
import sys
import glob
import numpy as np
import torch

from train import EdgeUNet # Imports the architecture

def load_model():
    """Strictly loads local weights. Evaluators require NO internet access."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EdgeUNet() 
    
    weight_path = os.path.join(os.path.dirname(__file__), 'models', 'best_weights.pth')
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weights not found at {weight_path}")
        
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, device

def process_file(input_path, output_path, model, device):
    """Processes a single .npy file according to KLA's strict constraints."""
    degraded_img = np.load(input_path, allow_pickle=False).astype(np.float32)
    original_shape = degraded_img.shape
    
    # Handle both (H, W) and (H, W, 1) inputs
    if degraded_img.ndim == 3 and degraded_img.shape[-1] == 1:
        degraded_img = degraded_img[..., 0]
        
    # Add batch and channel dimensions: (1, 1, H, W)
    tensor_img = torch.from_numpy(degraded_img).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        restored_tensor = model(tensor_img)
    
    restored_img = restored_tensor.squeeze().cpu().numpy()
    
    # SANITIZATION: Purge NaNs/Infs and clip strictly to [0, 1]
    restored_img = np.nan_to_num(restored_img, nan=0.0, posinf=1.0, neginf=0.0)
    restored_img = np.clip(restored_img, 0.0, 1.0)
    
    # Reshape to match input dimensions exactly
    if len(original_shape) == 3:
        restored_img = np.expand_dims(restored_img, axis=-1)
        
    np.save(output_path, restored_img.astype(np.float32))

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)
    input_files = glob.glob(os.path.join(input_dir, '*.npy'))
    
    if not input_files:
        print(f"Warning: No .npy files found in {input_dir}")
        sys.exit(0)

    print(f"Found {len(input_files)} .npy files. Loading Edge-UNet...")
    model, device = load_model()
    
    print("Starting KLA compliant restoration pipeline...")
    for idx, input_path in enumerate(input_files):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)
        
        process_file(input_path, output_path, model, device)
        
    print("Restoration complete. Outputs strictly sanitized.")

if __name__ == '__main__':
    main()
