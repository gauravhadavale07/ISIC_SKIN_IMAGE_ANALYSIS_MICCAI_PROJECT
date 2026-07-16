import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from viz_style import setup_style
except ImportError:
    def setup_style(): pass

def get_image_path(img_id, base_dir):
    for part in ["imgs_part_1/imgs_part_1", "imgs_part_2/imgs_part_2", "imgs_part_3/imgs_part_3", "imgs_part_1", "imgs_part_2", "imgs_part_3"]:
        path = os.path.join(base_dir, part, img_id)
        if os.path.exists(path):
            return path
    return None

def plot_biopsy_leak():
    setup_style()
    # Script is run from root, so path is data/
    meta_path = "data/raw_pad_ufes/metadata.csv"
    if not os.path.exists(meta_path):
        meta_path = "../data/raw_pad_ufes/metadata.csv"
    if not os.path.exists(meta_path):
        print(f"Metadata not found at {meta_path}")
        return
        
    df = pd.read_csv(meta_path)
    
    # We will pick a specific MEL image with a clear surgical marker, and 2nd NEV
    mel_row = df[df['img_id'] == 'PAT_717_1347_899.png'].iloc[0]
    nev_row = df[df['diagnostic'] == 'NEV'].iloc[1]
    
    # Path relative to script location or root
    data_dir = "data/raw_pad_ufes/" if os.path.exists("data/raw_pad_ufes/") else "../data/raw_pad_ufes/"
    
    mel_path = get_image_path(mel_row['img_id'], data_dir)
    nev_path = get_image_path(nev_row['img_id'], data_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    if nev_path and os.path.exists(nev_path):
        img_nev = Image.open(nev_path)
        axes[0].imshow(img_nev)
    axes[0].set_title("Benign (NEV)\nClean Lesion", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    if mel_path and os.path.exists(mel_path):
        img_mel = Image.open(mel_path)
        axes[1].imshow(img_mel)
    axes[1].set_title("Malignant (MEL)\nSurgical Marker Visible", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    fig.suptitle("Visual Evidence of Biopsy Leak in PAD-UFES-20", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    out_dir = "figures/output" if os.path.exists("figures") else "output"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/fig_biopsy_leak.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_dir}/fig_biopsy_leak.pdf", bbox_inches='tight')
    print("Saved fig_biopsy_leak.pdf")

if __name__ == "__main__":
    plot_biopsy_leak()
