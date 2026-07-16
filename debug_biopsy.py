import os
import pandas as pd

meta_path = "data/raw_pad_ufes/metadata.csv"
df = pd.read_csv(meta_path)
mel_row = df[df['diagnostic'] == 'MEL'].iloc[1]
nev_row = df[df['diagnostic'] == 'NEV'].iloc[1]

print("MEL ID:", mel_row['img_id'])
print("NEV ID:", nev_row['img_id'])

data_dir = "data/raw_pad_ufes/"

def get_image_path(img_id, base_dir):
    for part in ["imgs_part_1", "imgs_part_2", "imgs_part_3"]:
        path = os.path.join(base_dir, part, img_id)
        if os.path.exists(path):
            return path
    return None

mel_path = get_image_path(mel_row['img_id'], data_dir)
nev_path = get_image_path(nev_row['img_id'], data_dir)

print("MEL Path:", mel_path)
print("NEV Path:", nev_path)
