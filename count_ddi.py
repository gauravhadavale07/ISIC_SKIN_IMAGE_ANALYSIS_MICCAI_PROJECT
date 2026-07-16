import pandas as pd
df = pd.read_csv('data/ddi/ddidiversedermatologyimages/ddi_metadata.csv')
fst12 = df[df['skin_tone'] == 12]
fst56 = df[df['skin_tone'] == 56]

pairs = 0
diseases_both = 0
for disease in df['disease'].unique():
    c12 = len(fst12[fst12['disease'] == disease])
    c56 = len(fst56[fst56['disease'] == disease])
    if c12 > 0 and c56 > 0:
        pairs += c12 * c56
        diseases_both += 1
        
print(f'Total diseases with both FST12 and FST56: {diseases_both}')
print(f'Total possible cross-tone pairs for same disease: {pairs}')
print(f'Total FST12 images with a match: {sum(fst12.disease.isin(fst56.disease))}')
print(f'Total FST56 images with a match: {sum(fst56.disease.isin(fst12.disease))}')
