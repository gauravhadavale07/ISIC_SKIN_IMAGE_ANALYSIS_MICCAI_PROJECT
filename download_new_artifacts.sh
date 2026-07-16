#!/bin/bash
echo "Downloading results..."
modal volume get miccai-results / /home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/results_new/
cp -r /home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/results_new/* /home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/results/

echo "Downloading new checkpoints..."
for model in "Cross-Attention_V→T_seed_1337" "ImageOnly_seed_2024" "TextOnly_seed_456" "TextOnly_seed_789" "TextOnly_seed_1337" "TextOnly_seed_2024"; do
  echo "Downloading $model..."
  modal volume get miccai-checkpoints /$model /home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/checkpoints/$model/
done
echo "Done!"
