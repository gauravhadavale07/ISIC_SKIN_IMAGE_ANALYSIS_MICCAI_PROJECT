import os

with open("paper/main_template.tex", "r") as f:
    content = f.read()

# Replace MILK10k clean split metrics
content = content.replace("$MILK10K_IMG_ACC", "61.40\\%")
content = content.replace("$MILK10K_IMG_F1", "26.63\\%")
content = content.replace("$MILK10K_IMG_AUC", "0.7768")

content = content.replace("$MILK10K_TXT_ACC", "54.71\\%")
content = content.replace("$MILK10K_TXT_F1", "11.79\\%")
content = content.replace("$MILK10K_TXT_AUC", "0.6695")

content = content.replace("$MILK10K_LF_ACC", "69.07\\%")
content = content.replace("$MILK10K_LF_F1", "38.68\\%")
content = content.replace("$MILK10K_LF_AUC", "0.8799")

content = content.replace("$MILK10K_GMU_ACC", "69.29\\%")
content = content.replace("$MILK10K_GMU_F1", "42.13\\%")
content = content.replace("$MILK10K_GMU_AUC", "0.8823")

content = content.replace("$MILK10K_VT_ACC", "70.55\\%")
content = content.replace("$MILK10K_VT_F1", "45.24\\%")
content = content.replace("$MILK10K_VT_AUC", "0.8843")

content = content.replace("$MILK10K_TV_ACC", "71.57\\%")
content = content.replace("$MILK10K_TV_F1", "50.81\\%")
content = content.replace("$MILK10K_TV_AUC", "0.8994")

with open("paper/main.tex", "w") as f:
    f.write(content)
print("Updated paper/main.tex")
