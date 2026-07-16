import re

with open('paper/main_template.tex', 'r') as f:
    content = f.read()

# Shrink pipeline diagram
content = content.replace(r'\includegraphics[width=\textwidth]{figures/pipeline_diagram.png}', 
                          r'\includegraphics[width=0.8\textwidth]{figures/pipeline_diagram.png}')

# Shrink Fig 5 further
content = content.replace(r'\includegraphics[width=0.8\textwidth]{figures/fig16_cross_attention_visualization.pdf}', 
                          r'\includegraphics[width=0.6\textwidth]{figures/fig16_cross_attention_visualization.pdf}')

# Shrink Fig 6 and 7
content = content.replace(r'\begin{minipage}[t]{0.48\textwidth}', r'\begin{minipage}[t]{0.45\textwidth}')

with open('paper/main_template.tex', 'w') as f:
    f.write(content)
with open('paper/main.tex', 'w') as f:
    f.write(content)

print("Second rewrite successful.")
