import graphviz
import os

os.makedirs('figures', exist_ok=True)

# Initialize graph
dot = graphviz.Digraph(comment='MI4MedFM Architecture Flow', format='pdf')
dot.attr(rankdir='LR', size='10,6')
dot.attr('node', shape='box', style='rounded,filled', fontname='sans-serif', margin='0.2')

# Define nodes with MI4MedFM specific language
dot.node('Image', 'Clinical Image\n(PAD-UFES-20)', fillcolor='#d5dbdb')
dot.node('Prompt', 'Clinical Prompt\n("Is the diagnosis MEL or NEV?")', fillcolor='#d5dbdb')

dot.node('Vision', 'Vision Encoder\n(CLIP)', fillcolor='#aed6f1')
dot.node('Proj', 'Cross-Modal Projector', fillcolor='#aed6f1')

dot.node('LLM', 'Foundation Model Backbone\n(Mistral-7B)', fillcolor='#f9e79f', shape='cylinder')

dot.node('Lens', 'Interpretability Lens\n(Layer 15 Causal Steering)', fillcolor='#f5b041', penwidth='3', color='#d35400')
dot.node('ActAdd', r'Add $\alpha(V_{benign} - V_{malignant})$', fillcolor='#fdebd0', shape='note')

dot.node('Output', 'Logit Extraction\n(Tokens: "M" vs "N")', fillcolor='#d2b4de')

dot.node('Failure', 'Result: Brittle Reasoning\n(0% Override Rate)', fillcolor='#f2d7d5', color='#c0392b', penwidth='2')

# Edges
dot.edge('Image', 'Vision')
dot.edge('Vision', 'Proj')
dot.edge('Proj', 'LLM')

dot.edge('Prompt', 'LLM')

dot.edge('LLM', 'Lens', label=' Hidden States')
dot.edge('ActAdd', 'Lens', style='dashed')
dot.edge('Lens', 'Output', label=' Steered Forward Pass')

dot.edge('Output', 'Failure')

# Grouping / Subgraphs
with dot.subgraph(name='cluster_medfm') as c:
    c.attr(style='dashed', color='gray')
    c.node('LLM')
    c.node('Lens')
    c.attr(label='Medical Foundation Model')

# Render
dot.render('figures/fig_mi4medfm_architecture_flow', view=False)
# Also render PNG
dot.format = 'png'
dot.render('figures/fig_mi4medfm_architecture_flow', view=False)

# Delete the dot source file to keep clean
if os.path.exists('figures/fig_mi4medfm_architecture_flow'):
    os.remove('figures/fig_mi4medfm_architecture_flow')

print("Saved fig_mi4medfm_architecture_flow")
