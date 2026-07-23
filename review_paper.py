import sys
try:
    from gradio_client import Client, handle_file
except ImportError:
    print("gradio_client not installed")
    sys.exit(1)

import os

client = Client("maxidl/openreviewer")

# 1. Process the file to get the text
# Assuming we pass the .tex file directly since it's text, or compile to PDF
# Let's just try to read the .tex file locally and pass as text, which is faster and doesn't require PDF extraction if we use /generate directly!
# Wait, the API has /process_file which takes a file. 
# Alternatively, we can just pass the raw latex text to /generate directly.
# Let's read the .tex file.
tex_file = "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/paper/mi4medfm_submission_FINAL_v5.tex"
with open(tex_file, "r") as f:
    paper_text = f.read()

# 2. Set up the review template for 1st MEDFM MICCAI workshop 2026
review_template = """## Summary
Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary.

## Relevance to MI4MedFM
Please evaluate how well the paper aligns with the themes of the 1st MICCAI Workshop on Mechanistic Interpretability for Medical Foundation Models (MI4MedFM). Does it address internal analysis, validation, debugging, or safer clinical deployment of medical foundation models?

## Soundness
Please assign the paper a numerical rating on the following scale to indicate the soundness of the technical claims, experimental and research methodology and on whether the central claims of the paper are adequately supported with evidence. Choose from the following:
4: excellent
3: good
2: fair
1: poor

## Presentation
Please assign the paper a numerical rating on the following scale to indicate the quality of the presentation. This should take into account the writing style and clarity, as well as contextualization relative to prior work. Choose from the following:
4: excellent
3: good
2: fair
1: poor

## Contribution to Mechanistic Interpretability
Please assign the paper a numerical rating on the following scale to indicate the quality of the overall contribution this paper makes to the research area being studied (specifically mechanistic interpretability in medical AI). Are the questions being asked important? Does the paper bring a significant originality of ideas and/or execution? Are the results valuable to share with the broader MICCAI community? Choose from the following:
4: excellent
3: good
2: fair
1: poor

## Strengths
A substantive assessment of the strengths of the paper, touching on each of the following dimensions: originality, quality, clarity, and significance. We encourage reviewers to be broad in their definitions of originality and significance.

## Weaknesses
A substantive assessment of the weaknesses of the paper. Focus on constructive and actionable insights on how the work could improve towards its stated goals. Be specific, avoid generic remarks.

## Questions
Please list up and carefully describe any questions and suggestions for the authors. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation.

## Rating
Please provide an "overall score" for this submission. Choose from the following:
1: strong reject
3: reject, not good enough
5: marginally below the acceptance threshold
6: marginally above the acceptance threshold
8: accept, good paper
10: strong accept, should be highlighted at the conference
"""

print("Sending request to openreviewer API...")
result = client.predict(
    paper_text=paper_text,
    review_template=review_template,
    api_name="/generate"
)

with open("/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/paper/openreviewer_feedback.md", "w") as f:
    f.write(result)

print("Review generated and saved to paper/openreviewer_feedback.md")
