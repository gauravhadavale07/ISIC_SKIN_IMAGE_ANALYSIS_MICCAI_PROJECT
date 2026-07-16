import modal
import os
import subprocess

image = modal.Image.debian_slim().apt_install("texlive-latex-base", "texlive-fonts-recommended", "texlive-extra-utils", "texlive-latex-extra", "texlive-publishers", "texlive-science").add_local_dir("/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/paper", remote_path="/root/paper").add_local_dir("/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/figures", remote_path="/root/figures")

app = modal.App("compile-latex")

@app.function(image=image)
def compile():
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "reading_between_lesions_master_draft.tex"], cwd="/root/paper", check=True)
    subprocess.run(["bibtex", "reading_between_lesions_master_draft"], cwd="/root/paper", check=False)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "reading_between_lesions_master_draft.tex"], cwd="/root/paper", check=True)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "reading_between_lesions_master_draft.tex"], cwd="/root/paper", check=True)
    with open("/root/paper/reading_between_lesions_master_draft.pdf", "rb") as f:
        return f.read()

@app.local_entrypoint()
def main():
    pdf_bytes = compile.remote()
    with open("/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/paper/main2.pdf", "wb") as f:
        f.write(pdf_bytes)
    print("PDF Compiled and saved to paper/main2.pdf")
