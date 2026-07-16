import modal
import subprocess

image = modal.Image.debian_slim().apt_install(
    "texlive-latex-base",
    "texlive-fonts-recommended",
    "texlive-extra-utils",
    "texlive-latex-extra",
    "texlive-publishers",
    "texlive-science",
    "pandoc"
).add_local_dir("/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/paper", remote_path="/root/paper")

app = modal.App("compile-extended-latex")

@app.function(image=image)
def compile():
    # Convert Markdown to LaTeX using pandoc
    subprocess.run(["pandoc", "walkthrough.md", "-o", "walkthrough.tex"], cwd="/root/paper", check=True)
    subprocess.run(["pandoc", "deep_analysis.md", "-o", "deep_analysis.tex"], cwd="/root/paper", check=True)
    subprocess.run(["pandoc", "verification_report.md", "-o", "verification_report.tex"], cwd="/root/paper", check=True)
    
    # Compile the extended LaTeX document
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main_extended.tex"], cwd="/root/paper", check=False)
    subprocess.run(["bibtex", "main_extended"], cwd="/root/paper", check=False)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main_extended.tex"], cwd="/root/paper", check=False)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main_extended.tex"], cwd="/root/paper", check=False)
    
    with open("/root/paper/main_extended.pdf", "rb") as f:
        return f.read()

@app.local_entrypoint()
def main():
    pdf_bytes = compile.remote()
    with open("/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/main1.pdf", "wb") as f:
        f.write(pdf_bytes)
    print("PDF Compiled and saved to main1.pdf")
