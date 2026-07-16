import modal
import subprocess
import os

app = modal.App("compile-pdf")

image = modal.Image.debian_slim().apt_install(
    "texlive-latex-base", "texlive-latex-extra", "texlive-fonts-recommended"
).workdir("/root/project").add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/paper", 
    remote_path="/root/project/paper"
)

@app.function(image=image)
def compile_pdf():
    print("Compiling PDF...")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd="/root/project/paper")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd="/root/project/paper")
    with open("/root/project/paper/main.pdf", "rb") as f:
        return f.read()

@app.local_entrypoint()
def main():
    pdf_bytes = compile_pdf.remote()
    os.makedirs("/home/ec2-user/.gemini/antigravity-ide/brain/b2419de2-2a1c-40c3-bc25-292cde2ff43d", exist_ok=True)
    with open("/home/ec2-user/.gemini/antigravity-ide/brain/b2419de2-2a1c-40c3-bc25-292cde2ff43d/main.pdf", "wb") as f:
        f.write(pdf_bytes)
    print("PDF compiled and saved!")
