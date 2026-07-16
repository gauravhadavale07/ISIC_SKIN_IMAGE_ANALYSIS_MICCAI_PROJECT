import modal
app = modal.App("test-app")
@app.function()
def foo():
    print("Hello from modal")
@app.local_entrypoint()
def main():
    foo.remote()
