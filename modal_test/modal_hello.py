import modal

app = modal.App("vn-history-modal-learning")


@app.function()
def hello():
    import platform

    print("Hello from Modal!")
    print("OS:", platform.platform())


@app.local_entrypoint()
def main():
    hello.remote()