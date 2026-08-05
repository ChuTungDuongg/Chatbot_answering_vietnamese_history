import modal

app = modal.App("vn-history-volume-test")

volume = modal.Volume.from_name(
    "vn-history-artifacts",
    create_if_missing=False,
)


@app.function(volumes={"/artifacts": volume})
def test_volume():
    path = "/artifacts/test_modal.txt"

    print("Reading:", path)

    with open(path, encoding="utf-8-sig") as file:
        print("Content:", file.read().strip())


@app.local_entrypoint()
def main():
    test_volume.remote()