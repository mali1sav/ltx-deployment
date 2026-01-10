import os
import sys
import runpod

# Make repo importable
sys.path.insert(0, "/app/repo")

MODEL_ID = os.environ.get("HF_MODEL_ID", "Lightricks/LTX-Video")
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

PIPE = None

def load_pipe():
    global PIPE
    if PIPE is not None:
        return PIPE

    # Download weights at runtime (avoids serverless build timeout)
    from huggingface_hub import snapshot_download
    local_dir = "/app/models/LTX-Video"
    os.makedirs(local_dir, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=local_dir,
        token=HF_TOKEN,
        local_dir_use_symlinks=False,
    )

    # Import LTX pipeline from the cloned repo
    # NOTE: exact import path can vary by repo version; adjust if needed.
    from ltx.pipeline import LTXPipeline  # <-- if this import fails, we’ll fix it based on the repo structure

    PIPE = LTXPipeline.from_pretrained(local_dir, token=HF_TOKEN).to("cuda")
    return PIPE


def handler(job):
    try:
        inp = job["input"]

        prompt = inp.get("prompt", "")
        num_frames = int(inp.get("num_frames", 24))
        height = int(inp.get("height", 480))
        width = int(inp.get("width", 704))
        num_inference_steps = int(inp.get("num_inference_steps", 30))
        seed = inp.get("seed")

        pipe = load_pipe()

        kwargs = dict(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
        )
        if seed is not None:
            kwargs["seed"] = int(seed)

        result = pipe(**kwargs)

        # Return something small; you can later add S3/R2 upload if you want
        return {"status": "ok", "message": "generated", "meta": {"frames": num_frames, "height": height, "width": width}}

    except Exception as e:
        return {"status": "error", "error": str(e)}


runpod.serverless.start({"handler": handler})
