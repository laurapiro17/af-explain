# Deploying the Space

One-time setup (run from the repo root):

```bash
# 1. Install the HF CLI (uv handles this — already in dev extras).
uv run huggingface-cli login   # paste your HF token

# 2. Create the Space (only the first time).
uv run huggingface-cli repo create af-explain --type space --space_sdk streamlit

# 3. Add HF as a git remote inside this folder.
cd hf_space
git init -b main
git remote add hf https://huggingface.co/spaces/laurapiro17/af-explain
git lfs install
git lfs track "*.ckpt"
```

Every time you want to update the demo:

```bash
# 1. Copy the latest checkpoint into this folder.
cp ../outputs/checkpoints/best.ckpt model.ckpt

# 2. Stage + commit + push (LFS handles the .ckpt).
git add -A
git commit -m "deploy: update checkpoint to <metric>"
git push hf main
```

After the push, HF Spaces will rebuild the container automatically and
the demo will go live at:

    https://huggingface.co/spaces/laurapiro17/af-explain

## Sanity checks before pushing

- `python app.py` runs locally (Streamlit opens, demo record loads).
- `model.ckpt` is < 1 GB (HF Spaces free tier limit is generous but
  the file should be a single best checkpoint, not the full training
  log).
- `requirements.txt` lists only what Spaces will install at build time
  (no `dev` extras).
