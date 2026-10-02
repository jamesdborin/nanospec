"""Modal is only machine provisioning; torchrun owns the eight local processes."""

from pathlib import Path

import modal

ROOT = Path(__file__).parent
TORCHSPEC = "076a738fdd7e64be1495338f055e8188784633c4"
app = modal.App("nanospec")
volume = modal.Volume.from_name("nanospec-artifacts", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.10.0",
        "transformers==5.3.0",
        "safetensors==0.8.0",
        "huggingface-hub==1.33.0",
        "einops",
        "loguru",
        "omegaconf",
        "numba",
        "numpy<2.4",
        "psutil",
        "kernels",
    )
    .pip_install(
        "wandb",
        "ray",
        "pyzmq",
        "datasets",
        "accelerate",
        "qwen-vl-utils",
        "openai-harmony",
        "openai",
        "pillow",
    )
    .run_commands("pip uninstall -y kernels")
    .run_commands(
        "git clone https://github.com/lightseekorg/torchspec.git /opt/torchspec",
        f"git -C /opt/torchspec checkout {TORCHSPEC}",
        "pip install --no-deps -e /opt/torchspec",
        "python -c 'from torchspec.models.dspark import DSparkModel'",
    )
    .env(
        {
            "HF_HOME": "/artifacts/hf",
            "PYTHONPATH": "/root/nanospec",
            "NCCL_DEBUG": "WARN",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(ROOT / "nanospec", "/root/nanospec/nanospec")
    .add_local_dir(ROOT / "experiments", "/root/nanospec/experiments")
    .add_local_dir(ROOT / "configs", "/root/nanospec/configs")
)


def comparison_cases():
    return [
        ("draft", "tiny", 1, "float32"),
        ("draft", "tiny", 2, "bfloat16"),
        ("draft", "tiny", 4, "bfloat16"),
        ("full", "qwen-parity", 2, "bfloat16"),
        ("full", "qwen", 4, "bfloat16"),
        ("train", "qwen-parity", 4, "bfloat16"),
    ]


@app.function(image=image, volumes={"/artifacts": volume}, timeout=3600, cpu=8, memory=32768)
def cache_target():
    from huggingface_hub import snapshot_download

    from nanospec.config import TARGET_MODEL, TARGET_REVISION

    path = snapshot_download(
        TARGET_MODEL,
        revision=TARGET_REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
    )
    volume.commit()
    return path


@app.function(
    image=image,
    gpu="H100!:8",
    cpu=16,
    memory=131072,
    volumes={"/artifacts": volume},
    timeout=7200,
    max_containers=1,
)
def run_gpu(
    mode="draft",
    train_gpus=2,
    steps=3,
    dtype="float32",
    config="tiny",
    checkpoint="",
    gpu_model="H100",
    learning_rate=0.0,
    tokens="",
    seq_len=32,
    eval_tokens="",
    eval_every=0,
    eval_steps=4,
    eval_at_end=True,
):
    import hashlib
    import json
    import os
    import subprocess

    if mode == "sequence":
        results = []
        for case in comparison_cases():
            case_mode, case_config, count, case_dtype = case
            # Run all subprocesses inside this allocation. Separate remote calls
            # may release the node and queue again between cases.
            result = run_gpu.local(
                case_mode,
                count,
                steps,
                case_dtype,
                case_config,
                "",
                gpu_model,
                learning_rate,
            )
            results.append((case, result))
            if result.get("passed") is False:
                break
        return results

    volume.reload()
    import torch

    probe = subprocess.run(
        ["nvidia-smi", "topo", "-m"], text=True, capture_output=True, check=False
    )
    topology = probe.stdout + probe.stderr
    print(f"nvidia-smi topology (exit {probe.returncode}): {topology}", flush=True)
    if probe.returncode:
        links = subprocess.run(
            ["nvidia-smi", "nvlink", "--status"], text=True, capture_output=True, check=False
        )
        topology += links.stdout + links.stderr
        print(topology, flush=True)
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(devices) != 8:
        raise RuntimeError(f"Expected eight GPUs, got {devices}")
    peers = [
        [torch.cuda.can_device_access_peer(i, j) if i != j else True for j in range(8)]
        for i in range(8)
    ]
    print({"devices": devices, "peer_access": peers}, flush=True)
    if not all(gpu_model in d for d in devices) or not all(map(all, peers)):
        raise RuntimeError(f"Expected eight {gpu_model}s with all-to-all CUDA peer access")
    if "GB/s" not in topology and "NV" not in topology:
        raise RuntimeError("Could not verify active NVLink connectivity")
    suffix = "-resume" if checkpoint else ""
    output = f"/artifacts/{mode}-{config}-{train_gpus}-{dtype}-ce-kl-{gpu_model}{suffix}.json"
    module = (
        "experiments.overlap"
        if mode == "overlap"
        else "experiments.compare"
        if mode in ("draft", "full")
        else "nanospec.train"
    )
    args = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=8",
        "-m",
        module,
        "--train-gpus",
        str(train_gpus),
        "--steps",
        str(steps),
        "--dtype",
        dtype,
        "--config",
        config,
        "--output",
        output,
    ]
    if mode == "full":
        args.append("--target")
    if mode == "overlap":
        args.extend(["--seq-len", "128", "--anchors", "8"])
    if checkpoint:
        args.extend(["--checkpoint", checkpoint])
    if learning_rate:
        args.extend(["--learning-rate", str(learning_rate)])
    if mode == "train":
        args.extend(["--seq-len", str(seq_len)])
        if tokens:
            args.extend(["--tokens", tokens])
        if eval_tokens:
            args.extend(["--eval-tokens", eval_tokens])
        args.extend(["--eval-every", str(eval_every), "--eval-steps", str(eval_steps)])
        if not eval_at_end:
            args.append("--no-eval-at-end")
    env = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        NCCL_TOPO_DUMP_FILE="/tmp/nccl-topology.xml",
        # Verifier pipeline transfers remain synchronous.
        TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING="false",
    )
    subprocess.run(args, cwd="/root/nanospec", env=env, check=True)
    report = json.loads(Path(output).read_text())
    report["topology"] = topology
    report["devices"] = devices
    report["peer_access"] = peers
    sources = sorted(Path("/root/nanospec").glob("**/*.py"))
    report["source_sha256"] = {
        str(p.relative_to("/root/nanospec")): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }
    if Path("/tmp/nccl-topology.xml").exists():
        report["nccl_topology"] = Path("/tmp/nccl-topology.xml").read_text()
    Path(output).write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main(
    mode: str = "draft",
    train_gpus: int = 2,
    steps: int = 3,
    dtype: str = "float32",
    config: str = "tiny",
    checkpoint: str = "",
    gpu: str = "H100!",
    learning_rate: float = 0.0,
    tokens: str = "",
    seq_len: int = 32,
    eval_tokens: str = "",
    eval_every: int = 0,
    eval_steps: int = 4,
    eval_at_end: bool = True,
):
    import json

    if mode == "cache":
        print(cache_target.remote())
        return
    if mode not in ("draft", "full", "train", "sequence", "overlap"):
        raise ValueError("mode must be cache, draft, full, train, sequence, or overlap")
    if gpu not in ("H100!", "H200"):
        raise ValueError("gpu must be H100! (the target) or H200 (supplementary diagnostics)")
    gpu_model = gpu.removesuffix("!")
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    if checkpoint and mode != "train":
        raise ValueError("--checkpoint resumes training; use --mode train")
    if mode != "train" and (tokens or eval_tokens or eval_every or not eval_at_end):
        raise ValueError("Token files and evaluation options require --mode train")
    runner = run_gpu.with_options(gpu=f"{gpu}:8")
    results = runner.remote(
        mode,
        train_gpus,
        steps,
        dtype,
        config,
        checkpoint,
        gpu_model,
        learning_rate,
        tokens=tokens,
        seq_len=seq_len,
        eval_tokens=eval_tokens,
        eval_every=eval_every,
        eval_steps=eval_steps,
        eval_at_end=eval_at_end,
    )
    if mode != "sequence":
        results = [((mode, config, train_gpus, dtype), results)]
    for (case_mode, case_config, count, case_dtype), result in results:
        suffix = "-resume" if checkpoint else ""
        path = (
            out / f"{case_mode}-{case_config}-{count}-{case_dtype}-ce-kl-{gpu_model}{suffix}.json"
        )
        path.write_text(json.dumps(result, indent=2))
        print(f"Saved {path}", flush=True)
        if result.get("passed") is False:
            raise RuntimeError(f"Numerical comparison exceeded its tolerances; see {path}")
