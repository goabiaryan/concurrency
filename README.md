# test-lambda — Stress testing lab runbook

This lab teaches **how to stress test** a real inference stack: you drive **TinyLLaMA + vLLM** with repeatable load profiles, collect latency, throughput, errors, and GPU behavior, then **compare runs** to see how the system degrades under pressure. Follow the steps in order on your laptop, then on the Lambda GPU instance.

**Learning goals:** understand smoke vs baseline vs burst vs max-stress scenarios; read `summary.json` and CSV outputs; use CLI overrides to scale a run to your GPU; practice copying artifacts off the VM for reports.

GPU sizing (L40S first, then multi-A100) is summarized in `docs/LAMBDA_INSTANCE_SELECTION.md` if you need deploy form values.

---

## 1. Create an SSH key on your laptop (once)

If you already have a key you use for Lambda, skip to step 2.

```bash
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/id_ed25519_lambda -N ""
```

Show the **public** key and copy the whole line (starts with `ssh-ed25519`):

```bash
cat ~/.ssh/id_ed25519_lambda.pub
```

Never share or paste the file **without** `.pub`.

---

## 2. Create the Lambda instance

In the Lambda console, start a new GPU instance. When it asks for **SSH keys**, paste the contents of `id_ed25519_lambda.pub`.

 Use a GPU image with NVIDIA drivers (Ubuntu + CUDA is typical). Give the boot disk at least **100 GB** if the console allows it, so Docker images and logs fit.

Wait until the instance shows a **public IP** (or hostname) and the SSH user Lambda documents. Many GPU images use **`root`**; some use `ubuntu`—use exactly what the console shows.

The public address must be a **full IPv4** (four dotted parts, e.g. `203.0.113.42`) or the hostname they give. A truncated address (only three numbers) will not connect.

---

## 3. Log in over SSH

You need **one** SSH target from the Lambda console: either the **public IPv4** or the **hostname** they list (if any). Both work the same for `ssh`; use whichever they show and your DNS/network can resolve. You do **not** have to pass IP and hostname together.

From your laptop (replace `LAMBDA_HOST` with that IP or hostname, and `root` if Lambda gives a different user):

```bash
chmod 600 ~/.ssh/id_ed25519_lambda
ssh -i ~/.ssh/id_ed25519_lambda ubuntu@LAMBDA_IP
```

Examples: `root@203.0.113.42` or `root@gpu-abc123.example.lambda.net`—same command shape.

First connection: type `yes` if asked to trust the host key.

---

## 4. On the instance — sanity checks

```bash
nvidia-smi
```

You should see your GPU(s). If this fails, fix drivers / instance type before continuing.

**`nvidia-smi` not found** (Ubuntu may suggest `sudo apt install nvidia-utils-…`): those packages only add the **user-space tool**. They do **not** install the **kernel driver** that talks to the GPU. You need a **full driver** stack (or a provider **GPU image** that already includes it).

**Check hardware vs driver:**

```bash
lspci | grep -i nvidia
lsmod | grep nvidia
```

- **No line from `lspci`** → no GPU passed through to the VM; fix **instance type / console**, not `apt`.
- **GPU in `lspci` but `lsmod` shows no `nvidia` modules** → driver not loaded. On **Ubuntu 24.04** (generic image), install drivers and reboot:

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
```

If you prefer an explicit package, install the **`nvidia-driver-…`** line **`ubuntu-drivers devices` recommends** (for example `nvidia-driver-550-open`), then:

```bash
sudo reboot
```

After the instance comes back, `lsmod | grep nvidia` should list modules and `nvidia-smi` should work.

**Optional** host-Python check (you can **skip this**): the stress suite runs vLLM **inside Docker**, so `nvidia-smi` plus a successful GPU Docker run (step 5) is enough for most setups. Installing **vLLM on the host** pulls a large stack (PyTorch/CUDA wheels); **15–40+ minutes** and several GB are common—if it feels “stuck,” it is often still downloading. Prefer skipping and going straight to Docker.

---

## 5. Install Docker and NVIDIA Container Toolkit (if not preinstalled)

If `docker run hello-world` already works with GPUs, skip this block.

Ubuntu-style install (adjust if Lambda uses another distro):

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
```

Log out and SSH back in so the `docker` group applies, then:

```bash
docker run hello-world
```

Install the **NVIDIA Container Toolkit** (this provides **`nvidia-ctk`**; without it, CDI commands will fail with `command not found`). On Ubuntu/Debian, add NVIDIA’s `apt` repo and install the package ([full guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo mkdir -p /etc/cdi
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo chmod a+r /etc/cdi/nvidia.yaml
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

That last command must print GPU info from inside a container.

**`sudo: nvidia-ctk: command not found`:** install **`nvidia-container-toolkit`** with the `apt` block above first — do not run only the CDI lines.

**`failed to discover GPU vendor from CDI: no known GPU vendor found`:** usually before **`/etc/cdi/nvidia.yaml`** exists. After installing the toolkit, run **`sudo nvidia-ctk runtime configure --runtime=docker`**, then **`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`**, **`sudo chmod a+r /etc/cdi/nvidia.yaml`**, and **`sudo systemctl restart docker`**, then retry the test container.

**`permission denied` on `unix:///var/run/docker.sock`:** your user is not in the `docker` group yet (common on preinstalled Docker images). Run `sudo usermod -aG docker "$USER"`, then **disconnect SSH and log in again** so the new group applies. Until then, use `sudo docker …` or a one-off: `sg docker -c 'docker run …'`. The stress harness needs Docker access from the same shell that runs `python run_suite.py` — after a fresh login, `docker ps` should work without `sudo`.

---

## 6. Get this repo onto the instance

Either clone with git:

```bash
sudo apt-get install -y git
git clone <YOUR_REPO_URL> test-lambda
cd test-lambda
```

Or `scp` / upload a zip from your laptop into `~/test-lambda` and `cd` there.

---

## 7. Python environment and stress harness dependencies

```bash
cd ~/test-lambda
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If Hugging Face blocks the model download, set a token for the shell session:

```bash
export HF_TOKEN=hf_xxxxxxxx
# or
export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxx
```

---

## 8. Build the vLLM inference image (once per machine, or after Dockerfile changes)

```bash
cd ~/test-lambda
source .venv/bin/activate
docker build -f Dockerfile.vllm -t stress-vllm:local .
```

---

## 9. Measure VRAM beyond raw weights (optional)

This matches the back-of-envelope in `speaker_notes.md`: **load the model once**, read **VRAM used**, subtract a **parameter count × bytes-per-weight** estimate, and treat the rest as **runtime and framework footprint** (allocators, library workspaces, vLLM’s KV pool reservation, etc.).

**1. Optional baseline — GPU with no inference containers**

On the instance (no vLLM Docker replicas running):

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Note **memory.used** (driver and display may use a few hundred MiB even when “idle”).

**2. Start exactly one replica and sample while it is loaded**

Single-replica configs include `config/examples/smoke.yaml`. In **one** terminal:

```bash
cd ~/test-lambda
source .venv/bin/activate
python run_suite.py --config config/examples/smoke.yaml
```

In a **second** SSH session on the same instance, wait until the run has started containers and is serving (or watch the first terminal for “ready” / request activity), then:

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Record **memory.used** again. The increase over step 1 (or the absolute value if you skipped baseline) is roughly **one vLLM replica’s footprint** on the GPU for this model and stack.

**3. Naive weight size (first-principles estimate)**

TinyLlama in our YAMLs is about **1.1 × 10⁹** parameters at **FP16** (`dtype: float16` → **2 bytes** per weight):

**1.1 × 10⁹ × 2 ≈ 2.2 × 10⁹ bytes** → about **2.2 GB** order-of-magnitude for weights alone (not exact to the megabyte).

**4. What is “framework footprint” here?**

Subtract that **~2.2 GB** weight estimate from the **extra** VRAM you attributed to the replica in step 2 (after subtracting baseline). The remainder is **not** a precise line-by-line bill — it bundles PyTorch/vLLM **caching allocators**, **cuBLAS/cuDNN-style workspaces**, **KV cache pool** pre-allocation, and short-lived **scratch** buffers. Use **`nvidia-smi`** as the source of truth; see `speaker_notes.md` for vocabulary.

**5. Several replicas on one GPU**

After `standard_burst` / `max_stress`, repeat step 2 while all replicas are up: total used memory should scale **roughly** with **one-replica footprint × number of replicas** when each process loads its own weights (our harness pins replicas with `i % num_gpus`, so on a **single-GPU** VM every replica shares that GPU and loads a **full copy** of the model).

---

## 10. Run the experiments (recommended order)

Stay in `~/test-lambda` with `.venv` activated. Each command writes under `results/<profile>/` (and `logs/` inside that tree).

**Smoke** — one replica, short run; confirms wiring.

```bash
python run_suite.py --config config/examples/smoke.yaml
```

**Baseline** — single replica, light load; keep this `summary.json` for comparisons.

```bash
python run_suite.py --config config/examples/baseline_single.yaml
```

**Standard burst** — four replicas, three waves, tiered prompts (lighter fleet so the lab finishes quickly; scale up via CLI in section 12 if you have a large GPU).

```bash
python run_suite.py --config config/examples/standard_burst.yaml
```

**Max stress** — four replicas, four waves, sustained tail (still the heaviest profile in the lab, but sized to finish in reasonable time).

```bash
python run_suite.py --config config/examples/max_stress.yaml
```

**Matrix** (optional) — several back-to-back runs from `config/matrix/quantization_matrix.example.yaml`. Edit that file for real quantized models before running.

```bash
python -m stress_suite.matrix_runner --matrix config/matrix/quantization_matrix.example.yaml
```

**Compare baseline vs max** (paths assume default `output_dir` in the YAMLs):

```bash
python -m stress_suite.compare_runs \
  results/baseline_single/summary.json \
  results/max_stress/summary.json
```

Add `--json` for machine-readable output.

---

## 11. Where outputs go

Each run creates a directory named by the config’s `output_dir` (for example `results/max_stress/`). Inside you will typically find:

- `summary.json` — rolled-up stats (latency percentiles, errors, throughput, GPU peaks, Docker event summary).
- `requests.csv` — one row per request with timestamps and error classes.
- `cold_start.csv` — time until each replica answered health/models.
- `gpu_timeseries.csv` — periodic `nvidia-smi` samples (if enabled in YAML).
- `docker_events.jsonl` — container lifecycle lines.
- `logs/run_<timestamp>/` — full orchestrator log plus per-span vLLM `/metrics` scrapes.

Copy the whole `results/` tree off the VM when you are done (`scp -r` from your laptop).

---

## 12. Overrides without editing YAML (optional)

Example: smaller run on a dev GPU, tag the SKU in metadata:

```bash
python run_suite.py --config config/examples/max_stress.yaml \
  --containers 4 \
  --waves 2 \
  --concurrent 4 \
  --output-dir ./results/max_smoke \
  --gpu-sku L40S-48GB
```

Other flags: `--base-host-port`, `--suite-name`, `--seed`, `--quantization ""` for FP16-only, `--skip-build` if the image is already built and tagged.

---

## 13. Optional first-boot script on Lambda

If the console offers a **Startup script**, you can adapt `scripts/lambda-cloud-init-example.sh` to install Docker on first boot. Review it before pasting; images differ.

---

## 14. When you are finished

Stop or delete the Lambda instance so you are not billed for idle GPU time. Keep your SSH key on your laptop for the next lab run.

---

## 15. Copy `results/` to your laptop (after all tests)

Use a **new terminal on your Mac** (or VS Code “local” terminal). If you are **inside SSH** on the instance, these commands are backwards—you need the **source** on the remote and the **destination** on your laptop, which only works when `rsync`/`scp` run **locally**.

Replace:

- `LAMBDA_HOST` — instance IP or hostname (step 3).
- `REMOTE_REPO` — parent of `results/` on the instance (often `~/test-lambda`).
- `~/.ssh/lambda_stress` — your key path if different.

```bash
mkdir -p ~/Downloads/test-lambda-results
```

**Whole `results` tree**

```bash
scp -i ~/.ssh/lambda_stress -r "root@LAMBDA_HOST:~/REMOTE_REPO/results" ~/Downloads/test-lambda-results
```

**One file (e.g. summary)**

```bash
scp -i ~/.ssh/lambda_stress "root@LAMBDA_HOST:~/REMOTE_REPO/results/standard_burst/summary.json" ~/Downloads/
```

**`rsync` (large trees, resumable)**

```bash
rsync -avz --progress -e "ssh -i ~/.ssh/lambda_stress" \
  "root@LAMBDA_HOST:~/REMOTE_REPO/results/" \
  ~/Downloads/test-lambda-results/
```

**Concrete example** (IP + `~/test-lambda` + key `lambda_stress`):

```bash
mkdir -p ~/Downloads/test-lambda-results
rsync -avz --progress -e "ssh -i ~/.ssh/lambda_stress" \
  "root@95.133.253.69:~/test-lambda/results/" \
  ~/Downloads/test-lambda-results/
```

The trailing `/` on the remote path copies the *contents* into `~/Downloads/test-lambda-results/`.

---

Equivalent entrypoint for any run: `python -m stress_suite.runner --config <yaml>` instead of `run_suite.py`.
