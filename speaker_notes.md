# Speaker notes — VRAM back-of-envelope (TinyLlama + this lab)

Teleprompter style: short lines, pause where you see a blank line.

---

## Opening hook

You do not need a PhD in memory architecture to size a GPU for inference.

You need two buckets:

one — **weights** stay resident while the model is loaded.

two — **KV cache** grows with how many tokens you keep “live” at once — prompts, generations, and concurrency.

Everything else is smaller noise until you are tight on memory.

---

## HBM versus SRAM — one sentence for engineers

**HBM** on the GPU is the big pool where weights and KV cache live.

**SRAM** inside streaming multiprocessors is tiny and fast — holds pieces of work briefly during math.

When we say “how much VRAM do I need,” we mean **HBM**, not on-chip SRAM.

---

## Bucket one — model weights (TinyLlama, FP16)

**Rule of thumb:**

parameters × bytes per parameter ≈ weight memory.

TinyLlama is about **1.1 billion** parameters.

Our configs use **`dtype: float16`** — **two bytes** per weight.

Multiply:

1.1 × 10⁹ × 2 bytes ≈ **2.2 gigabytes** for weights alone.

That **2.2 GB** is the only part we computed from first principles — parameter count × storage width.

---

## What “framework overhead, CUDA kernels, scratch buffers” actually means

These are **real categories** of extra GPU memory **beyond** the raw weight tensor.

They are **not** a second secret formula with ten digits of precision.

**Framework overhead** is the **training/inference stack** — here **PyTorch** under **vLLM** — keeps structures on the GPU:

so, ur allocator **caches** that hold freed blocks so the next allocation is fast,

**metadata** for tensors, sometimes **CUDA graphs** or **captured** sequences that duplicate buffers, and whatever the library **pre-reserves** so it does not malloc mid-request, which means failed allocation.

You rarely add these up by hand; you see them as “used memory” in **`nvidia-smi`** minus the weight tensor you can account for.

**CUDA kernels — clarify the word**

People say “kernels” and mean two different things. The **GPU programs** themselves are **tiny** on disk and in instruction memory — not gigabytes.

What **does** cost VRAM are the **workspaces** those kernels need at **runtime**:

**cuBLAS** / **cuDNN** (and similar) pick **fast algorithms** that need **temporary buffers** for big matrix multiplies and fusions. Buffers on a GPU are contiguous blocks of VRAM used to store data.

So in these notes, “CUDA kernel” really means **workspace memory tied to GPU math libraries**, not the size of the `.cubin` file.

**Scratch buffers**

During a **forward pass**, intermediate results exist briefly: partial attention results, outputs of one layer feeding the next, temporary tensors for **fused** or **chunked** ops.

Peak usage can **spike above** “weights plus KV” for a **short** window — that peak is what sometimes tips you into OOM under batching.

---

## How we got “2.5 to 3 GB” — honesty

We **did not** add line items like “300 MB framework + 400 MB cuDNN + …” to derive **2.5 to 3**.

There is **no** single public table that matches every driver, CUDA, and vLLM version.

**2.5 to 3 GB** is a **practical fudge band** on top of **2.2 GB**: think **roughly ten to thirty percent** headroom, or “what instructors often see on **`nvidia-smi`** right after one vLLM replica has loaded TinyLlama FP16.”

**Teach the method:** students should **measure**:

load the model once, read **VRAM used**, subtract their **param × bytes** estimate, call the rest **runtime and framework footprint** for that stack.

Concrete steps on the VM: **README.md** or **run.md**, section **9** — *Measure VRAM beyond raw weights*.

That is how you **calculate** it for real — not from this slide alone.

---

## Bucket two — KV cache (why load spikes when many people hit the API)

Transformers store **key and value** tensors per layer so they do not recompute attention over old tokens.

More tokens in flight — longer prompts, longer answers, more **parallel requests** — more KV.

TinyLlama uses **grouped-query attention**:

**22 layers**, **4 key-value heads**, **64 dimensions per head**, FP16.

A standard back-of-envelope for **KV bytes per token** in the cache:

two — for K and V,

times number of layers,

times number of KV heads,

times head dimension,

times bytes per element.

**2 × 22 × 4 × 64 × 2 bytes**

equals about **23 kilobytes per token** in KV — in the right ballpark for this architecture.

Example — **512 tokens** of context worth of KV:

512 × 23 KB ≈ **12 megabytes**.

Example — **2048 tokens**:

about **47 megabytes** — still small next to weights for one sequence.

**Why concurrency matters:**

vLLM can batch many requests.

The KV pool must cover **all active tokens** the scheduler keeps resident — not just one chat.

So stress tests with **high concurrency** inflate KV far more than a single polite request.

That is why the same “small” model can still **OOM** under load if you crank batching and length.

---

## vLLM and “the rest of the bar” in `nvidia-smi`

vLLM reserves a large fraction of remaining VRAM for **paged KV** and batching — default behavior is aggressive about using the GPU.

So **weights plus KV pool** can approach **`gpu_memory_utilization`** times total VRAM — often around **0.9** unless you override flags.

If students ask “why is 90 percent used with a tiny model,” that reservation is the answer.

---

## This lab — **one Docker container** (baseline / smoke)

**Weights:** one copy — about **2.5 to 3 gigabytes** ballpark.

**KV:** depends on YAML — prompts up to hundreds of tokens, outputs up to hundreds, plus whatever concurrent requests you set.

For **one replica** and modest concurrency, **any classroom A100 or GH200** is enormous headroom.

---

## This lab — **four Docker containers** on **one GPU**

Our harness assigns **`gpu_id = replica_index % num_gpus`**.

On a **single-GPU** machine, **num_gpus is one**.

So **all four vLLM processes share the same GPU** — and each process **loads its own full copy of the weights**.

**Rough mental math:**

four times **~2.5 to 3 gigabytes** weights alone → **about 10 to 12 gigabytes** before heavy KV.

Add concurrent KV across four servers — still usually fine on **40 gigabyte** class GPUs for TinyLlama, but you are no longer in “trivial” territory if someone doubles replicas or max lengths.

That is how we sized the earlier “A100 is enough” claim — **orders of magnitude**, not a spreadsheet from the scheduler.

---

## If we ran **only inferencing** — one process, one model

**Single `vllm` server** or one notebook load:

one weight copy — **~2.5 to 3 gigabytes** ballpark.

KV scales with **your** batch and sequence lengths — use the **per-token KV** estimate times **tokens in flight**.

**Docker does not multiply weights** unless you **run multiple containers** each loading the model.

One container equals one load — same as bare metal for VRAM purposes, plus a small overhead.

---

## Docker detail students notice — `shm_size_gb`

Our compose path sets **shared memory** for the container — **not** where weights live.

vLLM and PyTorch use `/dev/shm` for some IPC and tensor paths.

If **shared memory** is too small, you can get confusing errors **without** running out of **VRAM**.

So: **`shm_size`** is about **POSIX shared memory**, not a second kind of “model RAM.”

---

## Quantization — quick aside

**FP16** — two bytes per weight — our default.

**INT8** — about half the weight bytes — rough trend, implementation-dependent.

**INT4** — about quarter — same caveat.

Quantization **primarily shrinks weights**; **KV** is often still FP16 or BF16 unless you configure otherwise — read the stack’s docs before teaching a number.

---

## Closing line for the room

**Calculate VRAM** by adding:

**weight footprint** times **how many processes load the model**,

plus **KV** scaled by **tokens you actually keep resident** under **real concurrency**.

If the sum is well below your GPU’s HBM, you are safe.

If it is close, profile with **`nvidia-smi`** during a **worst-case** run — the harness is built exactly for that story.
