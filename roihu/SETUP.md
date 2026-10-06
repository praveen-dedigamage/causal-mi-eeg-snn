# Running the experiments on CSC Roihu

Roihu is CSC's 2026 system. It is **not** Puhti with a new name — the GPU
nodes are NVIDIA GH200 Grace Hopper, whose CPU side is **ARM (aarch64)**.
That single fact drives most of the setup below.

---

## 0. Before anything: check access

Puhti access does **not** automatically grant Roihu access.

```bash
ssh <username>@roihu-gpu.csc.fi        # must succeed
csc-projects                            # list projects and their services
```

You need a project with **Roihu GPU** enabled and GPU billing units left.
If the login fails or no project lists Roihu, apply via MyCSC first —
nothing below will work until it does.

> **Use `roihu-gpu.csc.fi`, not `roihu-cpu.csc.fi`.**
> CSC's docs are explicit: software built on GPU nodes only works on GPU
> nodes, "and this also applies to Python environments". A environment built
> on the CPU login node is x86 and will fail inside an ARM GPU job. Confirm
> with `uname -m` -> expect `aarch64`.

---

## 1. Get the code onto Roihu

Clone the repository into your project's scratch directory:

```bash
# on roihu-gpu
cd /scratch/<project>            # NOT $HOME: scratch is bigger and faster
git clone https://github.com/praveen-dedigamage/causal-mi-eeg-snn.git fbcsp
```

or copy a local checkout directly:

```bash
rsync -av --exclude '.git' --exclude 'Results*' --exclude '.venv*' \
    ./ <username>@roihu-gpu.csc.fi:/scratch/<project>/fbcsp/
```

---

## 2. Find the PyTorch module name

Deliberately **not** hard-coded in the scripts — it must come from the system,
not from a guess.

```bash
module spider python-pytorch
```

Note the exact name/version it reports, then export it for every later step:

```bash
export PYTORCH_MODULE=python-pytorch/2.10
```

---

## 3. Set your project ID

The batch scripts take the billing account from the environment rather than
from an `#SBATCH` line. Add your project to `~/.bashrc`:

```bash
export SBATCH_ACCOUNT=project_<your-id>
```

---

## 4. One-time setup

CSC limits login nodes to "one-core jobs that finish in minutes and require
less than 1 GiB of memory", enforced by terminating offenders without warning.
Setup is therefore split in two.

**4a — login node (light: module load + pip install only)**

```bash
bash roihu/00_setup_roihu.sh
```

Aborts unless `uname -m` is `aarch64`, loads `python-pytorch/2.10`, and
installs `snntorch` / `moabb` / `mne` into `~/.local`. It deliberately does
*not* call `torch.cuda.is_available()` -- login nodes expose no GPU, so a
`False` there would be meaningless. CUDA is checked on a real GPU node by the
preflight job.

**4b — batch job (dataset download + epoching)**

```bash
sbatch roihu/00b_prepare_data.sh
```

This looks like "just a download", but MOABB also epochs the data (12 subjects
x 400 trials x 13 ch x 2561 samples) -- far past the login-node budget, so it
runs as a job. It also has to happen *before* the training array: 5 concurrent
folds per subject would otherwise race to populate the same `~/mne_data`
cache.

It runs on `gputest` rather than a CPU partition for a non-obvious reason: the
environment was installed on the **ARM** GPU login node, so it is aarch64 and
cannot be imported on Roihu's x86 CPU nodes.

> If this job fails with a DNS/network error, compute nodes have no outbound
> internet. Fall back to fetching on the login node with `--download-only`,
> which skips the expensive epoching and stays within the login-node budget.

## 5. Preflight (15 minutes, cheap)

```bash
sbatch roihu/preflight.sh
```

Checks arch, CUDA visibility inside a real job, every project import, and that
the dataset is cached. It runs on the 15-minute `gputest` partition, so a wrong
module name surfaces in minutes rather than as 60 simultaneous failures later.

**Do not skip this.** Read the output and confirm it ends with
`PREFLIGHT OK` before continuing.

---

## 6. Train, then sweep

The commands below use the scripts' defaults. The paper's runs set the filter
bank and the output folders explicitly; those exact commands are in the
top-level `README.md`.

```bash
TRAIN=$(sbatch --parsable roihu/01_train_array.sh)
sbatch --dependency=afterok:${TRAIN} roihu/02_quant_sweep.sh
```

The dependency means the sweep starts only if training succeeds.

**Layout:** 12 array tasks, one per subject, each running that subject's
5 folds **concurrently on one GPU**. This is deliberate — the workload leaves
the GPU ~80 % idle (measured: 3 concurrent trainings each slow by only 18 %
while total throughput rises 2.5x), so packing folds converts ~11.5 h of
serial work into ~3-4 h of wall clock and bills 12 GPU allocations instead
of 60.

Monitor:

```bash
squeue --me
tail -f logs/s1_f0.log            # a specific fold
seff <jobid>                       # efficiency after it finishes
```

---

## 7. Results

```
Results_bnci2015/Subject_<N>/fold_<K>/     # per-fold artifacts
Results_quant/quant_sweep_BNCI2015_001_uniform.csv
Results_quant/quant_sweep_BNCI2015_001_per-group.csv
```

Pull them back:

```bash
rsync -av <username>@roihu-gpu.csc.fi:/scratch/<project>/fbcsp/Results_quant/ ./
```

---

## What the two sweeps answer

**uniform** — every parameter group quantised to the same width, so a row
reported at N bits has no full-precision parameter anywhere: EA whiteners,
CSP filters, z-norm statistics, SNN weights **and** SNN biases.
`all@32bit` must equal `fp32_reference`; 32-bit quantisation is lossless
(measured 1e-10 relative error), so if those two disagree the harness is
broken, not the hardware story.

**per-group** — one group quantised at a time. This tests the prediction that
the **EA whitener fails first**: on a real fold it took 20.4 % relative RMS
error at 4 bits versus 15.2 % for CSP, because it is built from
`eigenvalue ** -0.5` and spans ~100x dynamic range (vs ~11x for CSP). The
classifier, by contrast, was insensitive down to 4 bits.

---

## Configuration these runs use

| Setting | Value | Why |
|---|---|---|
| `--seed 42` | fixed | Recorded in `pipeline_params.json`; runs are reproducible |
| AMP | **off** | So the FP32 baseline is genuinely computed in FP32, not float16 |
| `fast_lif` | **off** | Keeps the published snnTorch computation path exactly |
| bands | two-band (6-15, 12-32 Hz) | Set with `FREQ_BANDS`; the scripts' default is an earlier six-band bank (see `README.md`) |

Both `--amp` and `fast_lif=True` exist and are verified, but are off by
default: neither is worth altering a validated pipeline for.
