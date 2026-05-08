# Automated Multimodal Analysis of Air Traffic around Non-Towered Airports using Vision-Language Models

Code, dataset, and paper assets for the AIAA submission *Automated
Multimodal Analysis of Air Traffic around Non-Towered Airports using
Vision-Language Models

The system is a **vision-language-model (VLM) based safety advisor**
for non-towered airports. It ingests a transcribed CTAF radio session,
METAR weather data, ADS-B state information, and the VFR sectional
chart of the airport, and produces both a structured safety
classification (three-class *nominal* / *warning* / *hazard*, or
binary *nominal* / *danger*) and a free-text CTAF-style advisory.

The quantitative experiments in this repository instantiate the model
component of the pipeline with **six frozen LLMs** (Qwen 2.5-7B,
Mistral-7B, Gemma-2-9B, GPT-4o, GPT-5.4, Claude Sonnet 4.6) on the
synthetic benchmark, since the synthetic data does not include
visual inputs. The full multimodal **VLM** pipeline is exercised
qualitatively on a real CTAF recording at Half Moon Bay Airport
(KHAF) using Gemini 2.5 Pro — see Sec. V of the paper.

## Repository layout

```
.
├── generate_dataset.py            # Build CTAF-KHAF-Synthetic from the taxonomy
├── postprocess_dataset.py         # Post-process generated scenarios
├── run_experiments.py             # Main inference / evaluation driver
├── claude_inference.py            # Anthropic API wrapper
├── openai_inference.py            # OpenAI API wrapper
├── ablation_whisper.py            # Ablation 1: ASR model size
├── ablation_audio_noise.py        # Ablation 2: additive audio noise
├── ablation_text_mask.py          # Ablation 3: transcript word/utterance masking
├── make_paper_assets.py           # Rebuild every figure and table from raw JSONs
├── make_figures.py                # Earlier figure script (kept for reproducibility)
├── make_task_icons.py             # PR/ROC/CM/softmax icons for the architecture diagram
├── make_architecture.py           # Architecture-diagram figure
├── rescore_records.py             # Patch missing class_scores with confidence fallback
├── rescore_hf_logprobs.py         # Constrained-scoring rescore for HF models
├── visualize_trajectories.py      # ADS-B trajectory visualization helper
├── run_overnight.sh               # Overnight runner: GPT-4o + ablations
├── run_overnight_binary.sh        # Overnight runner: binary-task suite
├── paper_assets/                  # Final figures + LaTeX tables used in the paper
└── figures/                       # Architecture pdf/png/pptx
```

## Dataset

The CTAF-KHAF-Synthetic benchmark (100 scenarios) is **not stored in
this repository**. The audio files and per-scenario directories are
hosted on Google Drive:

> **Download CTAF-KHAF-Synthetic:** `<(https://drive.google.com/drive/folders/1ehX4EBxHLhKgYR58MC2kbVj1_KvGyjn5?usp=sharing)>`

After downloading, unpack so that the directory layout becomes:

```
dataset/
├── ctaf_khaf_synthetic_v2.json        # scenario metadata (100 entries)
└── scenarios/
    ├── S001/
    │   ├── audio.mp3
    │   └── transcript_ground_truth.srt
    ├── S002/
    │   ├── audio.mp3
    │   └── transcript_ground_truth.srt
    └── ...
```

Each scenario record in `ctaf_khaf_synthetic_v2.json` carries:
- a ground-truth safety label (`nominal` / `warning` / `hazard`),
- a hazard-type label drawn from a 12-category taxonomy,
- raw and decoded METAR text,
- a CTAF radio transcript in SRT format with timestamps,
- per-aircraft ADS-B state vectors,
- a path to the synthesized multi-voice audio (`audio.mp3`).

The audio is generated from the transcripts using OpenAI's TTS-1-HD,
with a distinct voice assigned to each aircraft on the frequency.
Every scenario was reviewed by human experts with general-aviation
flight experience to verify that the assigned label is unambiguous
and the radio call sequence is plausible in real CTAF operations.

Six held-out scenarios (two per class) are reserved as in-context
learning exemplars; the remaining 94 form the test set used in
every experiment reported in the paper.

## Experiment artifacts (`results/`, `results_binary/`)

The raw per-prediction JSONs, aggregated metrics, confusion matrices,
and intermediate figures from the full benchmark and the three
ablations are also hosted on Drive:

> **Download experiment artifacts:** `<INSERT GOOGLE DRIVE LINK HERE>`

These are only needed if you want to reproduce the paper's plots
without re-running the full sweep. To re-derive the paper's figures
and tables from these artifacts:

```bash
python make_paper_assets.py
```

## Quick start: re-run the LLM benchmark

1. Set the API keys for any closed-source models you want to run:

   ```bash
   export OPENAI_API_KEY=sk-...
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

2. Place the dataset under `dataset/` as described above.

3. Run a single condition (fast, no GPU required if you only call
   the closed-source models):

   ```bash
   python run_experiments.py --dataset dataset/ctaf_khaf_synthetic_v2.json \
       --models gpt-4o --strategies zero_shot
   ```

4. Run the full 36-condition sweep on the open-source models (GPU
   required, ~6 h total on an RTX 5090):

   ```bash
   python run_experiments.py --dataset dataset/ctaf_khaf_synthetic_v2.json \
       --models qwen mistral gemma
   ```

5. Run the binary-task suite:

   ```bash
   python run_experiments.py --dataset dataset/ctaf_khaf_synthetic_v2.json \
       --binary --output-dir results_binary
   ```

6. Run all three robustness ablations:

   ```bash
   python ablation_whisper.py
   python ablation_audio_noise.py
   python ablation_text_mask.py
   ```

7. Rebuild every figure and LaTeX table the paper uses:

   ```bash
   python make_paper_assets.py
   ```

## Hardware notes

- Open-source models (Qwen 2.5-7B, Mistral-7B-Instruct-v0.3,
  Gemma-2-9B-IT) run locally in fp16 with no quantization. We used
  an RTX 5090 (32 GB VRAM); they each fit comfortably.
- Closed-source models (GPT-4o, GPT-5.4, Claude Sonnet 4.6) are
  accessed through their respective HTTP APIs.


