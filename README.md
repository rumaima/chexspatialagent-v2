# CXR SpatialAgent

Training-free multi-agent spatial reasoning for chest X-rays. No API key required.

```
r_PE = Φsum( Φexec( Φplan(q, v; T) ), q, v )
```

## Model Stack

| Role | Model | Install |
|---|---|---|
| Pathology classification | TorchXRayVision DenseNet121 | `pip install torchxrayvision` |
| Lung / structure segmentation | TorchXRayVision PSPNet | same |
| Finding segmentation | **MedSAM ViT-B** | clone + `pip install -e .` |
| Open-vocab structure detection | **Grounding DINO** | `pip install transformers` |
| Planner + tool simulation | **CheXagent-2-3b** | auto-downloads (~6 GB) |
| Clinical summarizer | **Qwen2.5-VL-7B** (4-bit) | auto-downloads (~5 GB) |

**No Anthropic API key needed.** All models run locally.

GPU requirements: 12–16 GB VRAM for real mode (CheXagent + Qwen simultaneously).
CPU fallback available for TorchXRayVision and Grounding DINO.

---

## Setup (5 steps)

```bash
# 1. Clone and enter
git clone https://github.com/rumaima/chexspatialagent.git
cd chexspatialagent

# 2. Install core dependencies
pip install torch torchvision torchxrayvision
pip install transformers>=4.45.0 accelerate bitsandbytes qwen-vl-utils
pip install scikit-image scipy Pillow

# 3. Install MedSAM
git clone https://github.com/bowang-lab/MedSAM.git
cd MedSAM && pip install -e . && cd ..

# 4. Download MedSAM checkpoint (~357 MB)
python scripts/download_models.py

# 5. Run
python pipeline.py --image cxr.jpg --question "Is there pleural effusion?"
```

---

## Run

```bash
# Real mode — uses DL models (recommended, requires GPU)
python pipeline.py --image cxr.jpg --question "Where is the consolidation located?"

# Simulated mode — CheXagent-2 acts as all tools (no MedSAM weights needed)
python pipeline.py --image cxr.jpg --question "..." --mode simulated

# Preview the plan without running
python pipeline.py --image cxr.jpg --question "..." --plan-only

# Save full output as JSON
python pipeline.py --image cxr.jpg --question "..." --output result.json
```

---

## Spatial Question Types

| Type | Example |
|---|---|
| Presence | `Is pleural effusion present in the chest X-ray?` |
| Location | `Where is the consolidation located?` |
| Laterality | `Is the effusion more extensive in the right or left lung?` |
| Distribution | `What is the distribution pattern of the consolidation: focal, diffuse...?` |
| Containment | `Is the mass contained within the lung field?` |
| Relative position | `What is the relative position of the mass with respect to the hilum?` |
| Device tip | `Where is the tip of the endotracheal tube relative to the carina?` |
| Border/silhouette | `Does the consolidation involve or obscure the left hemidiaphragm?` |

---

## How It Works

1. **Φplan** — classifies question type (14 templates) → deterministic tool chain.
   Free-form questions fall back to CheXagent-2 plan generation.

2. **Φexec** — runs each tool sequentially:
   - `image_quality_assessor` — pixel heuristics (no model)
   - `lung_region_detector` — TorchXRayVision PSPNet 14-structure segmentation
   - `opacity_segmenter` — TorchXRayVision (classify) → Grounding DINO (locate) → MedSAM (segment)
   - `tube_line_localizer` — Grounding DINO (detect) → MedSAM (segment) → geometry (tip distance)
   - `trachea_mediastinum_analyzer` — PSPNet + Grounding DINO for reference structures
   - `airspace_density_mapper` — PSPNet zone grid + intensity
   - `differential_ranker` — TorchXRayVision 18-label probability ranking

3. **Φsum** — Qwen2.5-VL-7B-Instruct synthesises all tool outputs into a structured clinical report.

---

## Project Structure

```
chexspatialagent/
├── pipeline.py                    # Entry point
├── config.py                      # All model paths and thresholds
├── requirements.txt
├── agents/
│   ├── planner.py                 # Rule-based + CheXagent-2 fallback
│   ├── executor.py                # Real DL or CheXagent simulation
│   └── summarizer.py              # Qwen2.5-VL-7B
├── tools/
│   ├── general_perception/        # Lung segmentation, opacity, cardiac
│   ├── spatial_analysis/          # Zone maps, CP angles, mediastinum
│   ├── geometry/                  # Device localisation, pleural, ribs
│   └── auxiliary/                 # Quality, differential, terminate
├── utils/
│   ├── model_loader.py            # All 6 model loaders (lazy singletons)
│   ├── spatial_geometry.py        # Distances, zones, containment
│   └── question_router.py         # Question template parser
├── scripts/
│   └── download_models.py         # Downloads MedSAM checkpoint
└── model_weights/                 # Place medsam_vit_b.pth here
```
