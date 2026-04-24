# Florence-2 ALPR Integration Plan
## SadakSathi — Automatic License Plate Recognition Pipeline

---

## Overview

This document describes the complete strategy for integrating **Microsoft Florence-2-base** as the OCR backbone for the SadakSathi traffic violation complaint management system.

The pipeline is architected in two fully decoupled stages:
1. **YOLO26 (Fast)** — Detects violations (No Helmet, Triple Riding) and localizes License Plates in ~40ms
2. **Florence-2 (Accurate)** — Reads the License Plate text from YOLO's crop in ~4 seconds (runs asynchronously in the background)

---

## Why Florence-2 Over Traditional OCR

| Feature | PaddleOCR | Florence-2 Base |
|:--|:--|:--|
| Architecture | Mathematical shape contours | Vision-Language Model (VLM) |
| Accuracy on blurry crops | Poor (fails on sub-64px) | Excellent (contextual reasoning) |
| Handles inverted plates | No | Yes (spatial context) |
| CPU friendly | Yes (~0.05s/image) | Yes (~4s/image) |
| Hallucination resistance | Low | High |
| Indian HSRP plate support | Via regex correction only | Native + regex correction |
| GPU required | No | No (runs on CPU/CUDA both) |

**Verdict:** Florence-2 was empirically benchmarked on 100 real YOLO plate crops and significantly outperformed PaddleOCR in raw text extraction accuracy.

---

## Architecture: The Two-Stage Decoupled Pipeline

```
CCTV Frame / Uploaded Image
         │
         ▼
  ┌─────────────┐
  │  YOLO26s    │  ~40ms
  │  Detection  │
  └──────┬──────┘
         │
    ┌────┴─────────────────────┐
    │                          │
    ▼                          ▼
[Violation Detected]    [Plate Bounding Box]
(NoHelmet/Triple)       (Crop saved to memory)
    │                          │
    ▼                          ▼
[Register PENDING      [Background Task Queue]
 Complaint in DB]             │
    │                         ▼
    │                 ┌───────────────┐
    │                 │  Florence-2   │  ~4s per plate
    │                 │  OCR Engine   │
    │                 └───────┬───────┘
    │                         │
    │                         ▼
    │               [Indian RegEx Validator]
    │               AA-##-AAA-#### format
    │                         │
    └──────────────┬──────────┘
                   ▼
         [Update Complaint Record]
         [Attach Final Plate Text]
```

---

## Stage 1: YOLO26 Detection (Real-Time)

**File:** `backend/ml/traffic.py`

**What it does:**
- Receives an uploaded image or video frame
- Runs YOLO26s inference (~40ms)
- Identifies 4 classes: `WithHelmet`, `WithoutHelmet`, `TripleRiding`, `Plate`
- For every `WithoutHelmet` or `TripleRiding` detection, checks if a `Plate` bounding box overlaps nearby
- Crops the Plate region and saves it as a PIL image in memory
- Returns the list of violations immediately to the API layer

**YOLO Output to FastAPI:**
```json
{
  "violations": [
    {
      "type": "WithoutHelmet",
      "confidence": 0.91,
      "bbox": [120, 45, 300, 200],
      "plate_crop": "<PIL Image Object>",
      "complaint_id": "uuid-pending"
    }
  ]
}
```

---

## Stage 2: Florence-2 OCR (Background — Async)

**File:** `backend/ml/florence_alpr.py` *(New file to create)*

### 2a. Model Loading (Singleton Pattern)
Florence-2 takes ~8 seconds to load from disk on first run. We load it once at application startup and cache it in `app.state` — exactly how `traffic_model` is currently handled in `main.py`.

```python
# In backend/main.py — startup event
@app.on_event("startup")
async def load_models():
    app.state.traffic_model = YOLO("yolo26s_best.pt")
    app.state.florence_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base", trust_remote_code=True
    ).eval()
    app.state.florence_processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base", trust_remote_code=True
    )
```

### 2b. Core Inference Function
```python
def run_florence_ocr(model, processor, pil_image: Image) -> str:
    task_prompt = '<OCR>'
    inputs = processor(text=task_prompt, images=pil_image, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=128,
            num_beams=3
        )
    raw_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        raw_text, task=task_prompt,
        image_size=(pil_image.width, pil_image.height)
    )
    return parsed.get('<OCR>', '')
```

### 2c. Indian Plate RegEx Validator
```python
import re

def clean_indian_plate(text: str) -> str:
    """
    Validates and corrects OCR output against Indian HSRP format:
    [State: 2 Letters] [District: 1-2 Digits] [Series: 1-3 Letters] [Serial: 4 Digits]
    Example: MH12AB1234 | DL8CAF5032 | KA03MN4567
    """
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())

    if len(clean) < 6 or len(clean) > 10:
        return clean  # Return as-is if too short/long to validate

    corrected = list(clean)

    # First 2 chars must be State Letters
    NUM_TO_LET = {'0':'O','1':'I','2':'Z','4':'A','5':'S','8':'B'}
    for i in [0, 1]:
        if corrected[i] in NUM_TO_LET:
            corrected[i] = NUM_TO_LET[corrected[i]]

    # Last 4 chars must be Serial Number Digits
    LET_TO_NUM = {'O':'0','Q':'0','D':'0','I':'1','L':'1','Z':'2','A':'4','S':'5','B':'8'}
    for i in range(max(2, len(corrected) - 4), len(corrected)):
        if corrected[i] in LET_TO_NUM:
            corrected[i] = LET_TO_NUM[corrected[i]]

    return "".join(corrected)
```

---

## Stage 3: FastAPI Background Task Integration

**File:** `backend/routers/traffic.py`

The key architectural principle: **The HTTP response must return instantly.** Florence-2 OCR runs in a background thread after the response is sent.

```python
from fastapi import BackgroundTasks

@router.post("/image")
async def detect_traffic_image(
    request: Request,
    file: UploadFile,
    background_tasks: BackgroundTasks,
):
    # Step 1: Run YOLO26 (fast — 40ms)
    violations, plate_crops = assess_traffic_image(image_bytes, model=traffic_model)

    # Step 2: Register pending complaints immediately
    complaint_ids = await register_pending_complaints(violations)

    # Step 3: Offload Florence-2 OCR to background (non-blocking)
    background_tasks.add_task(
        process_plates_with_florence,
        plate_crops=plate_crops,
        complaint_ids=complaint_ids,
        florence_model=request.app.state.florence_model,
        florence_processor=request.app.state.florence_processor,
    )

    # Step 4: Return instantly to the user
    return {
        "success": True,
        "violations_detected": len(violations),
        "ocr_status": "processing_in_background",
        "complaint_ids": complaint_ids
    }
```

---

## Stage 4: Database Schema Update (Prisma)

The `Complaint` table needs a new field to handle the async OCR state:

```prisma
model Complaint {
  id              String   @id @default(cuid())
  violationType   String   // "WithoutHelmet" | "TripleRiding"
  detectedAt      DateTime @default(now())

  // License Plate Fields (populated asynchronously)
  licensePlate    String?  // null until Florence-2 completes
  plateOcrStatus  String   @default("PENDING")  // PENDING | COMPLETED | FAILED
  plateRawOcr     String?  // raw Florence-2 output before regex cleanup

  imageUrl        String?
  confidence      Float
}
```

---

## Files to Create / Modify

| File | Action | Description |
|:--|:--|:--|
| `backend/ml/florence_alpr.py` | **CREATE** | Florence-2 model loader, inference, RegEx validator |
| `backend/main.py` | **MODIFY** | Load Florence-2 into `app.state` at startup |
| `backend/routers/traffic.py` | **MODIFY** | Add `BackgroundTasks` parameter, offload OCR |
| `backend/ml/traffic.py` | **MODIFY** | Return plate crops alongside violation detections |
| `prisma/schema.prisma` | **MODIFY** | Add `licensePlate`, `plateOcrStatus`, `plateRawOcr` fields |
| `requirements.txt` | **MODIFY** | Add `transformers==4.41.0`, `torch` (cpu), `einops`, `timm` |

---

## Dependency Requirements

```
# Florence-2 Stack (add to SadakSathi requirements.txt)
torch==2.3.1+cpu  --index-url https://download.pytorch.org/whl/cpu
torchvision==0.18.1+cpu  --index-url https://download.pytorch.org/whl/cpu
transformers==4.41.0
einops==0.8.2
timm==1.0.26
```

> **Note:** Use `torch==2.3.1+cpu` specifically. The generic `torch` pip install downloads the CUDA version which crashes on Windows with `[WinError 127] shm.dll`. This specific CPU wheel has been validated on the development machine.

---

## Performance Expectations

| Stage | Model | Time |
|:--|:--|:--|
| Detection | YOLO26s | ~40ms |
| OCR (Background) | Florence-2-base | ~4s per plate |
| API Response to User | — | ~50ms (instant) |
| DB Update after OCR | — | ~4s after response |

**Overall UX:** The user sees the violation detected instantly. The license plate number appears on their dashboard within ~4 seconds as the background job completes and updates the complaint record.

---

## Next Steps (Execution Order)

- [ ] Create `backend/ml/florence_alpr.py`
- [ ] Modify `backend/main.py` to preload Florence-2 at startup
- [ ] Modify `backend/routers/traffic.py` to use `BackgroundTasks`
- [ ] Update Prisma schema with plate OCR fields
- [ ] Run `prisma migrate dev` to apply DB changes
- [ ] Update `requirements.txt` in SadakSathi backend
- [ ] Integration test: Upload violation image → verify complaint created → verify plate populated within 5s
