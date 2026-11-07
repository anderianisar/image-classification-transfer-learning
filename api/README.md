# Malaria Cell Classifier + Case Note Analyzer API

**Project 2 (Week 8): a tested web service for a malaria blood-cell image model and a clinical-note analyzer.**

This service takes a blood-cell image and an optional clinical note and returns one structured JSON record: the model's prediction, a privacy-masked note, extracted fields, urgent warning words, and a `REFER` / `ROUTINE` decision for a human to review. It serves my Week 6 image model (ResNet-18) together with my Week 7 text-evidence rules.

> **Decision support only. A qualified human decides.** This is a student project and must not be used to diagnose patients.

**Author:** Anderia Nisar | **Course:** Introduction to Applied AI, BS 8th Semester, Department of Electronics

---

## Contents
1. [Features](#features)
2. [Project structure](#project-structure)
3. [Model](#model)
4. [Quick start](#quick-start)
5. [API reference](#api-reference)
6. [Error handling](#error-handling)
7. [Testing](#testing)
8. [Performance and capacity](#performance-and-capacity)
9. [Privacy and security](#privacy-and-security)
10. [Check Your Understanding](#check-your-understanding)
11. [Reflection](#reflection)
12. [Limitations and next steps](#limitations-and-next-steps)

---

## Features
- Three endpoints: `GET /health`, `POST /predict`, `POST /analyze`, with typed request and response schemas and automatic interactive docs at `/docs`.
- Model is loaded **once at startup** (FastAPI `lifespan`); inference runs in a thread pool so the event loop stays free.
- Preprocessing is built from the settings saved inside the checkpoint, so serving and training use the same transform.
- Defensive uploads: file-type allowlist, 5 MB cap, safe image decoding, clean 4xx errors and no stack traces.
- Privacy: CNIC and mobile numbers are masked **before** any other processing.
- 8 automated tests (including parity with the direct model and a PII-leak test) and a latency benchmark.
- Optional API-key protection through the `X-API-Key` header.

## Project structure
```
p2-api/
├── app.py             FastAPI service (/health, /predict, /analyze)
├── test_api.py        8 automated tests (pytest)
├── bench.py           Latency benchmark against a running server
├── requirements.txt   Python dependencies (PyTorch installed separately)
├── sample.jpeg        Test image used by the tests and the benchmark
├── week6_model.pt     Week 6 checkpoint (about 44 MB; place it here before running)
└── README.md
```

## Model
| Item | Value |
|------|-------|
| Backbone | ResNet-18 (transfer learning) |
| Input | 224 x 224 RGB, ImageNet normalisation |
| Classes | `Parasitized`, `Uninfected` |
| Positive class (must not be missed) | `Parasitized` |
| Decision threshold | **0.160** (`THRESHOLD`), chosen on the validation set for recall of at least 0.95; it is not stored in the checkpoint |
| Checkpoint | `week6_model.pt`, contains `state_dict`, `classes`, `backbone`, `img_size`, `grayscale` |

### Data and training (Week 6)
| Item | Result |
|------|--------|
| Dataset | Cell Images for Detecting Malaria (NIH, via Kaggle): 27,558 RGB images, Parasitized 13,779 / Uninfected 13,779 |
| Patients | 200 identified from file names (150 have both classes, 50 Uninfected only) |
| Exact duplicates (MD5) | 0 |
| Near-duplicates (32 x 32 fingerprint) | 0 below distance 0.05; 234 of 3,335 test images below 0.10, and the closest pairs were look-alike cells from different patients, not copies |
| Split | **By patient** (`StratifiedGroupKFold`): train 19,118 images / 139 patients; validation 5,105 / 33; test 3,335 / 28; no patient appears in two splits |
| Training | Phase 1: head only (1,026 trainable parameters), best validation macro-F1 0.866. Phase 2: fine-tune `layer4` + head (8,394,754 trainable parameters), best validation macro-F1 0.937 |
| Augmentation | Random resized crop, rotation up to 10 degrees, horizontal and vertical flips, mild brightness and contrast jitter |

**Test-set results** (3,335 cells from 28 patients, test set used once)

At the default 0.5 cut-off:

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Parasitized | 0.963 | 0.962 | 0.963 | 1,340 |
| Uninfected | 0.974 | 0.975 | 0.975 | 1,995 |
| Accuracy / macro-F1 | | | 0.970 / 0.969 | 3,335 |

| Confusion matrix | Predicted Parasitized | Predicted Uninfected |
|------------------|-----------------------|----------------------|
| **True Parasitized** | 1,289 | 51 |
| **True Uninfected** | 49 | 1,946 |

At the high-recall threshold **0.160** (chosen on validation, applied once to test):

| Metric | Value |
|--------|-------|
| Sensitivity (recall) | 0.982 (24 infected cells missed, against 51 at 0.5) |
| Specificity | 0.941 (117 false alarms, against 49 at 0.5) |
| PPV (precision) | 0.918 at the test-set prevalence of 40.2% |
| AUC | 0.994 |
| PPV at 5% prevalence | 0.468 |
| PPV at 1% prevalence | 0.145 |

The service uses 0.160 because missing an infected cell is the costly error; the price is more false alarms, and at realistic (low) prevalence most alarms would be false, so every `REFER` needs human review.

**Grad-CAM check:** on 300 test images, 14.5% of the heatmap attention fell on the black background, which covers 26.6% of the pixels, so the model mostly looks at the cell. A mild background shortcut is suspected but not confirmed; some heatmaps for errors sat on the cell rim instead of the parasite.

**Urgent words checked in notes:** fever, chills, rigors, sweating, headache, vomiting, breath, confusion, jaundice.

## Quick start

**Requirements:** Python 3.10+, about 2 GB of free disk space. CPU only; no GPU needed.

```
python -m venv .venv
.venv\Scripts\activate                 # Mac/Linux: source .venv/bin/activate
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# place week6_model.pt and sample.jpeg in this folder
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000/docs> and use **Try it out**.

**Settings (environment variables):**

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_PATH` | `week6_model.pt` | Checkpoint file |
| `POSITIVE` | `Parasitized` | Class that must not be missed |
| `THRESHOLD` | `0.16` | P(positive) at or above this triggers a referral reason (Week 6 high-recall threshold) |
| `API_KEY` | not set | If set, requests need an `X-API-Key` header |

## API reference

### `GET /health`
Returns `status`, `model_loaded`, `classes` and the `positive` class.

### `POST /predict`
Upload one JPEG or PNG image (multipart field `file`).
```
curl -X POST http://127.0.0.1:8000/predict ^
     -F "file=@sample.jpeg;type=image/jpeg"
```
```json
{
  "label": "Parasitized",
  "probabilities": { "Parasitized": 1.0, "Uninfected": 0.0 },
  "latency_ms": 256.1
}
```
The first request after startup includes warm-up time; later requests are faster (see benchmark).

### `POST /analyze`
Upload an image (`file`) and a text note (form field `note`, up to 2,000 characters).
```
curl -X POST http://127.0.0.1:8000/analyze ^
     -F "file=@sample.jpeg;type=image/jpeg" ^
     -F "note=Fever and chills since Monday. Call 0300-1234567"
```
```json
{
  "prediction": {
    "label": "Parasitized",
    "probabilities": { "Parasitized": 1.0, "Uninfected": 0.0 },
    "latency_ms": 39.1
  },
  "note_masked": "Fever and chills since Monday. Call [PHONE]",
  "fields": {},
  "urgent_terms": ["fever", "chills"],
  "decision": "REFER",
  "reasons": [
    "image: P(Parasitized) = 1.00",
    "note mentions: fever, chills"
  ],
  "disclaimer": "Decision support only; a qualified human decides."
}
```
- `decision` is `REFER` if the image probability for the positive class reaches the threshold **or** the note contains an urgent word; otherwise `ROUTINE`.
- `fields` is empty here because phone numbers and CNICs are masked before extraction. Amounts (PKR/Rs) and dates, if present, are still extracted.

*(On Mac/Linux use `\` instead of `^` to continue a line.)*

## Error handling
Every failure returns a short, fixable message. No stack traces are exposed.

| Bad input | Status | Response |
|-----------|--------|----------|
| Wrong file type (PDF, text) | 415 | `{"detail":"Upload a JPEG or PNG image"}` |
| Image larger than 5 MB | 413 | `{"detail":"Image larger than 5 MB"}` |
| File is not a readable image | 400 | `{"detail":"File is not a readable image"}` |
| Missing file or note | 422 | FastAPI validation error |
| Note longer than 2,000 characters | 422 | `{"detail":"Note longer than 2,000 characters"}` |
| Wrong or missing API key (only if `API_KEY` is set) | 401 | `{"detail":"Invalid or missing API key"}` |

**Bad inputs I tried by hand, beyond the automated tests:**

| Input | Result |
|-------|--------|
| `app.py` uploaded as `text/plain` to `/predict` | `415` `{"detail":"Upload a JPEG or PNG image"}` |
| `fake.jpg`, a text file renamed to `.jpg` (content: "not an image"), sent as `image/jpeg` | `400` `{"detail":"File is not a readable image"}` |

The second case matters because the declared content type was correct but the bytes were not: the API checks the actual image data, not only the label.

## Testing
```
pytest -q test_api.py
........                                                  [100%]
8 passed, 1 warning in 11.14s
```
The single warning is a library deprecation notice (httpx with Starlette's `TestClient`) and does not affect results.

| Test | What it checks | Type |
|------|----------------|------|
| `test_health` | Server starts and the model is loaded | Smoke |
| `test_predict_schema_and_probabilities` | 200 response; probabilities sum to 1 | Contract |
| `test_parity_with_direct_model` | API probabilities equal calling the model directly | Parity |
| `test_rejects_wrong_type` | Text file gives 415 | Robustness |
| `test_rejects_too_large` | Over-size upload gives 413 | Robustness |
| `test_rejects_corrupt_image` | Unreadable image gives 400 | Robustness |
| `test_missing_file_is_422` | Missing file gives 422 | Contract |
| `test_analyze_masks_pii` | No part of the phone number or CNIC appears anywhere in the response; decision is `REFER` | Privacy |

## Performance and capacity
**Setup:** 50 sequential requests to `/predict` against a running server (`python bench.py`), one worker, CPU only, one client.
**Hardware:** AMD Ryzen 5 7430U (6 cores), 7.4 GB RAM (as reported by Windows), Windows, CPU only (no GPU used)

| Metric | Value |
|--------|-------|
| p50 | 35 ms |
| p95 | 42 ms |
| max | 80 ms |
| Throughput, 1 worker | about 27 requests/s |

**Little's law** (`L = λ · W`). With one worker busy at a time (L = 1) and W = p50 = 0.035 s, λ ≈ 1 / 0.035 ≈ **28.6 requests/s**. Over a day that is 28.6 x 86,400 ≈ **2.5 million requests per day** per worker if traffic were perfectly even (the measured mean gives about 27.2 requests/s, or about 2.35 million per day).

**Caveats.** The tail is tight (p95 42 ms against p50 35 ms) because requests were sent one after another by one client. Under many simultaneous requests the tail would grow, and real traffic arrives in bursts, so safe capacity is lower than the even-traffic estimate. Even so, a student-scale product is nowhere near a capacity limit; correctness and reliability matter far more.

## Privacy and security
- **PII masking first.** `/analyze` replaces CNIC numbers with `[CNIC]` and mobile numbers with `[PHONE]` before extracting fields or checking urgent words. The test suite checks the entire response for leaked digits.
- **Upload limits.** Type allowlist (JPEG/PNG), 5 MB cap, and safe decoding with Pillow.
- **No internals leaked.** Errors are clear 4xx messages; a 500 would mean a bug and is never returned on purpose.
- **Secrets.** The API key (optional) is read from an environment variable and must never be committed to GitHub.
- **Not deployed.** The service runs locally; a public deployment would need HTTPS.

## Check Your Understanding
1. **A client uploads a PDF to `/predict`. Which status code, and why not 500?** `415 Unsupported Media Type`. The client sent the wrong kind of file, which is a client error (4xx) that the client can fix. A 500 means the server failed because of a bug and tells the client nothing useful.
2. **Why is the model loaded in `lifespan` rather than inside the endpoint?** Loading weights and rebuilding the network takes seconds, while a request should take milliseconds. Loading once at startup means each request only runs the forward pass, and memory is not wasted reloading the model.
3. **p50 = 40 ms and p95 = 400 ms. What does that tell you, and what would you check?** Most requests are fast, but about 1 in 20 is ten times slower; the average hides this tail, and users feel the slow ones. I would check queueing under concurrent load, a blocked event loop, cold starts, very large inputs, and other programs using the CPU.
4. **500,000 requests a day, peak 5x the average, 25 requests/s per worker. How many workers?** Average = 500,000 / 86,400 ≈ 5.8 req/s; peak = 5 x 5.8 ≈ 29 req/s; workers = 29 / 25 ≈ 1.16, so **2 workers**.
5. **Why can model inference inside an `async def` endpoint slow down other requests?** An async server runs one event loop that only switches tasks at an `await`. Heavy CPU work with no `await` blocks the loop, so every other request, even `/health`, waits. `run_in_threadpool` (or a plain `def` endpoint) keeps the loop free.
6. **How many communication channels does a 4-person team have?** n(n - 1) / 2 = 4 x 3 / 2 = **6**.

## Reflection
- **Installing PyTorch was the first obstacle.** With only the CPU-only index, `pip` could not fetch some dependencies (`typing-extensions`). Adding the CPU index as an **extra** index fixed it, and a slow connection then caused download timeouts, which I solved by retrying with a longer timeout. Environment setup is part of the engineering, not a side issue.
- **A checkpoint is a file, not a folder.** A `.pt` file is a zip archive. Extracting it turned it into a folder, so `torch.load` could not open it; a stray space in the file name caused a second failure. I now confirm a checkpoint by loading it and printing its keys before serving it.
- **A probability of 1.0 is a demo, not evidence.** The model returned P(Parasitized) = 1.00 on my sample image. A single confident prediction says nothing about accuracy; the evidence is the test-set metrics, and I should demo with an image the model has never seen.
- **The tail matters more than the mean.** Measuring p50 and p95 together with hardware showed how steady the service is under sequential load, and why I should always test concurrent load before promising capacity.
- **Tests caught the things that matter.** The parity test proves the API gives the same numbers as the notebook, and the PII test proves private numbers never leave the service.

## Limitations and next steps
**Limitations**
- Trained on cell images from one dataset; performance on images from other microscopes, stains or labs is unknown. The model classifies single cell crops, not whole slides or patients.
- The test set is 40% infected. At 5% prevalence the PPV falls to 0.47 and at 1% to 0.15, so most alarms would be false in a low-prevalence setting.
- Grad-CAM showed a mild, unconfirmed background shortcut, and some errors had heat on the cell rim rather than the parasite.
- The 0.160 threshold lives in the service settings, not in the checkpoint, so it must be kept in sync with the model.
- The urgent-word list is a simple keyword rule, not a clinical model, and the PII patterns cover Pakistani CNIC and mobile formats only.
- Single worker, local deployment, no authentication by default.

**Next steps**
- Store the decision threshold inside the checkpoint so the model and its threshold cannot drift apart.
- Re-check the threshold on data from a second source before any real use.
- Turn on the API key and deploy behind HTTPS.
- Add Grad-CAM output so a reviewer can see what the model looked at.
- Test under concurrent load before claiming capacity.
