# ICARE–GAT IOP Prediction: Data Processing Pipeline

This project prepares a clinical dataset for machine-learning models that estimate Goldmann Applanation Tonometry (GAT) IOP from iCare IOP and Central Corneal Thickness (CCT).

The current dataset does not contain age or gender, so those variables are intentionally excluded from the pipeline.

## Table of Contents

- [Project Objective](#project-objective)
- [Data Pipeline Overview](#data-pipeline-overview)
- [Repository Structure](#repository-structure)
- [Stage-by-Stage Processing](#stage-by-stage-processing)
- [Running the Pipeline](#running-the-pipeline)
- [When to Rerun Stages](#when-to-rerun-stages)
- [Key Data Decisions](#key-data-decisions)
- [Current Dataset Status](#current-dataset-status)
- [Reproducibility](#reproducibility)
- [Expected ML Workflow](#expected-ml-workflow)
- [Core Principle](#core-principle)

---

## Project Objective

The working dataset contains intraocular pressure (IOP) measurements obtained using:

- iCare tonometry: model input measurement
- Applanation/GAT tonometry: reference/target measurement
- CCT (Central Corneal Thickness): additional model feature

The raw data originates from an Excel file containing records for patients and both eyes.

For every usable observation, the intended final structure is:

```text
date | euid | icare_iop | cct | applanation_iop
```

Where:

- `date` = date of the paired measurement
- `euid` = Eye Unique ID
- `icare_iop` = iCare IOP measurement
- `cct` = Central Corneal Thickness in micrometres (µm)
- `applanation_iop` = Applanation/GAT IOP measurement and prediction target

The right and left eyes are treated as separate observations for model training.

---

## Data Pipeline Overview

The pipeline is intentionally sequential. Each stage depends on the output of the previous stage.

```text
Raw Excel
  ↓
round1_cleaning.py
  ↓
icare_cct_dataset_round1_cleaned.csv
  ↓
remove_duplicates.py
  ↓
paired_iop_cct_round1_deduplicated.csv
  ↓
clean_ranges.py
  ↓
final_cleaned_dataset.csv
  ↓
remove_repeated_euid.py
  ↓
final_unique_euid_dataset.csv
  ↓
split_dataset.py
  ↓
train.csv
validation.csv
test.csv
```

> Do not skip stages unless the corresponding output has already been generated and you are intentionally rerunning the pipeline from that point.

---

## Repository Structure

```text
icare-ml/
├── .venv/
├── data/
│   ├── raw/
│   │   └── <original Excel dataset>
│   ├── cleaned/
│   │   ├── icare_cct_dataset_round1_cleaned.csv
│   │   ├── paired_iop_cct_round1_deduplicated.csv
│   │   ├── final_cleaned_dataset.csv
│   │   ├── final_unique_euid_dataset.csv
│   │   └── invalid_ranges.csv
│   └── splits/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── models/
├── notebooks/
├── results/
│   ├── metrics/
│   └── plots/
├── src/
│   ├── round1_cleaning.py
│   ├── remove_duplicates.py
│   ├── clean_ranges.py
│   ├── remove_repeated_euid.py
│   └── split_dataset.py
├── README.md
├── requirements.txt
└── .gitignore
```

The `.venv` directory contains the project's isolated Python environment and should not be committed to version control.

---

## Stage-by-Stage Processing

### 1. Clinical Measurement Pairing

Script: `src/round1_cleaning.py`

Input: the original Excel dataset.

The raw data includes fields such as:

- `uid`
- `mr_no`
- `LE_method`
- `RE_method`
- `LE_value`
- `RE_value`
- `CCT_RE_value`
- `CCT_LE_value`
- `iop_recorded_at`
- `cct_recorded_at`

Notes:

- `uid` is the unique patient identifier across the hospital chain.
- `mr_no` is a hospital-specific patient identifier and is redundant for this project.

Processing:

- remove the redundant `mr_no` field from working data
- convert right-eye and left-eye records into separate eye-level observations
- identify iCare measurements and Applanation measurements
- require iCare to occur before Applanation
- require both measurements to occur on the same calendar day
- require the time between iCare and Applanation to be no more than 3 hours
- associate the correct CCT measurement with each eye
- create an eye-level identifier (`euid`)
- produce the first paired dataset

> The 3-hour requirement is a deliberate clinical constraint and should not be changed casually.

Output:

```text
data/cleaned/icare_cct_dataset_round1_cleaned.csv
```

This output contains valid iCare → Applanation measurement pairs.

### 2. EUID: Eye Unique ID

The dataset uses an EUID to distinguish the two eyes of a patient.

Format:

```text
L<uid>
R<uid>
```

Examples:

- `L18255471` → left eye of patient `18255471`
- `R18255471` → right eye of patient `18255471`

This allows the two eyes to be treated as separate observations while preserving the patient-level link needed for safe splitting.

### 3. Exact Duplicate Removal

Script: `src/remove_duplicates.py`

Input: `data/cleaned/icare_cct_dataset_round1_cleaned.csv`

Purpose:

This stage removes observations that are exact copies of another observation. The duplicate definition is based on the complete measurement:

```text
date | euid | cct | icare_iop | applanation_iop
```

If all of these values are identical, the second copy does not add information and is removed.

Example:

```text
2026-03-02 | L123 | 540 | 18 | 20
2026-03-02 | L123 | 540 | 18 | 20
```

becomes:

```text
2026-03-02 | L123 | 540 | 18 | 20
```

Different measurements from the same EUID are not considered exact duplicates at this stage.

Output:

```text
data/cleaned/paired_iop_cct_round1_deduplicated.csv
```

### 4. Range Cleaning

Script: `src/clean_ranges.py`

Input: `data/cleaned/paired_iop_cct_round1_deduplicated.csv`

Purpose:

This stage removes measurements outside the project’s predefined validity boundaries.

Current validity ranges:

- CCT: 480–580 µm inclusive
- iCare IOP: > 0 and ≤ 80 mmHg
- Applanation/GAT IOP: > 0 and ≤ 80 mmHg

Examples:

- `479` CCT → invalid
- `480` CCT → valid
- `500` CCT → valid
- `580` CCT → valid
- `581` CCT → invalid

This is a data-validity filter, not a statistical outlier filter. A clinically unusual measurement is not automatically rejected.

A measurement is removed if any of its three values is outside the allowed range.

Example:

```text
iCare = 20
CCT = 600
GAT = 21
```

This row is rejected because CCT is outside the valid range.

Outputs:

- valid observations: `data/cleaned/final_cleaned_dataset.csv`
- rejected observations: `data/cleaned/invalid_ranges.csv`

The rejected file is retained as an audit trail rather than silently deleted.

### 5. One Observation per EUID

Script: `src/remove_repeated_euid.py`

Input: `data/cleaned/final_cleaned_dataset.csv`

Purpose:

This stage is more aggressive than ordinary duplicate removal. It ensures that each EUID appears only once in the dataset, even if the same eye was measured on multiple dates.

Example:

```text
2026-03-02 | L123 | 18 | 540 | 20
2026-03-15 | L123 | 17 | 542 | 19
2026-04-01 | L123 | 19 | 545 | 21
```

becomes:

```text
2026-03-02 | L123 | 18 | 540 | 20
```

The current implementation sorts by date and retains the earliest valid observation for each EUID. This is an intentional methodological decision and should not be confused with exact duplicate removal.

Output:

```text
data/cleaned/final_unique_euid_dataset.csv
```

At this point, every EUID occurs exactly once.

### 6. Train / Validation / Test Split

Script: `src/split_dataset.py`

Input: `data/cleaned/final_unique_euid_dataset.csv`

Important: the split is performed at the patient UID level, not the individual row level.

Examples:

- `L18255471` and `R18255471` are separate rows for model training, but they belong to the same patient.
- They must remain in the same split.

A valid split:

```text
Patient A
├── L123
└── R123
    ↓
  TRAIN

Patient B
├── L456
└── R456
    ↓
  VALIDATION

Patient C
├── L789
└── R789
    ↓
   TEST
```

An invalid split would be:

```text
L123 → TRAIN
R123 → TEST
```

This would leak patient information across train and test sets and can make test performance look artificially good.

Target distribution:

- Training: ~70% of patients
- Validation: ~15% of patients
- Testing: ~15% of patients

The split uses `RANDOM_STATE = 42` to make the procedure reproducible.

Outputs:

- `data/splits/train.csv`
- `data/splits/validation.csv`
- `data/splits/test.csv`

### 7. Why the Split Is Done This Way

The purpose of the test set is to estimate how the model performs on patients it has never seen during training.

If a patient appears in both training and testing sets, the test set is no longer a clean estimate of generalization to unseen patients.

Therefore:

- same patient → same split
- different patients → may be assigned to different splits

Within a split, the two eyes can still be treated as separate observations for model training.

### 8. Final Dataset Structure

After the cleaning pipeline, the intended final structure is:

```text
date | euid | icare_iop | cct | applanation_iop
```

The model will eventually use:

- features: `icare_iop`, `cct`
- target: `applanation_iop`

Identifiers:

- `date`
- `euid`

These are metadata and should not be fed into the model as numerical predictive features.

Age and gender are currently unavailable and are therefore not included in the current model.

---

## Running the Pipeline

Activate the virtual environment first:

```bash
cd ~/projects/icare-ml
source .venv/bin/activate
```

Then run the stages in order:

```bash
python src/round1_cleaning.py
python src/remove_duplicates.py
python src/clean_ranges.py
python src/remove_repeated_euid.py
python src/split_dataset.py
```

Before running the final split, verify that `split_dataset.py` reads:

```text
data/cleaned/final_unique_euid_dataset.csv
```

not the earlier `final_cleaned_dataset.csv`.

---

## When to Rerun Stages

The pipeline is sequential, but you do not need to rerun everything after every change.

### If you change the CCT or IOP ranges

Rerun:

```text
clean_ranges.py
  ↓
remove_repeated_euid.py
  ↓
split_dataset.py
```

You do not need to redo the original pairing.

### If you change the duplicate rule

Rerun:

```text
remove_duplicates.py
  ↓
clean_ranges.py
  ↓
remove_repeated_euid.py
  ↓
split_dataset.py
```

### If you change the 3-hour pairing rule

Changing the pairing window from `≤ 3 hours` to another threshold requires rerunning the entire downstream pipeline:

```text
round1_cleaning.py
  ↓
remove_duplicates.py
  ↓
clean_ranges.py
  ↓
remove_repeated_euid.py
  ↓
split_dataset.py
```

This is necessary because changing the pairing criteria changes the underlying observations.

### If you change only the train/validation/test split

Only rerun:

```bash
python src/split_dataset.py
```

---

## Key Data Decisions

The current pipeline includes the following decisions:

- Eye independence: right and left eyes are treated as separate observations.
- Clinical pairing: only measurements where iCare occurs before Applanation, on the same day, and within 3 hours are accepted.
- Exact duplicates: exact copies are removed.
- Repeated EUIDs: after range cleaning, only the earliest valid observation for each EUID is retained.
- Range filtering: CCT must be between 480 and 580 µm, while iCare and GAT IOP must each be greater than 0 and no more than 80 mmHg.
- Patient-level splitting: the same UID cannot appear in multiple datasets.

---

## Current Dataset Status

The range-cleaning stage previously produced:

- Original rows: 10,446
- Invalid rows removed: 5
- Final rows: 10,441

These results corresponded to the earlier CCT range and should be regenerated after changing the CCT boundary to 480–580 µm.

Do not assume the old row count remains valid after changing the CCT range. After rerunning `clean_ranges.py`, use its new output as the source of truth.

---

## Reproducibility

The project uses a Python virtual environment:

```text
.venv/
```

This keeps the project’s Python packages isolated from the system Python installation.

Installed dependencies are recorded in:

```text
requirements.txt
```

To recreate the environment on another machine:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project was configured around Python 3.12 rather than the system Python 3.14 to provide better compatibility with the ML package ecosystem being used.

---

## Expected ML Workflow

Once the split datasets are finalized, the next phase is model development.

```text
train.csv
  ↓
feature preprocessing
  ↓
model training
  ↓
validation.csv
  ↓
hyperparameter/model selection
  ↓
final model
  ↓
test.csv
  ↓
final unbiased evaluation
```

The test set should not be repeatedly used to tune the model. It should remain isolated until final evaluation.

The eventual evaluation should compare predicted Applanation IOP against the actual Applanation IOP in the test set.

Relevant evaluation measures may include:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R²
- Pearson/Spearman correlation where appropriate
- Bland–Altman analysis
- error distributions
- performance across clinically relevant IOP ranges

The exact evaluation strategy should be finalized before model training rather than chosen after seeing favorable results.

---

## Core Principle

The central rule for this project is:

> Do not let information from the test patients influence model training or model selection.

The data-processing pipeline exists primarily to make that possible while preserving clinically meaningful measurements.

Final sequence:

```text
Raw clinical data
  ↓
Correct ICARE → GAT pairing
  ↓
Exact duplicate removal
  ↓
Clinical validity filtering
  ↓
One observation per EUID
  ↓
Patient-level splitting
  ↓
Machine-learning training
  ↓
Independent evaluation
```
