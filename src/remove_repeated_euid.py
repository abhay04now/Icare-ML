import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/cleaned/final_cleaned_dataset.csv"
)

OUTPUT_FILE = Path(
    "data/cleaned/final_unique_euid_dataset.csv"
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("EUID DEDUPLICATION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nOriginal observations: {len(df):,}")


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "date",
    "euid",
    "icare_iop",
    "cct",
    "applanation_iop",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 3. CONVERT DATE
# ============================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# ============================================================
# 4. CHECK FOR NULL EUIDs
# ============================================================

if df["euid"].isna().any():
    raise ValueError(
        "NULL EUID values found."
    )


# ============================================================
# 5. SORT BY DATE
#
# Earliest measurement for each EUID will be retained.
# ============================================================

df = df.sort_values(
    by=["euid", "date"]
).copy()


# ============================================================
# 6. IDENTIFY REPEATED EUIDs
# ============================================================

duplicate_mask = df.duplicated(
    subset=["euid"],
    keep=False
)

repeated = df.loc[
    duplicate_mask
].copy()

print(
    f"Unique EUIDs before removal: "
    f"{df['euid'].nunique():,}"
)

print(
    f"Observations belonging to repeated EUIDs: "
    f"{len(repeated):,}"
)

print(
    f"Number of EUIDs appearing more than once: "
    f"{repeated['euid'].nunique():,}"
)


# ============================================================
# 7. KEEP ONLY ONE ROW PER EUID
#
# Because we sorted by date, this keeps the earliest
# observation for each eye.
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=["euid"],
    keep="first"
).copy()

removed = before - len(df)


# ============================================================
# 8. RESTORE CHRONOLOGICAL ORDER
# ============================================================

df = df.sort_values(
    by=["date", "euid"]
).copy()


# ============================================================
# 9. CONVERT DATE BACK TO YYYY-MM-DD
# ============================================================

df["date"] = df["date"].dt.strftime(
    "%Y-%m-%d"
)


# ============================================================
# 10. VERIFY
# ============================================================

remaining_duplicates = df.duplicated(
    subset=["euid"]
).sum()

if remaining_duplicates != 0:
    raise ValueError(
        "Repeated EUIDs still remain."
    )

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"Observations before : {before:,}"
)

print(
    f"Observations removed: {removed:,}"
)

print(
    f"Observations after  : {len(df):,}"
)

print(
    f"Unique EUIDs        : {df['euid'].nunique():,}"
)

print(
    f"Remaining EUID duplicates: "
    f"{remaining_duplicates}"
)


# ============================================================
# 11. SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)

print("\nFinal columns:")
print(df.columns.tolist())
