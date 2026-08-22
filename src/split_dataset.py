import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/cleaned/final_cleaned_dataset.csv"
)

OUTPUT_DIR = Path(
    "data/splits"
)

TRAIN_FILE = OUTPUT_DIR / "train.csv"
VALIDATION_FILE = OUTPUT_DIR / "validation.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("PATIENT-LEVEL DATASET SPLIT")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nTotal observations: {len(df):,}")


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
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# 3. DERIVE UID FROM EUID
#
# L18255471 → 18255471
# R18255471 → 18255471
# ============================================================

df["euid"] = df["euid"].astype(str).str.strip()

valid_euid = df["euid"].str.match(
    r"^[LR]\d+$"
)

if not valid_euid.all():

    invalid_euids = df.loc[
        ~valid_euid,
        "euid"
    ].unique()

    raise ValueError(
        f"Invalid EUID values found: {invalid_euids[:20]}"
    )

df["uid"] = (
    df["euid"]
    .str[1:]
)


# ============================================================
# 4. COUNT UNIQUE PATIENTS
# ============================================================

unique_uids = df["uid"].unique()

number_of_patients = len(unique_uids)

print(
    f"Unique patients: {number_of_patients:,}"
)


# ============================================================
# 5. SPLIT PATIENT IDs
#
# IMPORTANT:
# We split UIDs, NOT individual observations.
#
# Therefore no patient can appear in more than one dataset.
# ============================================================

train_uids, temp_uids = train_test_split(
    unique_uids,
    test_size=(VALIDATION_SIZE + TEST_SIZE),
    random_state=RANDOM_STATE,
    shuffle=True,
)


# Remaining 30% is split equally:
#
# 15% validation
# 15% test

validation_uids, test_uids = train_test_split(
    temp_uids,
    test_size=0.50,
    random_state=RANDOM_STATE,
    shuffle=True,
)


# ============================================================
# 6. CREATE DATASETS
# ============================================================

train_df = df[
    df["uid"].isin(train_uids)
].copy()

validation_df = df[
    df["uid"].isin(validation_uids)
].copy()

test_df = df[
    df["uid"].isin(test_uids)
].copy()


# ============================================================
# 7. REMOVE UID FROM SAVED DATA
#
# UID is used for splitting and leakage checking.
# It is NOT a model feature.
#
# We retain EUID as the observation identifier.
# ============================================================

train_output = train_df.drop(
    columns=["uid"]
)

validation_output = validation_df.drop(
    columns=["uid"]
)

test_output = test_df.drop(
    columns=["uid"]
)


# ============================================================
# 8. SAVE DATASETS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

train_output.to_csv(
    TRAIN_FILE,
    index=False
)

validation_output.to_csv(
    VALIDATION_FILE,
    index=False
)

test_output.to_csv(
    TEST_FILE,
    index=False
)


# ============================================================
# 9. CHECK PATIENT COUNTS
# ============================================================

print("\n" + "=" * 70)
print("PATIENT SPLIT")
print("=" * 70)

print(
    f"Training patients   : {len(train_uids):,} "
    f"({len(train_uids) / number_of_patients * 100:.1f}%)"
)

print(
    f"Validation patients : {len(validation_uids):,} "
    f"({len(validation_uids) / number_of_patients * 100:.1f}%)"
)

print(
    f"Testing patients    : {len(test_uids):,} "
    f"({len(test_uids) / number_of_patients * 100:.1f}%)"
)


# ============================================================
# 10. CHECK OBSERVATION COUNTS
# ============================================================

print("\n" + "=" * 70)
print("OBSERVATION SPLIT")
print("=" * 70)

print(
    f"Total observations      : {len(df):,}"
)

print(
    f"Training observations   : {len(train_df):,} "
    f"({len(train_df) / len(df) * 100:.1f}%)"
)

print(
    f"Validation observations : {len(validation_df):,} "
    f"({len(validation_df) / len(df) * 100:.1f}%)"
)

print(
    f"Testing observations    : {len(test_df):,} "
    f"({len(test_df) / len(df) * 100:.1f}%)"
)


# ============================================================
# 11. VERIFY NO PATIENT LEAKAGE
# ============================================================

train_set = set(train_uids)
validation_set = set(validation_uids)
test_set = set(test_uids)

train_validation_overlap = (
    train_set & validation_set
)

train_test_overlap = (
    train_set & test_set
)

validation_test_overlap = (
    validation_set & test_set
)

print("\n" + "=" * 70)
print("PATIENT LEAKAGE CHECK")
print("=" * 70)

print(
    "Train ∩ Validation:",
    len(train_validation_overlap)
)

print(
    "Train ∩ Test:",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test:",
    len(validation_test_overlap)
)

if (
    len(train_validation_overlap)
    + len(train_test_overlap)
    + len(validation_test_overlap)
    > 0
):
    raise ValueError(
        "PATIENT LEAKAGE DETECTED!"
    )

print("\nPatient leakage check: PASSED")


# ============================================================
# 12. VERIFY ALL OBSERVATIONS WERE ASSIGNED
# ============================================================

total_split_rows = (
    len(train_df)
    + len(validation_df)
    + len(test_df)
)

print(
    f"\nRows assigned to splits: "
    f"{total_split_rows:,}"
)

if total_split_rows != len(df):
    raise ValueError(
        "Some observations were not assigned to a split."
    )

print("All observations accounted for: PASSED")


# ============================================================
# 13. VERIFY NO DUPLICATE EUID/DATES ACROSS SPLITS
# ============================================================

train_euids = set(train_output["euid"])
validation_euids = set(validation_output["euid"])
test_euids = set(test_output["euid"])

print("\n" + "=" * 70)
print("EUID CHECK")
print("=" * 70)

print(
    "EUIDs shared by Train/Validation:",
    len(train_euids & validation_euids)
)

print(
    "EUIDs shared by Train/Test:",
    len(train_euids & test_euids)
)

print(
    "EUIDs shared by Validation/Test:",
    len(validation_euids & test_euids)
)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print("\nFiles created:")

print(
    f"  Training   : {TRAIN_FILE}"
)

print(
    f"  Validation : {VALIDATION_FILE}"
)

print(
    f"  Testing    : {TEST_FILE}"
)

print("\nFinal columns:")

print(
    train_output.columns.tolist()
)
