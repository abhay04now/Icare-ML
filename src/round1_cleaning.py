import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/raw/icare_cct_dataset.xlsx")
OUTPUT_FILE = Path("data/cleaned/icare_cct_dataset_round1_cleaned.csv")

MAX_TIME_GAP_HOURS = 3
MAX_TIME_GAP = pd.Timedelta(hours=MAX_TIME_GAP_HOURS)


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

print("=" * 70)
print("ROUND 1 DATA CLEANING")
print("=" * 70)

print(f"\nLoading: {INPUT_FILE}")

df = pd.read_excel(INPUT_FILE)

print(f"Raw rows: {len(df):,}")
print(f"Raw columns: {len(df.columns)}")


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "uid",
    "mr_no",
    "LE_method",
    "RE_method",
    "LE_value",
    "RE_value",
    "CCT_RE_value",
    "CCT_LE_value",
    "iop_recorded_at",
    "cct_recorded_at",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 3. REMOVE REDUNDANT mr_no
# ============================================================

df = df.drop(columns=["mr_no"])

print("\nRemoved column: mr_no")


# ============================================================
# 4. PARSE TIMESTAMPS
# ============================================================

print("\nParsing timestamps...")

df["iop_recorded_at"] = pd.to_datetime(
    df["iop_recorded_at"],
    errors="coerce"
)

df["cct_recorded_at"] = pd.to_datetime(
    df["cct_recorded_at"],
    errors="coerce"
)


# ============================================================
# 5. REMOVE INVALID NULL / ZERO VALUES
#
# We only require fields necessary to construct a valid
# eye-level IOP/CCT observation.
# ============================================================

measurement_columns = [
    "uid",
    "LE_method",
    "RE_method",
    "LE_value",
    "RE_value",
    "CCT_RE_value",
    "CCT_LE_value",
    "iop_recorded_at",
    "cct_recorded_at",
]

before_invalid_filter = len(df)

# Remove rows where any required field is NULL.
df = df.dropna(subset=measurement_columns)

# Remove rows containing zero in measurement fields.
numeric_measurement_columns = [
    "LE_value",
    "RE_value",
    "CCT_RE_value",
    "CCT_LE_value",
]

zero_mask = (df[numeric_measurement_columns] == 0).any(axis=1)

df = df.loc[~zero_mask].copy()

after_invalid_filter = len(df)

print(
    f"Rows removed due to NULL/zero values: "
    f"{before_invalid_filter - after_invalid_filter:,}"
)

print(f"Rows remaining: {after_invalid_filter:,}")


# ============================================================
# 6. CONVERT RAW DATA INTO EYE-LEVEL RECORDS
#
# Each eye becomes an independent observation.
#
# RE:
#   RE_method
#   RE_value
#   CCT_RE_value
#
# LE:
#   LE_method
#   LE_value
#   CCT_LE_value
# ============================================================

right_eye = df[
    [
        "uid",
        "RE_method",
        "RE_value",
        "CCT_RE_value",
        "iop_recorded_at",
        "cct_recorded_at",
    ]
].copy()

right_eye = right_eye.rename(
    columns={
        "RE_method": "method",
        "RE_value": "iop_value",
        "CCT_RE_value": "cct",
    }
)

right_eye["eye"] = "RE"


left_eye = df[
    [
        "uid",
        "LE_method",
        "LE_value",
        "CCT_LE_value",
        "iop_recorded_at",
        "cct_recorded_at",
    ]
].copy()

left_eye = left_eye.rename(
    columns={
        "LE_method": "method",
        "LE_value": "iop_value",
        "CCT_LE_value": "cct",
    }
)

left_eye["eye"] = "LE"


eye_df = pd.concat(
    [right_eye, left_eye],
    ignore_index=True
)


# Reorder columns for readability.

eye_df = eye_df[
    [
        "uid",
        "eye",
        "method",
        "iop_value",
        "cct",
        "iop_recorded_at",
        "cct_recorded_at",
    ]
]


print("\nEye-level records created:")
print(f"RE records: {(eye_df['eye'] == 'RE').sum():,}")
print(f"LE records: {(eye_df['eye'] == 'LE').sum():,}")
print(f"Total eye records: {len(eye_df):,}")


# ============================================================
# 7. SEPARATE ICARE AND APPLANATION MEASUREMENTS
# ============================================================

icare = eye_df[
    eye_df["method"].str.upper().eq("ICARE")
].copy()

applanation = eye_df[
    eye_df["method"].str.upper().eq("APPLANATION")
].copy()

print("\nMethod records:")
print(f"ICARE: {len(icare):,}")
print(f"Applanation: {len(applanation):,}")


# ============================================================
# 8. PREPARE TIMESTAMP COLUMNS
# ============================================================

icare = icare.rename(
    columns={
        "iop_value": "icare_iop",
        "cct": "icare_cct",
        "iop_recorded_at": "icare_time",
        "cct_recorded_at": "icare_cct_time",
    }
)

applanation = applanation.rename(
    columns={
        "iop_value": "applanation_iop",
        "cct": "applanation_cct",
        "iop_recorded_at": "applanation_time",
        "cct_recorded_at": "applanation_cct_time",
    }
)


# ============================================================
# 9. CREATE CALENDAR-DATE COLUMNS
#
# The doctor's rule requires the measurements to occur
# on the SAME DAY.
# ============================================================

icare["date"] = icare["icare_time"].dt.date
applanation["date"] = applanation["applanation_time"].dt.date


# ============================================================
# 10. MATCH ICARE → APPLANATION
#
# Conditions:
#
#   1. Same patient (uid)
#   2. Same eye
#   3. Same calendar date
#   4. ICARE occurs BEFORE Applanation
#   5. Difference <= 3 hours
#
# For multiple valid ICARE measurements, the closest
# preceding ICARE measurement is selected.
# ============================================================

print("\nMatching ICARE → Applanation measurements...")

icare = icare.sort_values(
    ["uid", "eye", "date", "icare_time"]
)

applanation = applanation.sort_values(
    ["uid", "eye", "date", "applanation_time"]
)


# merge_asof finds the nearest preceding ICARE measurement.
#
# Because we first match by uid, eye and date, matches cannot
# cross patients, eyes, or calendar days.

paired = pd.merge_asof(
    applanation.sort_values("applanation_time"),
    icare.sort_values("icare_time"),
    left_on="applanation_time",
    right_on="icare_time",
    by=["uid", "eye", "date"],
    direction="backward",
    tolerance=MAX_TIME_GAP,
    suffixes=("_applanation", "_icare"),
)


# ============================================================
# 11. REMOVE UNMATCHED APPLANATION MEASUREMENTS
# ============================================================

before_pair_filter = len(paired)

paired = paired[
    paired["icare_time"].notna()
].copy()

after_pair_filter = len(paired)

print(
    f"Applanation measurements without a valid "
    f"preceding ICARE match: "
    f"{before_pair_filter - after_pair_filter:,}"
)


# ============================================================
# 12. EXPLICITLY VERIFY THE DOCTOR'S RULE
#
# These checks should all pass for every retained row.
# ============================================================

paired["time_difference"] = (
    paired["applanation_time"]
    - paired["icare_time"]
)

paired["icare_date"] = paired["icare_time"].dt.date
paired["applanation_date"] = paired["applanation_time"].dt.date


same_day = (
    paired["icare_date"]
    == paired["applanation_date"]
)

icare_first = (
    paired["icare_time"]
    < paired["applanation_time"]
)

within_three_hours = (
    paired["time_difference"]
    <= MAX_TIME_GAP
)

valid_pair = (
    same_day
    & icare_first
    & within_three_hours
)

invalid_pairs = (~valid_pair).sum()

if invalid_pairs > 0:
    print(
        f"WARNING: {invalid_pairs} invalid pairs found "
        f"during verification."
    )

    paired = paired.loc[valid_pair].copy()

else:
    print("\nAll retained pairs satisfy the 3-hour rule.")


# ============================================================
# 13. USE CCT ASSOCIATED WITH THE IOP MEASUREMENT
#
# We retain the CCT from the ICARE-side record.
#
# Since the final dataset requires one CCT per paired
# observation, this is the CCT associated with the ICARE
# measurement used in the pair.
# ============================================================

paired["cct"] = paired["icare_cct"]


# ============================================================
# 14. CONSTRUCT FINAL DATASET
#
# Target variable is deliberately LAST:
#
# uid | date | eye | icare_iop | cct | applanation_iop
# ============================================================

final_df = paired[
    [
        "uid",
        "date",
        "eye",
        "icare_iop",
        "cct",
        "applanation_iop",
    ]
].copy()


# ============================================================
# 15. FINAL NULL / ZERO CHECK
# ============================================================

final_measurement_columns = [
    "icare_iop",
    "cct",
    "applanation_iop",
]

null_count = final_df[
    final_measurement_columns
].isna().sum().sum()

zero_count = (
    final_df[final_measurement_columns] == 0
).sum().sum()

if null_count > 0:
    raise ValueError(
        f"Final dataset contains {null_count} NULL values."
    )

if zero_count > 0:
    raise ValueError(
        f"Final dataset contains {zero_count} zero values."
    )


# ============================================================
# 16. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(f"Final paired observations: {len(final_df):,}")
print(f"Unique patients: {final_df['uid'].nunique():,}")

print("\nEye distribution:")
print(final_df["eye"].value_counts())

print("\nMissing values:")
print(final_df.isna().sum())

print("\nBasic statistics:")
print(
    final_df[
        [
            "icare_iop",
            "cct",
            "applanation_iop",
        ]
    ].describe()
)


# ============================================================
# 17. SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print(f"Output saved to: {OUTPUT_FILE}")

print("\nFinal columns:")
print(final_df.columns.tolist())

print("\nFirst 10 final observations:")
print(final_df.head(10).to_string(index=False))