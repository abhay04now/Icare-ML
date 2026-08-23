import pandas as pd
import re

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/cleaned/final_unique_euid_dataset.csv"
OUTPUT_FILE = "data/serialized_data.csv"

EUID_COLUMN = "euid"


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("SERIALIZATION")
print("=" * 60)

print(f"Input rows: {len(df)}")


# ============================================================
# 2. CHECK REQUIRED COLUMN
# ============================================================

if EUID_COLUMN not in df.columns:
    raise ValueError(
        f"Missing required column: {EUID_COLUMN}"
    )


# ============================================================
# 3. EXTRACT EYE AND ORIGINAL UID
# ============================================================

# Expected EUID format:
#
# L10889346
# R10889346

euid_pattern = r"^([LR])(\d+)$"

extracted = df[EUID_COLUMN].astype(str).str.extract(
    euid_pattern
)

df["eye"] = extracted[0]
df["original_uid"] = extracted[1]


# ============================================================
# 4. VALIDATE EUID FORMAT
# ============================================================

invalid_euid = (
    df["eye"].isna() |
    df["original_uid"].isna()
)

if invalid_euid.any():

    print(
        f"Invalid EUIDs found: {invalid_euid.sum()}"
    )

    print(
        df.loc[
            invalid_euid,
            EUID_COLUMN
        ].tolist()
    )

    raise ValueError(
        "Invalid EUID format detected."
    )


# ============================================================
# 5. CREATE PATIENT SERIAL NUMBERS
# ============================================================

# Sort the original patient UIDs numerically.
#
# Each patient gets ONE serial number.
#
# Example:
#
# UID 10889346 → 00001
# UID 10902594 → 00002
# UID 12680737 → 00003

unique_uids = sorted(
    df["original_uid"].unique(),
    key=int
)

uid_to_serial = {
    uid: index
    for index, uid in enumerate(
        unique_uids,
        start=1
    )
}


# ============================================================
# 6. CREATE SERIALIZED EUID
# ============================================================

df["serial_number"] = (
    df["original_uid"]
    .map(uid_to_serial)
)

df["serialized_euid"] = (
    df["eye"]
    + df["serial_number"].map(
        lambda x: f"{x:05d}"
    )
)


# ============================================================
# 7. VERIFY SERIALIZED IDs
# ============================================================

if df["serialized_euid"].duplicated().any():

    duplicates = df[
        df["serialized_euid"].duplicated(
            keep=False
        )
    ]

    print("\nDuplicate serialized IDs:")
    print(duplicates)

    raise ValueError(
        "Serialization created duplicate EUIDs."
    )


# ============================================================
# 8. REPLACE OLD EUID
# ============================================================

df[EUID_COLUMN] = df["serialized_euid"]


# ============================================================
# 9. REMOVE INTERNAL COLUMNS
# ============================================================

df.drop(
    columns=[
        "eye",
        "original_uid",
        "serial_number",
        "serialized_euid"
    ],
    inplace=True
)


# ============================================================
# 10. SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 11. FINAL CHECK
# ============================================================

print("\n" + "=" * 60)
print("SERIALIZATION COMPLETE")
print("=" * 60)

print(f"Rows: {len(df)}")

print(
    f"Unique serialized IDs: "
    f"{df['euid'].nunique()}"
)

print("\nExample IDs:")

print(
    df["euid"]
    .head(10)
    .to_string(index=False)
)

print("\nFinal columns:")
print(df.columns.tolist())

print(f"\nOutput file: {OUTPUT_FILE}")