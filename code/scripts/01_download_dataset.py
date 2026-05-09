"""Download SWE-bench Verified and verify the test split schema."""

from datasets import load_dataset

DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
REQUIRED_FIELDS = (
    "instance_id",
    "problem_statement",
    "patch",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
)


def main() -> None:
    ds = load_dataset(DATASET_NAME)
    print({k: len(v) for k, v in ds.items()})

    if "test" not in ds:
        raise RuntimeError(f"{DATASET_NAME} is expected to contain a test split.")

    for split in ["test"]:
        for ex in ds[split]:
            for field in REQUIRED_FIELDS:
                assert field in ex, f"{split}:{ex.get('instance_id')} missing {field}"
        print(f"{split}: {len(ds[split])} instances OK")


if __name__ == "__main__":
    main()
