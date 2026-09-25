# Local Hugging Face export preparation

CORE-LLM-Bench v1.1.0 is the current 9,048-row dataset revision and is published
at https://huggingface.co/datasets/jloe2911/CORE-LLM-Bench/tree/v1.1.0.

## Historical v1.0.0 export workflow

The commands and 9,032-row counts below apply only to the preserved v1.0.0
export workflow. They do not reproduce or upload v1.1.0.

To reproduce the export locally without uploading, generate either release
profile from the repository root:

```console
python scripts/export_huggingface.py --profile full
python scripts/export_huggingface.py --profile public-safe
```

The commands create `release/huggingface/full/` (9,032 rows) and
`release/huggingface/public-safe/` (5,272 rows). Each directory contains one
Parquet row per unique question-hop instance, a profile-specific dataset card,
dataset information, a release manifest, and SHA-256 checksums. They perform no
network requests and upload nothing.

The full profile is the intended canonical v1.0.0 release. Family/FHKB-derived
material in that profile is distributed under the applicable CC BY-SA 3.0
terms documented in `NOTICE.md`. The public-safe profile is an optional reduced
profile that contains Pizza 100, Pizza 250, and OWL2Bench and fails validation
if a Family artifact or record is detected.
