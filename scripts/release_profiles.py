"""Single source of truth for CORE-LLM-Bench release profiles."""

from __future__ import annotations

from dataclasses import dataclass


VERSION = "1.0.0"


@dataclass(frozen=True)
class Artifact:
    dataset: str
    hop: str
    zip_name: str
    member: str
    rows: int


@dataclass(frozen=True)
class ReleaseProfile:
    name: str
    datasets: tuple[str, ...]
    dataset_totals: dict[str, int]
    task_totals: dict[str, int]
    overall_total: int
    includes_family: bool


ARTIFACTS = (
    Artifact("Family", "1hop", "FamilyOWL.zip", "FamilyOWL_1hop.json", 1880),
    Artifact("Family", "2hop", "FamilyOWL.zip", "FamilyOWL_2hop.json", 1880),
    Artifact("Pizza100", "1hop", "pizza_100.zip", "pizza_100_1hop.json", 492),
    Artifact("Pizza100", "2hop", "pizza_100.zip", "pizza_100_2hop.json", 492),
    Artifact("Pizza250", "1hop", "pizza_250.zip", "pizza_250_1hop.json", 616),
    Artifact("Pizza250", "2hop", "pizza_250.zip", "pizza_250_2hop.json", 616),
    Artifact("OWL2Bench", "1hop", "OWL2Bench.zip", "OWL2Bench_1hop.json", 1466),
    Artifact("OWL2Bench", "2hop", "OWL2Bench.zip", "OWL2Bench_2hop.json", 1590),
)


PROFILES = {
    "full": ReleaseProfile(
        name="full",
        datasets=("Family", "Pizza100", "Pizza250", "OWL2Bench"),
        dataset_totals={
            "Family": 3760,
            "Pizza100": 984,
            "Pizza250": 1232,
            "OWL2Bench": 3056,
        },
        task_totals={"BQA": 5999, "OEQA": 3033},
        overall_total=9032,
        includes_family=True,
    ),
    "public-safe": ReleaseProfile(
        name="public-safe",
        datasets=("Pizza100", "Pizza250", "OWL2Bench"),
        dataset_totals={"Pizza100": 984, "Pizza250": 1232, "OWL2Bench": 3056},
        task_totals={"BQA": 3455, "OEQA": 1817},
        overall_total=5272,
        includes_family=False,
    ),
}


def get_profile(name: str) -> ReleaseProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown release profile: {name}") from exc


def artifacts_for(profile: ReleaseProfile) -> tuple[Artifact, ...]:
    return tuple(item for item in ARTIFACTS if item.dataset in profile.datasets)


def zip_names_for(profile: ReleaseProfile) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.zip_name for item in artifacts_for(profile)))
