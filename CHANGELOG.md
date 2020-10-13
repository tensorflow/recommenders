# Changelog

## [unreleased][unreleased]

### Added

-   `tfrs.tasks.Ranking.call` now accepts a `compute_metrics` argument to allow
    switching off metric computation.
-   `tfrs.tasks.Ranking` now accepts label and prediction metrics.

### Changed

-   `Dataset` parallelism enabled by default in `DatasetTopK` and
    `DatasetIndexedTopK` layers, bringing over 2x speed-ups to evaluations
    workloads.
-   `evaluate_metrics` argument to `tfrs.tasks.Retrieval.call` renamed to
    `compute_metrics`.
