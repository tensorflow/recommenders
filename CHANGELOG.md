# Changelog

## [0.2.0][2020-10-15]

### Added

-   `tfrs.tasks.Ranking.call` now accepts a `compute_metrics` argument to allow
    switching off metric computation.
-   `tfrs.tasks.Ranking` now accepts label and prediction metrics.
-   Add metrics setter/getters on `tfrs.tasks.Retrieval`.

### Breaking changes

-   Corpus retrieval metrics and layers have been reworked.

    `tfrs.layers.corpus.DatasetTopk` has been removed,
    `tfrs.layers.corpus.DatasetIndexedTopK` renamed to
    `tfrs.layers.factorized_top_k.Streaming`, `tfrs.layers.ann.BruteForce`
    renamed to `tfrs.layers.factorized_top_k.BruteForce`. All top-k retrieval
    layers (`BruteForce`, `Streaming`) now follow a common interface.

### Changed

-   `Dataset` parallelism enabled by default in `DatasetTopK` and
    `DatasetIndexedTopK` layers, bringing over 2x speed-ups to evaluations
    workloads.
-   `evaluate_metrics` argument to `tfrs.tasks.Retrieval.call` renamed to
    `compute_metrics`.
