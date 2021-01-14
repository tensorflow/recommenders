# Changelog

## [Unreleased][unreleased]

### Added

-   `TopK` layers now come with a `query_with_exclusions` method, allowing
    certain candidates to be excluded from top-k retrieval.
-   `TPUEmbedding` Keras layer for accelerating embedding lookups for large
    tables with TPU.

### Changed

-   `factorized_top_k.Streaming` layer now accepts a query model, like other
    `factorized_top_k` layers.

-   Updated ScaNN to 1.2.0, which requires TensorFlow 2.4.x. When not using
    ScaNN, any TF >= 2.3 is still supported.

## [0.3.2][2020-12-22]

### Changed

-   Pinned TensorFlow to >= 2.3 when ScaNN is not being installed. When ScaNN is
    being installed, we pin on >= 2.3, < 2.4. This allows users to use TFRS on
    TF 2.4 when they are not using ScaNN.

## [0.3.1][2020-12-22]

### Changed

-   Pinned TensorFlow to 2.3.x and ScaNN to 1.1.1 to ensure TF and ScaNN
    versions are in lockstep.

## [0.3.0][2020-11-18]

### Added

-   Deep cross networks: efficient ways of learning feature interactions.
-   ScaNN integration: efficient approximate maximum inner product search for
    fast retrieval.

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
