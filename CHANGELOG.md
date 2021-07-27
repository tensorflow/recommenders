# Changelog

## [unreleased][unreleased]

### Breaking changes

-   `TopK` layer indexing API changed. Indexing with datasets is now done via
    the `index_from_dataset` method. This change reduces the possibility of
    misaligning embeddings and candidate identifiers when indexing via
    indeterministic datasets.

## [0.5.2][2021-07-15]

### Fixed

-   Fixed error in default arguments to `tfrs.experimental.models.Ranking`
    (https://github.com/tensorflow/recommenders/issues/311).
-   Fix TPUEmbedding layer to use named parameters.

### Added

-   Added `batch_metrics` to `tfrs.tasks.Retrieval` for measuring how good the
    model is at picking out the true candidate for a query from other candidates
    in the batch.
-   Added `tfrs.experimental.layers.embedding.PartialTPUEmbedding` layer, which
    uses `tfrs.layers.embedding.TPUEmbedding` for large embedding lookups and
    `tf.keras.layers.Embedding` for smaller embedding lookups.

## [0.5.1][2021-05-14]

### Changed

-   Supplying incompatibly-shaped candidates and identifiers inputs to
    `factorized_top_k` layers will now raise (to prevent issues similar to
    https://github.com/tensorflow/recommenders/issues/286).

## [0.5.0][2021-05-06]

### Changed

-   Fixed the bug in `tfrs.layers.loss.SamplingProbablityCorrection` that logits
    should subtract the log of item probability.
-   `tfrs.experimental.optimizers.CompositeOptimizer`: an optimizer that
    composes multiple individual optimizers which can be applied to different
    subsets of the model's variables.
-   `tfrs.layers.dcn.Cross` and `DotInteraction` layers have been moved to
    `tfrs.layers.feature_interaction` package.

### Added

-   `tfrs.experimental.models.Ranking`, an experimental pre-built model for
    ranking tasks. Can be used as DLRM like model with Dot Product feature
    interaction or DCN like model with Cross layer.

## [0.4.0][2021-01-20]

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
