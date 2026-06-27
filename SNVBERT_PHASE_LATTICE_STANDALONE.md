# SNVBERT PhaseLattice standalone source

This source surface implements a phased-genotype reconstruction model without depending on another project package or compatibility wrapper.

## Architecture

- paired haplotype input with a shared locus mask;
- neutral-mixed masked genotype states, allele-contrast embeddings, learned three-scale
  coordinates, chromosome embeddings and window-span context;
- four repeated `state → state → attention` stages;
- eight bidirectional state layers;
- four per-phase physical-position attention layers;
- mean/difference phase exchange after each attention layer;
- one local depthwise Conv-SwiGLU inside every layer;
- latent cross-attention/refinement reconstruction followed by ordinal-RBF link diffusion;
- one shared two-class allele head from which phased four-state probabilities and dosage are derived.

The default release configuration is H0: no directional geometry modulation and no distance-bias table.

## Training objective

- masked haploid cross-entropy, weight `1.0`;
- diploid cross-entropy derived from paired allele probabilities, weight `0.5`;
- window-level allele-prior alignment, weight `0.02`;
- token-level allele-prior alignment, weight `0.02`;
- global target-count normalization across devices and accumulation microbatches.

## Source boundary

The standalone distribution contains only:

- `src/snvbert_mamba/`;
- `src/train_snvbert_mamba.py`;
- `configs/snvbert_mamba/`;
- the dedicated launcher and tests;
- this document and a per-file checksum manifest.

Research compatibility code, historical benchmarks, archived references, checkpoints and datasets are excluded. This code is a release candidate until its own from-scratch training and validation receipts complete.

## External comparison boundary

`benchmark_port.py` reserves a validation-only `ExternalBenchmarkPort`. Independent comparison
implementations may be connected outside this distribution through `BenchmarkRequest` and
`BenchmarkResponse`. The standalone release deliberately bundles zero comparator implementations,
weights or comparator-specific configuration.
