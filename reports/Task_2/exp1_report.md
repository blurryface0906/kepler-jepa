# Experiment 1: frozen encoder spectra

T1 only; no training and no predictor rollout. Five seeds: 42–46. All encoders are initialized directly in float64 and evaluated with dropout disabled.

The unchanged test_id grid supplies 30 distinct radii in [0.502901079, 1.95139527]. All estimates, including undefined/failed cases, remain in the JSON and summary.

## Results

### T_eval = 256

| encoder        | method   | slope mean +/- std          | abs(slope+1.5) mean +/- std   | R2 mean +/- std          | SSE/SST mean +/- std        |   invalid /150 |
|----------------|----------|-----------------------------|-------------------------------|--------------------------|-----------------------------|----------------|
| M0-Linear      | S1       | 0 +/- 0                     | 1.5 +/- 0                     | nan +/- nan              | nan +/- nan                 |              0 |
| M0-Linear      | S2       | -0.00113819756 +/- 1.33e-17 | 1.4988618 +/- 0               | 0.913875685 +/- 2.43e-15 | 0.0861243146 +/- 2.45e-15   |              0 |
| M0-Linear      | S3a_std  | -1.42656547 +/- 2.82e-14    | 0.073434535 +/- 2.82e-14      | 0.999374008 +/- 1.39e-15 | 0.000625992136 +/- 1.42e-15 |              0 |
| M0-Linear      | S3b_std  | -0.221759133 +/- 8.44e-17   | 1.27824087 +/- 0              | 0.748122514 +/- 3.47e-16 | 0.251877486 +/- 3.74e-16    |              0 |
| M0-Linear      | S3a_diff | -1.5 +/- 1.99e-13           | 1.23367982e-13 +/- 1.52e-13   | 1 +/- 0                  | 3.48105647e-25 +/- 1.5e-25  |              0 |
| M0-Linear      | S3b_diff | -1.5 +/- 1.32e-15           | 7.99360578e-16 +/- 1.05e-15   | 1 +/- 0                  | 1.25007682e-29 +/- 2.64e-30 |              0 |
| M0-MLP         | S1       | 0 +/- 0                     | 1.5 +/- 0                     | nan +/- nan              | nan +/- nan                 |              0 |
| M0-MLP         | S2       | -0.0010695918 +/- 7.3e-05   | 1.49893041 +/- 7.3e-05        | 0.931686449 +/- 0.0453   | 0.0683135512 +/- 0.0453     |              0 |
| M0-MLP         | S3a_std  | -1.02806346 +/- 0.0306      | 0.471936538 +/- 0.0306        | 0.960752011 +/- 0.0166   | 0.0392479888 +/- 0.0166     |              0 |
| M0-MLP         | S3b_std  | -0.217487084 +/- 0.00942    | 1.28251292 +/- 0.00942        | 0.800109253 +/- 0.0232   | 0.199890747 +/- 0.0232      |              0 |
| M0-MLP         | S3a_diff | -1.09540006 +/- 0.0885      | 0.404599944 +/- 0.0885        | 0.968224458 +/- 0.0156   | 0.0317755422 +/- 0.0156     |              0 |
| M0-MLP         | S3b_diff | -1.09604501 +/- 0.032       | 0.403954994 +/- 0.032         | 0.964904458 +/- 0.0146   | 0.035095542 +/- 0.0146      |              0 |
| M0-Transformer | S1       | 0 +/- 0                     | 1.5 +/- 0                     | nan +/- nan              | nan +/- nan                 |              0 |
| M0-Transformer | S2       | -0.00113309113 +/- 1.07e-05 | 1.49886691 +/- 1.07e-05       | 0.911757741 +/- 0.00556  | 0.0882422591 +/- 0.00556    |              0 |
| M0-Transformer | S3a_std  | -1.39824505 +/- 0.0398      | 0.101754945 +/- 0.0398        | 0.988956183 +/- 0.0071   | 0.0110438173 +/- 0.0071     |              0 |
| M0-Transformer | S3b_std  | -0.222822291 +/- 0.00547    | 1.27717771 +/- 0.00547        | 0.736697743 +/- 0.0283   | 0.263302257 +/- 0.0283      |              0 |
| M0-Transformer | S3a_diff | -1.47097059 +/- 0.0444      | 0.0486389931 +/- 0.00828      | 0.991057207 +/- 0.0063   | 0.00894279342 +/- 0.0063    |              0 |
| M0-Transformer | S3b_diff | -1.4684551 +/- 0.0362       | 0.0372316082 +/- 0.0287       | 0.975587067 +/- 0.00601  | 0.0244129334 +/- 0.00601    |              0 |

### T_eval = 4096

| encoder        | method   | slope mean +/- std       | abs(slope+1.5) mean +/- std   | R2 mean +/- std          | SSE/SST mean +/- std        |   invalid /150 |
|----------------|----------|--------------------------|-------------------------------|--------------------------|-----------------------------|----------------|
| M0-Linear      | S1       | -1.50408368 +/- 0        | 0.0040836758 +/- 0            | 0.988577386 +/- 0        | 0.0114226144 +/- 0          |              0 |
| M0-Linear      | S2       | -1.49922798 +/- 1.11e-16 | 0.000772020986 +/- 9.93e-17   | 0.999975994 +/- 1.24e-16 | 2.40056176e-05 +/- 4.5e-19  |              0 |
| M0-Linear      | S3a_std  | -1.49999793 +/- 3.33e-15 | 2.06555004e-06 +/- 3.33e-15   | 1 +/- 0                  | 8.61037806e-11 +/- 6.45e-20 |              0 |
| M0-Linear      | S3b_std  | -1.49963118 +/- 2.94e-16 | 0.000368817304 +/- 2.9e-16    | 0.999998656 +/- 0        | 1.34402512e-06 +/- 2.3e-19  |              0 |
| M0-Linear      | S3a_diff | -1.5 +/- 8.32e-14        | 8.97948382e-14 +/- 6.9e-14    | 1 +/- 0                  | 6.51619526e-26 +/- 2.5e-26  |              0 |
| M0-Linear      | S3b_diff | -1.5 +/- 2.22e-16        | 1.77635684e-16 +/- 9.93e-17   | 1 +/- 0                  | 2.80629774e-31 +/- 6.71e-32 |              0 |
| M0-MLP         | S1       | -1.50408368 +/- 0        | 0.0040836758 +/- 0            | 0.988577386 +/- 0        | 0.0114226144 +/- 0          |              0 |
| M0-MLP         | S2       | -1.49865812 +/- 0.000309 | 0.00134187527 +/- 0.000309    | 0.999974429 +/- 3.03e-06 | 2.55710286e-05 +/- 3.03e-06 |              0 |
| M0-MLP         | S3a_std  | -1.50998801 +/- 0.0489   | 0.0397196713 +/- 0.0233       | 0.999756331 +/- 0.000146 | 0.000243668904 +/- 0.000146 |              0 |
| M0-MLP         | S3b_std  | -1.49939868 +/- 0.000514 | 0.000601315604 +/- 0.000514   | 0.999993393 +/- 2.66e-06 | 6.60660814e-06 +/- 2.66e-06 |              0 |
| M0-MLP         | S3a_diff | -1.53550618 +/- 0.181    | 0.149111397 +/- 0.0807        | 0.998655954 +/- 0.000933 | 0.00134404598 +/- 0.000933  |              0 |
| M0-MLP         | S3b_diff | -1.49972444 +/- 0.000306 | 0.000277423637 +/- 0.000303   | 0.999987885 +/- 1.91e-05 | 1.21149246e-05 +/- 1.91e-05 |              0 |
| M0-Transformer | S1       | -1.50408368 +/- 0        | 0.0040836758 +/- 0            | 0.988577386 +/- 0        | 0.0114226144 +/- 0          |              0 |
| M0-Transformer | S2       | -1.49919449 +/- 0.000164 | 0.000805511196 +/- 0.000164   | 0.999976282 +/- 2.17e-06 | 2.37176124e-05 +/- 2.17e-06 |              0 |
| M0-Transformer | S3a_std  | -1.49941675 +/- 0.00162  | 0.0014162833 +/- 0.00073      | 0.999996803 +/- 2.13e-06 | 3.19679082e-06 +/- 2.13e-06 |              0 |
| M0-Transformer | S3b_std  | -1.49953075 +/- 0.000546 | 0.00061852271 +/- 0.000308    | 0.999998173 +/- 7.31e-07 | 1.82738386e-06 +/- 7.31e-07 |              0 |
| M0-Transformer | S3a_diff | -1.49818344 +/- 0.00533  | 0.00434697327 +/- 0.00298     | 0.999982219 +/- 1.28e-05 | 1.77805836e-05 +/- 1.28e-05 |              0 |
| M0-Transformer | S3b_diff | -1.49976903 +/- 0.000429 | 0.000426099818 +/- 0.000155   | 0.999998802 +/- 1.89e-07 | 1.19814108e-06 +/- 1.89e-07 |              0 |

## Acceptance

| criterion                                                              | status   |
|------------------------------------------------------------------------|----------|
| H1 Linear S3a_diff, T=256                                              | PASS     |
| H1 Linear S3b_diff, T=256                                              | PASS     |
| H1 Linear S3a_diff, T=4096                                             | PASS     |
| H1 Linear S3b_diff, T=4096                                             | PASS     |
| H1 M0-MLP, T=4096, one phase method passes all seeds                   | FAIL     |
| H1 M0-Transformer, T=4096, one phase method passes all seeds           | PASS     |
| H2 Linear long FFT quantization and >=15-decade continuous improvement | PASS     |

Overall: NOT MET

Thresholds are checked on every seed, separately from the mean ± std summaries. For each nonlinear encoder, at least one prescribed phase method must pass on all five seeds; an estimator is not chosen separately for each orbit or seed.

### Nonlinear phase estimates by seed, T_eval=4096

| encoder        |   seed | method   |   abs(slope+1.5) |        SSE/SST | status   |
|----------------|--------|----------|------------------|----------------|----------|
| M0-MLP         |     42 | S3b_std  |   8.49904258e-05 | 1.09602781e-05 | FAIL     |
| M0-MLP         |     42 | S3b_diff |   3.29833278e-05 | 4.61851033e-05 | FAIL     |
| M0-Transformer |     42 | S3b_std  |   0.000293636766 | 8.12893708e-07 | PASS     |
| M0-Transformer |     42 | S3b_diff |   0.000211800905 | 1.44972415e-06 | PASS     |
| M0-MLP         |     43 | S3b_std  |   0.000167236095 | 4.65491718e-06 | PASS     |
| M0-MLP         |     43 | S3b_diff |   0.000745015815 | 5.67726991e-06 | PASS     |
| M0-Transformer |     43 | S3b_std  |   0.000373171869 | 2.65983796e-06 | PASS     |
| M0-Transformer |     43 | S3b_diff |   0.000519900284 | 9.44721277e-07 | PASS     |
| M0-MLP         |     44 | S3b_std  |   0.000778401187 | 6.90839445e-06 | PASS     |
| M0-MLP         |     44 | S3b_diff |   4.66836246e-06 | 3.97944736e-06 | PASS     |
| M0-Transformer |     44 | S3b_std  |   0.000554077337 | 2.39981719e-06 | PASS     |
| M0-Transformer |     44 | S3b_diff |   0.000588345191 | 1.20434432e-06 | PASS     |
| M0-MLP         |     45 | S3b_std  |   0.00135699596  | 6.1762407e-06  | PASS     |
| M0-MLP         |     45 | S3b_diff |   0.000217858117 | 2.35581357e-06 | PASS     |
| M0-Transformer |     45 | S3b_std  |   0.00086934596  | 1.5546134e-06  | PASS     |
| M0-Transformer |     45 | S3b_diff |   0.000322617416 | 1.28510539e-06 | PASS     |
| M0-MLP         |     46 | S3b_std  |   0.000618954348 | 4.33321032e-06 | PASS     |
| M0-MLP         |     46 | S3b_diff |   0.000386592563 | 2.37698862e-06 | PASS     |
| M0-Transformer |     46 | S3b_std  |   0.00100238162  | 1.70975702e-06 | PASS     |
| M0-Transformer |     46 | S3b_diff |   0.000487835293 | 1.10681027e-06 | PASS     |

## Interpretation limits

The isometric linear encoder preserves a harmonic circle in a two-dimensional subspace. First differences remove additive bias/finite-window centroid shifts. This establishes a controlled encoder-only baseline and its estimator precision; it does not establish learned gravity.

A nonlinear deterministic map preserves periodicity but need not preserve the fundamental spectral peak or a circular PCA phase. The nonlinear thresholds above are empirical tests, not identities.

A successful H2 control demonstrates that a continuous estimator can produce at least eight nines without training in this setup. It does not identify the procedure used by JEPA-Anything for its published number without reproduction of that original pipeline. FFT limits depend on the record length, sampling and estimators; they are not universal limits on discovering a law.

R² is a regression score and can conceal frequency bias. Per-orbit estimates and true frequencies are provided for separate inspection. Constant FFT outputs have undefined R² (JSON null), and no failed orbit is excluded from a regression.

The plot uses the 4096-sample regime. Panel (a) shows mean log frequencies and ±1 sample standard deviation across seeds. Panel (b) shows side-by-side horizontal histograms of log residual ratio for each encoder/method; widths represent 1–5 seeds, so the 10⁻⁸ threshold is a horizontal line. No finite residual is floored to meet an accuracy criterion.

## Reproduction

```sh
python -m experiments.exp1_encoder_tautology
```

Artifacts: [full stdout](exp1_stdout.txt), [detailed JSON](exp1_results.json), [figure](figures/exp1_encoder_tautology.png). Source SHA-256 hashes, parameter fingerprints, environment, numerical settings, timings, failed-estimate messages and criteria are stored in JSON.
