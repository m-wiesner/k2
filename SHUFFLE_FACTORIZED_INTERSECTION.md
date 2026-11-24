# Factorized dense intersection for Shuffle CTC

This branch contains the k2 companion changes used by the MTASR Shuffle CTC
recipe. It adds `intersect_dense_factorized`, which consumes separate token and
speaker log-score tensors instead of materializing the token-by-speaker output
space.

For a token vocabulary of size `V` (including blank), a nonblank joint graph
label is encoded as

```text
label = token + speaker * (V - 1)
```

and label 0 remains blank. The operation returns separate arc maps for token
and speaker scores so autograd can propagate to both factors.

The corresponding recipe, build instructions, reference implementation, and
tests live on the icefall branch `paper/longform-multitalker-alignment`, under
`egs/librispeech/MTASR/shuffle_ctc/`. The standalone extension in that directory
is the preferred reproduction path; this k2 branch preserves the integrated
implementation used during development.

`K2_CUDA_ARCHS` may be set at configure time to restrict compilation to the
CUDA architectures available on the target system.
