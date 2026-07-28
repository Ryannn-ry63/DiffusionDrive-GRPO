# Stage27 Phase3 Formal Generator Result

The fixed two-epoch eight-H100 provisional generator run completed on
2026-07-24 and was independently replayed from both checkpoints and the full
TensorBoard event.

- run:
  `stage27_generator_phase3_formal_seed0/2026.07.24.14.26.13`;
- optimizer steps: 160, exactly 80 per epoch;
- formal audit SHA256:
  `68b903d0c12fc5d99a040c176d4773afcdc009a95cec212c962217714dd7cc5a`;
- checkpoint-freeze SHA256:
  `95bab4782996a25083b21309e459da3826e8c6bfcc14b0d240770c4f61f4b3bd`;
- epoch-1/step-80 SHA256:
  `665e0fda0526e8ffe33d1931f97e48816cf72acd85ba93d4c04dbae90c84d3b3`;
- epoch-2/step-160 SHA256:
  `aa4c287814ec25fdb73fafe797251a45d2ac6fc8aad82bfb8ce39f285f4baae1`.

For both checkpoints:

- 64 permitted decoder tensors changed, 32 per layer;
- zero public-base tensors outside the registered decoder boundary changed;
- frozen reference mismatch count was zero;
- frozen S-multi selector mismatch count was zero;
- all prohibited gradients were exactly zero;
- all 160 logged reward, advantage, log-probability, loss, gradient, and
  optimizer-state observations were finite.

This result authorizes the preregistered fold5 Phase4 evaluation only. It does
not itself establish a PDMS improvement and does not authorize NavTest.
