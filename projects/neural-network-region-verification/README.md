# Neural Network Region Verification

Implemented bounded research baseline, v0.1, 2026-10-01.

Maximize a one-hidden-layer scalar ReLU network over an input box with interval-derived big-M constraints. An independent activation-pattern LP enumeration checks small cases. The API returns a feasible witness separately from the solver dual upper bound: counterexample, certified_with_tolerance, or unknown.

The triangular-bump fixture is harmless at both endpoints, but its maximum is 0.5 at x=0.5. Threshold 0.4 is violated; 0.6 is certified within numerical tolerance. Seven local checks passed, including random two-dimensional instances.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest checks -v
python study.py --output local-results.json
```

Core API: network(W,b,v,bias,lower,upper), maximize(net,threshold), enumerate_regions(net). Pass trained weights directly. checks.py is explicitly invoked to avoid changing existing root pytest discovery.

## Boundaries

Single hidden layer and scalar output only. Floating-point solver bounds are not formal interval-arithmetic proofs. An unfinished solve is not a certificate unless its valid upper bound suffices. This module does not train a new proxy, reproduce a paper-scale verifier, or certify arbitrary deep networks.

Research reference: https://research.google/pubs/strong-mixed-integer-programming-formulations-for-trained-neural-networks/

Independent implementation. Existing repository license applies. See SOURCE_REPOSITORY.md and VALIDATION.json.
