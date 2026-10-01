# Self-Supervised Primal-Dual Proxy

Implemented bounded research baseline, v0.1, 2026-10-01.

Solve the parametric convex QP min sum(0.5*a*x^2+b*x), x>=0, sum(x)=d, with a>0 and d>0. A primal network returns d*softmax(logits); a dual network predicts an unrestricted equality multiplier nu. The valid dual objective is -nu*d-0.5*sum(relu(-(b+nu))^2/a). Label-free training minimizes f(x)-g(nu).

A matched supervised primal comparator uses exact labels. Both methods train the dual head without dual labels. The independent KKT bisection oracle is prohibited during self-supervised fitting by a test, then used to audit actual gaps and certificates at evaluation. Primal and dual training terms are separable in this construction.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest checks -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python study.py --output local-results.json
```

Core API: train(Z,mode), oracle(Z), objectives(Z,x,nu), evaluate(model,Z). Seven checks passed. Three training seeds and disjoint nominal/cost-demand-shift tests are provided. Source hashes and local environment are recorded in VALIDATION.json.

## Boundaries

Simplex QPs only: no upper bounds, general conic constraints, or integer variables. This is not DLL/E2ELR reproduction or a claim of industrial speedup. Numerical feasibility and weak-duality checks do not certify worst-case behavior over a parameter region. The short benchmark is not an algorithm ranking.

Research reference: https://arxiv.org/abs/2402.03086

Independent implementation. Existing repository license applies. checks.py does not participate in existing root pytest discovery.
