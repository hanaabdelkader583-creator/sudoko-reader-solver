| Misreads | Method | Recovery % | Solved % | Time s |
|---|---|---|---|---|
| 0 | NN-guided (A*) | 100.0 ± 0.0 | 100.0 ± 0.0 | 0.0037 ± 0.0017 |
| 0 | No correction | 100.0 ± 0.0 | 100.0 ± 0.0 | 0.0012 ± 0.0003 |
| 0 | Random correction | 100.0 ± 0.0 | 100.0 ± 0.0 | 0.0014 ± 0.0005 |
| 1 | NN-guided (A*) | 100.0 ± 0.0 | 100.0 ± 0.0 | 0.0074 ± 0.0009 |
| 1 | No correction | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0001 ± 0.0000 |
| 1 | Random correction | 30.0 ± 27.4 | 40.0 ± 22.4 | 0.0023 ± 0.0007 |
| 2 | NN-guided (A*) | 30.0 ± 27.4 | 30.0 ± 27.4 | 0.0077 ± 0.0013 |
| 2 | No correction | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0001 ± 0.0001 |
| 2 | Random correction | 0.0 ± 0.0 | 20.0 ± 27.4 | 0.0018 ± 0.0010 |
| 3 | NN-guided (A*) | 20.0 ± 27.4 | 30.0 ± 27.4 | 0.0121 ± 0.0012 |
| 3 | No correction | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0001 ± 0.0000 |
| 3 | Random correction | 0.0 ± 0.0 | 20.0 ± 27.4 | 0.0055 ± 0.0054 |
| 4 | NN-guided (A*) | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.1039 ± 0.2017 |
| 4 | No correction | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0001 ± 0.0001 |
| 4 | Random correction | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0150 ± 0.0122 |

Note: same instances as Experiment 1 (30 givens, 0 skipped, 5 seed groups, std across groups). Random correction: up to 200 attempts of 1-3 random changes to conflicting cells; the first conflict-free grid goes to the CSP. 'Solved' = some valid solution; 'Recovery' = the true solution.