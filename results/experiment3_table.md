| Misreads | Method | Recovery % | Solved % | Time s |
|---|---|---|---|---|
| 0 | NN-guided (A*) | 100.0 ± 0.0 | 100.0 ± 0.0 | 0.0038 ± 0.0001 |
| 0 | No correction | 100.0 ± 0.0 | 100.0 ± 0.0 | 0.0012 ± 0.0001 |
| 0 | Random correction | 100.0 ± 0.0 | 100.0 ± 0.0 | 0.0012 ± 0.0001 |
| 1 | NN-guided (A*) | 85.0 ± 7.1 | 100.0 ± 0.0 | 0.0065 ± 0.0007 |
| 1 | No correction | 0.0 ± 0.0 | 13.0 ± 5.7 | 0.0003 ± 0.0001 |
| 1 | Random correction | 13.0 ± 12.0 | 43.0 ± 9.1 | 0.0016 ± 0.0004 |
| 2 | NN-guided (A*) | 69.0 ± 9.6 | 100.0 ± 0.0 | 0.0110 ± 0.0014 |
| 2 | No correction | 0.0 ± 0.0 | 4.0 ± 4.2 | 0.0002 ± 0.0000 |
| 2 | Random correction | 1.0 ± 2.2 | 22.0 ± 10.4 | 0.0025 ± 0.0006 |
| 3 | NN-guided (A*) | 59.0 ± 8.9 | 100.0 ± 0.0 | 0.0150 ± 0.0019 |
| 3 | No correction | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0001 ± 0.0000 |
| 3 | Random correction | 0.0 ± 0.0 | 16.0 ± 11.4 | 0.0071 ± 0.0027 |
| 4 | NN-guided (A*) | 0.0 ± 0.0 | 60.0 ± 6.1 | 0.0425 ± 0.0057 |
| 4 | No correction | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0001 ± 0.0000 |
| 4 | Random correction | 0.0 ± 0.0 | 11.0 ± 6.5 | 0.0173 ± 0.0063 |

Note: same instances as Experiment 1 (30 givens, 0 skipped, 5 seed groups, std across groups). Random correction: up to 200 attempts of 1-3 random changes to conflicting cells; the first conflict-free grid goes to the CSP. 'Solved' = some valid solution; 'Recovery' = the true solution.