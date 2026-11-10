# P7: Handwritten Sudoku Reader and Solver 🧩

An end-to-end pipeline combining a Neural Network (MNIST digit recognition), Heuristic Correction Search ($A^*$, UCS, Greedy), and CSP Constraint Solving (Backtracking, Forward Checking, AC-3) to read and solve handwritten Sudoku puzzles.

---

## 🛠️ Environment Setup & Installation

Make sure you have Python installed. Clone the repository and install the dependencies:

```bash
git clone <https://github.com/hanaabdelkader583-creator/sudoko-reader-solver.git>
cd SODOKU
pip install -r requirements.txt


1. Run the Integrated Live DemoTo view the full pipeline demo 
python run_demo.py 3 1

2. Run Experiments
Experiment 1: Search Algorithm Comparison (UCS vs Greedy vs A*)
python run_experiment1.py quick
python run_experiment1.py

Experiment 2: CSP Solver Efficiency (BT vs BT+FC vs BT+FC+AC-3)
python run_experiment2.py

Experiment 3: System Integration Baseline Comparison
python run_experiment3.py

To verify that all modules (Data generation, NN, Search, CSP, and Integration) pass tests:
pytest