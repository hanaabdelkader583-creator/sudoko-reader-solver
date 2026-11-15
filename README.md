# P7: Handwritten Sudoku Reader and Solver

An end-to-end hybrid AI system that reads handwritten Sudoku puzzles from image mosaics using a calibrated Convolutional Neural Network (CNN), corrects OCR misclassifications via heuristic search ($A^*$, UCS, Greedy), and solves the resulting grid using Constraint Satisfaction Problem (CSP) techniques (Backtracking, Forward Checking, AC-3, and MRV).

## 📁 Project Structure
```text
├── data/               # Raw and processed MNIST/Sudoku datasets
├── results/            # Generated Markdown tables, PNG plots, and CSV logs
├── src/
│   ├── csp/            # Switchable BT, FC, AC-3, and MRV solver modules
│   ├── data/           # Seeded Sudoku puzzle, mosaic, and misread generators
│   ├── eval/           # Benchmarking and metric aggregation scripts
│   ├── integration/    # Joint NN + Search + CSP correction pipeline
│   ├── nn/             # CNN classifier, baseline models, and Temperature Scaling
│   └── search/         # UCS, Greedy Best-First, and A* search implementations
├── tests/              # Pytest suite (normal, edge, and impossible cases)
├── config.py           # Global hyperparameters, seeds, and path configurations
├── make_data.py        # Reproducible dataset download and preprocessing script
├── run_demo.py         # Integrated visual demo & ablation runner
├── run_experiment1.py  # Experiment 1: Search algorithm comparison
├── run_experiment2.py  # Experiment 2: CSP pruning comparison
├── run_experiment3.py  # Experiment 3: System integration baselines
└── requirements.txt    # Pinned Python dependencies

🛠️ Environment Setup
1.Clone the repository:
git clone [https://github.com/hanaabdelkader583-creator/sudoko-reader-solver.git](https://github.com/hanaabdelkader583-creator/sudoko-reader-solver.git)
cd sudoko-reader-solver

2.Create a virtual environment (recommended):
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

3.Install pinned dependencies:
pip install -r requirements.txt

🚀 Exact Run Commands
1.Data Preparation
python make_data.py

2.Run Automated Tests
pytest

3.Run the Live Integrated Demo
Outputs the search trace to the console and saves the visual comparison grid to results/demo.png:
python run_demo.py 3 1

4. Reproduce All Experiments
Experiment 1 — Heuristic Search Comparison (UCS vs. Greedy vs. A*):
python run_experiment1.py
Experiment 2 — CSP Pruning & Heuristics (BT vs. FC vs. AC-3 vs. MRV):
python run_experiment2.py
Experiment 3 — End-to-End Integration Baselines (NN-guided vs. None vs. Random):
python run_experiment3.py




