# StayWise
### A Machine Learning-Based Hotel Booking Cancellation Prediction System

**CpE Elective 1 — Applied Machine Learning and Software Development**

StayWise is an academic web application featuring a professional hotel analytics dashboard, model-based booking cancellation assessments, batch predictions, local history, and reproducible real-data model evaluation.

## One-click Windows setup

1. Extract the entire ZIP file (do not try to start it inside the ZIP).
2. Install **Python 3.10 or newer** from https://www.python.org/downloads/ and check **Add Python to PATH** during installation.
3. Connect to the Internet for the first launch.
4. Double-click **`START_STAYWISE.bat`**.
5. The launcher creates a `.venv`, installs the requirements, downloads the **real original dataset**, trains and compares two models, and starts a local Streamlit web application.
6. The launcher **explicitly opens your default browser** at the selected local address (starting at `http://localhost:8501`, or the next free port).
7. Keep the black console window open while using StayWise. Close it or press `Ctrl+C` to stop.

**Important:** The first launch is not fully offline because the original 119,390-row CSV and pre-trained model are **not inside the ZIP**. After the first successful setup, future starts normally skip installation, dataset download, and retraining. Model artifacts and dataset are preserved on your computer. There is no made-up fallback model.

### One-click VS Code: launch straight into the website

**First-time setup:** Double-click `START_STAYWISE.bat` once to install dependencies, download the original dataset, and train the model. After that, keep using these methods:

- **Top-right ▶ Run Python File:** With `app.py` open and the `.venv` interpreter selected, click Run Python File. The app starts Streamlit using `launch_staywise.py`, waits for the server to become reachable, and explicitly opens your default browser.
- **VS Code Run and Debug (F5 / Ctrl+F5):** Choose `StayWise: Run and Open Website` in the Run and Debug menu. The included `.vscode/launch.json` launches the dedicated browser-opening script.
- **Double-click `START_STAYWISE.bat`:** No need to use VS Code.

For any method, keep its terminal open while using the website. The port may be `8501`, `8502`, or another available port; Streamlit chooses the free port. Avoid starting multiple copies at once. If Chrome does not open automatically, look for the Local URL in the terminal.

### Manual setup / VS Code

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

python -m pip install -r requirements.txt
python setup_project.py
python launch_staywise.py
```

For Linux/macOS, `./START_STAYWISE.sh` is available if `python3-venv` is installed.

### Troubleshooting
- **Python not recognized**: reinstall Python with Add to PATH, or use `py -3` on Windows.
- **Unable to download dataset**: check internet / firewall. Manually download `hotels.csv` from https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-02-11/hotels.csv and rename to `data/hotel_bookings.csv`.
- **Cannot access localhost**: keep the launcher window open and navigate to http://localhost:8501.
- **Dependency installation failed**: try a fresh Python 3.11 or 3.12 install, delete `.venv` and `.setup_complete`, rerun the launcher.
- **Need new model / performance results**: run `python setup_project.py --force` and then restart Streamlit.
- **File not found**: extract all ZIP contents to a normal folder before starting.

## Navigation

| Section | What it does |
|---|---|
| **Overview** | Real historical booking statistics, monthly cancellation chart, assessment activity |
| **New assessment** | Booking input with validation, ML probability, risk band, follow-up guidance, stored anonymous history |
| **Scenario explorer** | Compare model predictions after varying lead time, daily rate, and special requests (association only; not causal inference) |
| **Batch analysis** | CSV template, CSV upload validation, bulk ML inference up to 5,000 rows, CSV export |
| **Model report** | Comparison of Logistic Regression and Random Forest using actual held-out metrics, confusion matrix, global feature signals |
| **About & guide** | Data source, methodology, limitations, live-demo steps, history erase control |

## Training methodology

- Historical data: 119,390 original bookings from 2 hotels in Portugal, July 2015–August 2017.
- Binary target: `is_canceled`: **1 = canceled; 0 = not canceled**.
- Before training, remove invalid booking rows (zero guests / zero nights / negative nightly price), report retained total.
- Features: 20 **pre-outcome** booking fields. No cancellation outcome or post-outcome status columns.
- Numeric values: median imputation and scaling. Categorical values: one-hot encoding with `handle_unknown="ignore"`.
- Split: stratified 60/20/20 training/validation/test, `random_state=42`.
- Models: Logistic Regression baseline and Random Forest alternative; preprocessor fit **on training data only** for each.
- Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, classification report.
- Select by highest canceled-class **F1 on the validation set** (ROC-AUC tie breaker). Then refit the selected algorithm on training+validation (80%) and report one independent final test evaluation on the remaining 20%.
- Model output uses explicit `classes_` to extract class-1 cancellation probability.
- Probability threshold 0.50 for predicted cancellation; risk bands <30%, 30–69.99%, ≥70% are **interface categories**, not validated statistical thresholds.

## Project folder structure

```text
StayWise_Professional_ML_Project/
├── START_STAYWISE.bat       <- Double-click on Windows
├── START_STAYWISE.sh
├── app.py                   <- Professional multipage Streamlit UI
├── launch_staywise.py       <- Waits for server, opens browser automatically
├── setup_project.py         <- Download + train on first run
├── download_data.py         <- Retrieve and verify original CSV
├── train_model.py           <- Two ML algorithms + real evaluation
├── requirements.txt
├── requirements-dev.txt    <- Optional pytest for software tests
├── src/
│   ├── config.py            <- 20-feature schema, exact target labels
│   ├── pipeline.py          <- Preprocessing + robust probability API
│   ├── history.py           <- Local anonymous SQLite history
│   └── theme.py             <- Professional responsive CSS
├── data/                   <- CSV created on first run
├── models/                 <- .joblib and .json created on first run
├── tests/
└── docs/                   <- Proposal, methodology, demo, Q&A
```

## Reliability and ethics

The dataset is historical and concerns two hotels, so model outputs cannot be assumed accurate for arbitrary current hotels. No real-time bookings, payments, or third-party hotel systems are accessed. Probabilities are model-estimated, not guarantees or calibrated real-world forecasts. Do not refuse service or charge guests based solely on risk scores. The SQLite history saves only anonymous reservation details and can be cleared from the Guide page.

**AI disclosure:** Cite any AI assistance you actually used while developing the project, as required by the course brief. Make sure you can explain your code, model and results during Q&A.

**Attribution:** António, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126. TidyTuesday public dataset mirror: https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-02-11.

## Verification and instructor deliverables
- `python -m pip install -r requirements-dev.txt` then `python -m pytest -q` — run unit tests; these use small constructed **software test inputs only**, never as the academic training dataset.
- `python verify_project.py` — report setup status (requires dataset + model after first launch).
- `models/evaluation.json` — populated with **measured** model metrics on your laptop; attach to the GitHub submission.
- `docs/RUBRIC_ALIGNMENT.md` — five rubric criteria and evidence to demonstrate.
- `docs/StayWise_Final_Presentation_Template.pptx` — editable 10-slide PowerPoint; replace [ACTUAL] model values and presenter details after training.
- `docs/PROJECT_PROPOSAL.md` — **draft**, not automatically approved. Obtain actual instructor approval.
- `docs/AI_ASSISTANCE_DISCLOSURE.md` — accurately disclose how AI assisted with the project.

### GitHub notes
Source code can be uploaded directly. Model binary and dataset are excluded from Git by default due to size: after training and verifying licensing/submission rules, you may intentionally use `git add -f models/staywise_pipeline.joblib models/evaluation.json` for your academic deliverables, or provide the downloadable training workflow if the GitHub size limit prevents uploading. Do not upload your local `.venv` or personal assessment history.
