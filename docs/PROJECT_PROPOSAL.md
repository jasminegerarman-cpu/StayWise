# PROJECT PROPOSAL

## Project title
**StayWise: A Machine Learning-Based Hotel Booking Cancellation Prediction System**

**Course:** CpE Elective 1 – Applied Machine Learning and Software Development

**Prepared by:** [Enter your actual full name and section]

**Instructor:** [Confirm name with instructor]

## I. Background and rationale
Hotels must plan room inventory, staffing and daily operations despite changes in expected booking arrivals. A cancellation can affect room availability and operating decisions. StayWise uses historical booking patterns to estimate the probability of cancellation before the booking outcome is known. It is an educational decision-support tool, not an automated reservation management system.

## II. Problem statement
Hotel staff often cannot quickly identify which incoming reservations merit routine confirmation or planning attention. This project explores whether a machine-learning application can estimate booking cancellation risk from relevant reservation characteristics.

## III. General objective
To develop and demonstrate a usable web-based machine-learning system for estimating the risk of hotel booking cancellations.

## IV. Specific objectives
1. Source a credible, original booking dataset containing more than 5,000 records.
2. Clean the data and prevent post-outcome information from leaking into model training.
3. Train and compare Logistic Regression and Random Forest classification pipelines.
4. Evaluate the approaches using a reproducible held-out split and five standard metrics.
5. Integrate the selected model into a professional web interface with booking input, validation, clear risk results, batch prediction and export.
6. Demonstrate the app, explain outputs, and report limitations honestly.

## V. Intended beneficiaries
Hotel front-desk staff and operations managers may use the prototype as a demonstration of data-assisted planning. Researchers and students may learn about practical supervised machine learning and system integration.

## VI. Dataset
**Hotel Booking Demand**; António et al. (2019). Full public corpus: 119,390 booking observations from two Portuguese hotels, 2015–2017. Download is automated during first setup. **Target:** `is_canceled` (0: not canceled; 1: canceled). Exact retained modeling count is logged after preprocessing.

## VII. Development tools
Python, pandas, scikit-learn, joblib, Streamlit, Plotly, SQLite.

## VIII. Core functionality
1. Validated user booking input.
2. ML-generated cancellation probability and binary classification.
3. Clear result display with educational risk categories.
4. Usable professional dashboard and historical charts.
5. Multiple-file-row prediction and CSV export, plus anonymous local history.

## IX. Approach
Acquire data → validate and clean → select pre-outcome fields → split training/validation/independent test → fit preprocessing within ML pipelines → train Logistic Regression and Random Forest → compare validation metrics → independently test selected model → save model and results → integrate Streamlit UI → test scenarios and error handling.

## X. Expected deliverables
Approved proposal; dataset/preparation documentation; GitHub repository with project files and model/evaluation after running training; functional prototype; final presentation and live demonstration.

## XI. Known limitations
The published dataset is historical, geographically narrow, and observational. Predictions may not generalize to current hotel operations; model probability estimates are not independently calibrated. Do not deny guests service based on the prediction.

## XII. Reference
António, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief, 22*, 41–49. https://doi.org/10.1016/j.dib.2018.11.126

**Approval status:** Draft for instructor review. Do not present it as approved until your instructor confirms approval.
