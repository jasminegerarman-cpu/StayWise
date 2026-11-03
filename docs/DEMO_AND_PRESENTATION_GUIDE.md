# FINAL PRESENTATION — CONTENT & LIVE DEMO

## Slide 1 — Project title
**StayWise: A Machine Learning-Based Hotel Booking Cancellation Prediction System**
CpE Elective 1 – Applied Machine Learning and Software Development
Presenter: [Your name]

**Script:** "Good day. My final project is StayWise, a web-based system that uses machine learning to estimate hotel booking cancellation risk. It combines a trained classification model with a functional booking assessment application."

## Slide 2 — Problem and users
Problem: cancellations make room inventory and operations planning less predictable. Users: hotel staff and managers.

**Script:** "Hotel teams benefit from knowing which reservations need routine confirmation or attention. StayWise does not replace hotel staff decisions; it gives an estimate from historical booking information."

## Slide 3 — Project objectives
Real-world booking cancellation prediction; valid original dataset; train and compare models; integrate in working software; validate input and show useful results.

## Slide 4 — Dataset and features
Source: António et al. (2019), via TidyTuesday. Original rows: **119,390**, two Portuguese hotels, 2015–2017. Prediction target: `is_canceled` (1 yes, 0 no). Modeling count: **[insert actual from evaluation.json]**. 20 selected booking features.

## Slide 5 — Methodology
Dataset → clean invalid bookings → select 20 pre-outcome features → 60/20/20 stratified train/validation/test → fit per-model preprocessing → train Logistic Regression and Random Forest → compare on validation F1 → refit selected model on training+validation → independent test → Streamlit.

## Slide 6 — Results
Copy **actual** numbers from Model report after running training. Report validation accuracy, precision, recall, F1, ROC-AUC **for both models**; report **independent test metrics for the selected model** and show its confusion matrix. Do not invent model performance values.

## Slide 7 — System architecture
Input form or CSV → input validation → preprocessing pipeline (median imputation, scaling, one-hot encoding) → selected trained classification model → probability + risk category + result table → optional local history / export.

## Slide 8 — System demonstration (LIVE)
1. Launch `START_STAYWISE.bat`.
2. Explain overview metrics and the difference between historic dataset metrics and saved assessments.
3. Click **New assessment** → choose **Typical booking** → Analyze → discuss the result and risk category.
4. Try an invalid booking: zero guests or zero nights → show validation.
5. Show **Scenario explorer**: adjust lead time and discuss changing *model scores*, not causation.
6. Show **Batch analysis**: download CSV template, upload template, click Analyze, export CSV.
7. Open **Model report**: explain validation model selection and independent final test metrics.

## Slide 9 — Limitations and responsibility
Historical records from two hotels; no live reservations; scores not independently calibrated; do not automatically penalize guests based on a model prediction.

## Slide 10 — Conclusion
The system demonstrates both genuine machine learning and software development: a trained model integrated into an interactive, validated, exportable tool.

## Important defense reminder
Understand every feature and line of code you demonstrate. Disclose AI tools actually used during your project development, as the rubric requires.
