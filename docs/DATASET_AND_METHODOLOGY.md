# DATASET AND MODEL PREPARATION DOCUMENTATION

## Dataset provenance
Full public original dataset: **Hotel Booking Demand** (António, de Almeida & Nunes, 2019). The TidyTuesday public-hosted data mirror provides a CSV with **119,390 historical booking observations**, collected across two hotels in Portugal between July 2015 and August 2017. TidyTuesday mirror: https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-02-11. Source paper: https://doi.org/10.1016/j.dib.2018.11.126.

This project automatically downloads the original dataset from the documented URL(s) and performs a header-and-count integrity check. No synthetic, duplicated or augmented records are used to meet the five-thousand-original-record rule. It may contain repeated-looking reservations which are legitimate separate observations; the cleaning pipeline does not artificially duplicate or indiscriminately de-duplicate bookings.

## Prediction objective
Binary target `is_canceled` (1 = canceled, 0 = not canceled). Cancellation is predicted based on booking details, not status fields that become available after the outcome is known.

## Selected input features (20)
1. `hotel`
2. `lead_time`
3. `arrival_date_month`
4. `stays_in_weekend_nights`
5. `stays_in_week_nights`
6. `adults`
7. `children`
8. `babies`
9. `meal`
10. `market_segment`
11. `distribution_channel`
12. `is_repeated_guest`
13. `previous_cancellations`
14. `previous_bookings_not_canceled`
15. `reserved_room_type`
16. `deposit_type`
17. `customer_type`
18. `adr` (average daily rate in historical data units)
19. `required_car_parking_spaces`
20. `total_of_special_requests`

## Removed/unselected variables
`reservation_status`, `reservation_status_date` are post-outcome data and would cause label leakage. `assigned_room_type` and `booking_changes` can reflect updates after the initial booking. Additionally, excluded columns include high-cardinality or operationally unsuitable fields such as `agent`, `company`, `country`, and others not needed in the live form. The target column itself is never included in the features.

## Preprocessing
- Load original CSV, check all required columns.
- Parse numeric inputs; missing numeric values imputed by training-set medians.
- Treat missing categorical values as `Unknown`; one-hot encode categorical inputs and ignore unknown categories at inference time.
- Discard source rows where the target is invalid, zero guests, zero length of stay, or negative ADR (if present).
- Retain eligible original booking observations without adding fabricated data.
- Use a stratified random 60/20/20 training/validation/independent-test split (`random_state=42`).
- Fit preprocessing independently inside each training pipeline, avoiding exposure of test data to imputers or encoders.

## Compared algorithms
**Logistic Regression (baseline)** and **Random Forest (alternative)**. Split the cleaned original rows into 60% training, 20% validation and 20% untouched test, stratified by cancellation. Compare both algorithms on the **validation set**, selecting the higher canceled-class F1 (ROC-AUC tie breaker). Refit the selected algorithm on combined training+validation data, then evaluate on the untouched test split **exactly once**. `models/evaluation.json` stores actual measured accuracy, precision, recall, F1, ROC-AUC, confusion matrix and classification report. No scores are asserted before training.

## Evaluation caveat
The independent test is kept untouched during model selection. Remaining limitations: the split is random across historical years (rather than a temporal forward-validation strategy), and neither model nor probability calibration has been prospectively verified in a modern operating hotel.

## Software development
Streamlit front end, scikit-learn Pipeline, trained model saved with joblib, SQLite for anonymous user assessment history, Plotly for actual historical charts and confusion matrix, batch prediction with CSV import/export, validation with clear error messages.

## Evaluation evidence to capture after launch
1. Startup screenshot (indicates successful real data verification and training).
2. Terminal output displaying actual numbers of raw and retained records.
3. `models/evaluation.json` including both algorithms' validation metrics and the selected model's independent test metrics.
4. Screenshot of main dashboard.
5. Screenshot of completed input form and prediction result.
6. Batch-upload and export demonstration.
7. Screenshot of model comparison / confusion matrix.

## Reproducibility commands
```bash
python setup_project.py --force
python -m streamlit run app.py
```

## Research, ethics and scope
This is an educational prototype. Historical two-hotel data (2015–2017) might not reflect other hotels or contemporary patterns. The model's probability is an estimate, not a verified real-world rate. The product must not be used for automated discrimination, penalties, or guest denial. The instructor must be informed of AI tools actually used to develop this project.
