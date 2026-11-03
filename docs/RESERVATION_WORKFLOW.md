# StayWise — Reservation Management Workflow

## What this page is
A **session-only demonstration** where hotel staff can record **anonymous mock reservation details**, use the already-trained model to estimate the chance of cancellation, and optionally record a **staff-reported outcome** later. It is **not** a room booking or payment engine and does **not** create, confirm, reject or cancel real hotel stays.

## How to demonstrate
1. Open **Reservations** from the sidebar.
2. Enter a **fictional** reservation scenario and select **Record demo reservation**.
3. Scroll to **Review and analyze recorded reservations**, choose the generated `SW-...` ID.
4. Click **Analyze this reservation** to get the real machine-learning score and risk level.
5. Explain that the estimate is about *future cancellation risk*, not an actual canceled booking.
6. If illustrating an actual historical outcome, the *staff* can manually select **Guest canceled (staff-reported)**. This does not cancel anything and must be differentiated from the prediction.
7. Use **Download demo reservation summary CSV** to export anonymous records.

## Technical and ethical limitations
- Records are stored in Streamlit session state: they may disappear when the tab is reloaded or the session ends. They are **not** accessible or editable across browsers.
- **Never enter real guest names, emails, phone numbers, ID numbers, contact details or financial data.** The public demo has no authentication.
- The model is trained on 2015–2017 historical bookings. Scores are estimates, not guarantees or necessarily calibrated probabilities.
- The system must not automatically decline guests, charge deposits, cancel reservations or make binding business decisions based on a score.
- For real hotel deployment, a separate authenticated booking database, privacy safeguards, staff permissions, audit logs and a verified integration with the hotel's PMS would be required. Those are **not** implemented.

## Relation to academic rubric
The addition strengthens **input**, **functional application**, **user interaction**, **clear results** and **innovation** without changing the ML source data, class labels or evaluation. See `docs/RUBRIC_ALIGNMENT.md` for the full mapping.
