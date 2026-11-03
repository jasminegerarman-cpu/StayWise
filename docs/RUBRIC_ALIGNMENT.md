# Rubric alignment — CpE Elective 1 Final Project

**Source:** instructor's supplied *CpE Elective 1 FINAL PROJECT* handout, page 3.

| Exact assessed criterion | Maximum | Evidence in StayWise |
|---|---:|---|
| Problem Relevance & Dataset Quality | 20 | Problem, intended hotel-staff users, published original 119,390-booking dataset, methodology and leakage safeguards (`docs/PROJECT_PROPOSAL.md`, `docs/DATASET_AND_METHODOLOGY.md`). |
| Technical Implementation & Functionality | 35 | Real Streamlit front end, saved scikit-learn model, preprocessor, 20 editable inputs, prediction/validation/errors, CSV batch inference, history, scenario explorer, Python tests. |
| Innovation, Design & Prototype Originality | 20 | Consistent branded interface, structured operations dashboard, scenario comparison, real charts and assessment flow, local-first anonymous activity history. This is a design choice, not a guarantee of grade. |
| Presentation & Demonstration | 15 | `docs/DEMO_AND_PRESENTATION_GUIDE.md` live script, sample CSV, training metrics, risk scenarios, screenshots to capture. |
| Q&A Handling | 10 | `docs/DEFENSE_QA.md`; explain split, features, target, baseline, limits, why results differ. |

## Five minimum application capabilities from handout page 2
1. **Input:** form for one reservation and upload for multiple bookings.
2. **Prediction:** a trained Logistic Regression or Random Forest pipeline supplies class-1 cancellation probability.
3. **Results:** explicit outcome, predicted percentage and risk-band interpretation.
4. **Validation:** guards against invalid numeric, zero guests/nights and missing CSV columns/rows.
5. **Usable interface:** clear navigation, guided steps, sample scenarios and help page.

## Handout-required final deliverables
- **Approved project proposal:** `docs/PROJECT_PROPOSAL.md` is a **draft only**. Student must obtain the instructor's real approval.
- **Dataset and preparation documentation:** source, target, original row count, columns, cleaning, split and imputation documented.
- **GitHub repository:** upload source, training/reproducibility instructions, evaluation JSON and model file after real local training; do not publish user assessment history.
- **Functional prototype:** demonstrate downloaded dataset + trained model in Streamlit with live input.
- **Final presentation:** prepare slide deck using `docs/DEMO_AND_PRESENTATION_GUIDE.md` and measured results; no slide deck is asserted to be included here.

**AI disclosure:** the handout requires disclosure of AI tools used, and the student must explain code/data/model/results.

**Honesty:** no dataset, model, or test metrics are shipped as fabricated placeholders. First launch downloads and trains on an actual public historical dataset. Until it runs successfully, the ML portion is *not yet trained* on that laptop.
