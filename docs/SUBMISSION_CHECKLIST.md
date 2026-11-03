# Required Submission Checklist (based on CpE Elective 1 brief)

## 1. Approved project proposal
- [ ] Submit `docs/PROJECT_PROPOSAL.md`, edited with your own name/section, for actual instructor approval.
- [ ] Save proof of approval if required.

## 2. Dataset and preparation
- [ ] Confirm `data/hotel_bookings.csv` exists after setup and record its original row count.
- [ ] Submit `docs/DATASET_AND_METHODOLOGY.md` and identify source/target/features.
- [ ] Record exclusions and missing-value handling, and disclose data limitations.

## 3. GitHub Repository
- [ ] Upload this source folder (do not upload `.venv`).
- [ ] Add screenshots, actual `models/evaluation.json` after training, technical docs, and setup guide.
- [ ] Keep the dataset link accessible; upload CSV if your instructor requires it and redistribution is allowed.
- [ ] Trained model: `.joblib` is generated locally. If submitting it, ensure it is included intentionally, since `.gitignore` excludes it by default.

## 4. Functional prototype
- [ ] Open from `START_STAYWISE.bat`.
- [ ] Show Overview, validated input, actual model-driven prediction, and Scenario Explorer.
- [ ] Demonstrate errors, saved history, batch upload and CSV export.

## 5. Presentation and Q&A
- [ ] Insert actual validation-comparison and independent-test metrics from `models/evaluation.json`; never invent accuracy.
- [ ] Use `docs/DEMO_AND_PRESENTATION_GUIDE.md` and `docs/DEFENSE_QA.md`.
- [ ] Disclose which AI tools actually helped you and demonstrate your understanding of the code.

## Project rubric overview
- **Problem relevance & dataset quality:** 20 points.
- **Technical implementation & functionality:** 35 points.
- **Innovation, design & originality:** 20 points.
- **Presentation & demonstration:** 15 points.
- **Q&A handling:** 10 points.

## Before you declare this complete
- [ ] **Approval:** Instructor actually reviewed/approved the proposal.
- [ ] **Real model:** Launch succeeded; `data/hotel_bookings.csv`, `models/staywise_pipeline.joblib`, and `models/evaluation.json` exist.
- [ ] **Evidence:** Capture terminal model-training log and screenshots of dashboard, risk assessment, scenario explorer, upload validation, model report.
- [ ] **Testing:** Record real test execution results; do not claim laptop/browser testing before performing it.
- [ ] **AI transparency:** Include `docs/AI_ASSISTANCE_DISCLOSURE.md` with a truthful statement of AI use.
