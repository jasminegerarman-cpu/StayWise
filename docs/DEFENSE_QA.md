# DEFENSE Q&A — PRACTICE ANSWERS

**Q1. What problem does the system solve?**
It estimates which hotel bookings are more likely to be canceled from reservation features. Staff can use the estimate to inform routine confirmation and planning, not automatic restrictions.

**Q2. What is the target?**
`is_canceled` is the binary label. Value 1 means canceled, and 0 means the booking was not canceled.

**Q3. Why did you choose this dataset?**
It is based on real historical hotel bookings, with 119,390 original observations, sufficient for the course's minimum requirement and suitable for supervised classification.

**Q4. Which algorithms were trained?**
Logistic Regression as a baseline and Random Forest as an alternative. The script trains both, compares metrics on a separate validation split, selects by canceled-class F1, refits the winner, and measures final test performance on a previously untouched split.

**Q5. Why F1 instead of accuracy alone?**
F1 balances precision and recall for the canceled class. Accuracy can hide the tradeoff between false alarms and missed cancellations when class frequencies are unequal.

**Q6. What data leakage did you prevent?**
The target `is_canceled` is not an input. We exclude `reservation_status`, `reservation_status_date`, assigned room, and booking changes which can reveal events after initial booking. Each preprocessing pipeline is fitted on the training partition only.

**Q7. Does a 90% risk mean cancellation is certain?**
No. It is the model's estimated probability for class 1, not a guarantee or an independently calibrated real-world cancellation rate.

**Q8. How are risk labels determined?**
Low below 30%, moderate 30–69.99%, high at 70% or above. They are an interface convention, not learned or validated business policies. Binary classification at 50% is also a configurable policy threshold.

**Q9. What are the training/validation/test splits?**
60% training, 20% validation, 20% final untouched test, stratified by canceled/not canceled and with `random_state=42`. We compare both models on validation data. The winner is then refitted using the 80% combined training+validation records, and the final 20% test is used only once for independent evaluation.

**Q10. What are the limitations?**
The data are historical bookings from two hotels in Portugal (2015–2017). They may not generalize to new hotels or current market conditions. We have not independently calibrated probabilities.

**Q11. Is it a static website?**
No. The app calls `predict_proba` on a trained scikit-learn Pipeline using actual user inputs. Prediction outcomes change with input.

**Q12. How does bulk prediction work?**
The CSV must include all required feature columns. Each row is validated, passed through the same preprocessing + model pipeline, and returned with cancellation probability and labels in a downloadable CSV.

**Q13. Where is the saved model?**
`models/staywise_pipeline.joblib` after first run. `models/evaluation.json` contains the measured model comparison and dataset counts.

**Q14. Why is Streamlit used?**
It offers rapid construction of a functional, Python-based interface with input widgets, charts, error handling, and file download in one application.

**Q15. Did you use AI development assistance?**
State exactly which tools were used and what they helped with, and demonstrate that you understand and can explain the delivered implementation and results.

**Q16. Does Scenario Explorer show what *causes* cancellations?**
No. It changes inputs to an observational machine-learning model and compares predicted scores; that is not experimental evidence of causation.

**Q17. Why does your model training not use actual reservation status?**
The reservation status is determined after a booking outcome is known. Using it in advance prediction would be target leakage, producing misleading evaluation.

**Q18. Have you tested the completed app on the Windows laptop?**
Before defense, run the local launcher, test a single prediction, invalid input, batch CSV and export. Only claim those actions were tested after doing them yourself. Model accuracy cannot be reported before the real dataset is downloaded and training completes.
