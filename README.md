Context & Objective
Heart disease remains one of the leading causes of mortality globally. Early diagnosis is crucial but often relies on complex, isolated medical tests. The goal of this project was to determine if we could reliably predict the presence of heart disease using a standard set of 13 clinical attributes (such as age, cholesterol levels, and maximum heart rate).

Methodology
I approached this as a binary classification problem (Presence vs Absence), focusing on interpretability and performance.

Data Processing: Handled categorical variables (like Chest Pain Type and Thallium stress results) using one-hot encoding and standardized numerical features to ensure model stability.

Model Selection: I benchmarked Logistic Regression against a Random Forest Classifier.

Evaluation: While Random Forest is often powerful, Logistic Regression proved to be the superior model for this specific dataset, likely due to the linear separability of the key risk factors.

Key Results
The final model achieved an accuracy of 87% on the test set.

Precision (94%): The model is highly conservative; when it flags a patient as "at risk," it is almost always correct.

Recall (71%): It successfully identifies the majority of positive cases, though there is room to improve sensitivity in future iterations.