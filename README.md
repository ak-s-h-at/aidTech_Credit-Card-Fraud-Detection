## Credit Card Fraud Detection

This project is a part of my Machine Learning internship and aims to develop a credit card fraud detection system. The dataset used for this project is obtained from [Kaggle](https://www.kaggle.com/datasets/isaikumar/creditcardfraud).

### Implementation Steps

1. Data Preprocessing: The dataset was cleaned by removing missing values and encoding categorical variables into numerical representations.

2. Model Training: A logistic regression model was trained using the preprocessed data to detect fraudulent transactions. The code for model training and saving is available in the `model.py` file.

3. Model Evaluation: The trained model was evaluated on a testing set using metrics like accuracy, precision, recall, and F1-score to assess its performance.

4. GUI Development: I created a user-friendly GUI where users can input credit card transaction details. The model predicts whether the transaction is fraudulent or not, and the prediction is displayed to the user.

### Model Performance

While the implemented model shows promising results, it is important to note that there is room for improvement. Fraud detection is a complex problem, and no model is perfect. As part of this project, I gained valuable insights into fraud detection challenges.

### Dataset

The dataset used in this project can be found on Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/isaikumar/creditcardfraud).

### Repository Contents

- `model.py`: Python script containing the code implementation for logistic regression model training and saving.
- `implement.py`: Python script containing the code for GUI development, and model prediction.
- `model.pkl`: The trained logistic regression model.
- `scaler.pkl`: The scaler object used for data scaling.
- `README.md`: Documentation providing an overview of the project and implementation details.

Feel free to explore the code and documentation available in this repository for more details.

