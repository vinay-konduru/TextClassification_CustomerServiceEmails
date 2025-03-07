BERT-Based Text Classification for Customer Service Emails

Overview :-

This project implements a BERT-based text classification model to categorize customer service emails into predefined categories (e.g., Billing, Technical Issue, General Inquiry). The model 
is fine-tuned using Hugging Face's BertForSequenceClassification and achieves high accuracy in predicting email categories.

Features :-

Uses a pre-trained BERT model (bert-base-uncased) for feature extraction and fine-tuning.

Supports multi-class classification with custom-defined labels.

Implements PyTorch for model training and inference.

Includes data preprocessing, tokenization, and model evaluation.

Dataset :-

The model is trained on a dataset of labeled customer service emails. The labels are mapped as follows:

label_map = {0: "Accounting", 1: "Hardware", 2: "Software"}

Training the Model :-

Loads the dataset.

Tokenizes the text using BERT tokenizer.

Trains the BertForSequenceClassification model.

Evaluates performance on the validation set.

Inference :-

To make predictions using a trained model:

Example input "IM UNABLE TO SEND WATSAPP STICKERS."

Example output:

Predicted Class: Software Issue

Evaluation Metrics :-

Accuracy: Measures the percentage of correctly classified emails.

Precision, Recall, F1-Score: Evaluates model performance per class.

Confusion Matrix: Provides insights into misclassified categories.

Next Steps :-

Deploy the model using FastAPI for real-time inference.

Fine-tune on a larger dataset for better generalization.

Implement explainability techniques (e.g., SHAP, LIME) for interpretability.

Author :-
Vinay Konduru
