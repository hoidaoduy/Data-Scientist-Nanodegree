# Udacity Data Scientist Nanodegree Capstone Project

This repository contains all code and reports for my Udacity Data Scientist Nanodegree Capstone project.

## Starbucks Capstone Challenge: Using Starbucks App User Data to Predict Effective Offers

### 1. Installation
This project requires Python 3.* and the following Python libraries:

1. Pandas
2. Numpy
3. Sklearn
4. Tensorflow
5. Keras
6. Matplotlib.pyplot

### 2. Project Motivation
This project is the capstone project of the Data Scientist Nanodegree program with Udacity. As a student in this program, we were given the option to participate in the Starbucks Capstone Challenge.

In this project, I discovered the following characteristics:
- Starbucks customers are primarily male.
- Customers are predominantly aged between 45-65 years.
- Customers have incomes ranging from $50,000 to $75,000.
- A significant portion of the customer base became members from 2016 onward, with a notable increase in 2018.
- Discount offers have a higher usage rate compared to BOGO (Buy One Get One) offers.
- Discounts tend to attract customer attention more than BOGO offers.
- The acceptance rates for BOGO and discount offers are similar.

To answer the question, "What factors respond best to which type of offer?":
- It was observed that the reward granted upon completing an offer (reward) and the membership duration (membership_days) are the two most critical factors influencing offer effectiveness.

To address the question, "Which model should be chosen?":
- My decision to use three separate models to predict the effectiveness of each type of offer yielded high accuracy for two models (0.79 for Deep Neural Network (DNN) and 0.7872 for Simple Neural Network), while the Random Forest model was overfitting. However, I would consider 78% to be acceptable in a business context, as informing users about offers incurs no additional cost.

### 3. File Descriptions
This repository contains four files. The report on my project is titled 'Starbucks_Capstone_notebook.ipynb'.
The data used in the project can be found in the files `portfolio.json`, `profile.json`, and `transcript.json` in `data` folder.

### 4. Blog post
Blog: [Analyzing Customer Behavior at Starbucks for Offer Effectiveness](https://forexinsightsweekly.blogspot.com/2024/11/analyzing-customer-behavior-at.html)

### 5. Licensing, Authors, Acknowledgements, etc.

The data for the project was provided by Udacity.
