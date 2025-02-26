# Wine Quality Prediction with Machine Learning

## Project Overview
This project implements a machine learning model to predict wine quality based on physicochemical properties. Using a Random Forest Classifier, the model analyzes various attributes of wine to predict quality ratings on a scale of 0-10.

## Dataset
The dataset (`wine-quality.csv`) contains 1599 wine samples with 11 features describing their chemical properties:

| Feature | Description |
|---------|-------------|
| Fixed acidity | Most acids involved with wine are fixed or nonvolatile (do not evaporate readily) |
| Volatile acidity | Amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste |
| Citric acid | Found in small quantities, citric acid can add 'freshness' and flavor to wines |
| Residual sugar | Amount of sugar remaining after fermentation stops (wines with >45 g/L are considered sweet) |
| Chlorides | Amount of salt in the wine |
| Free sulfur dioxide | Free form of SO₂ that prevents microbial growth and wine oxidation |
| Total sulfur dioxide | Amount of free and bound forms of SO₂ (becomes evident in taste at >50 ppm) |
| Density | Depends on percent alcohol and sugar content (close to water's density) |
| pH | Describes acidity on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 |
| Sulphates | Wine additive that can contribute to sulfur dioxide levels, acting as antimicrobial and antioxidant |
| Alcohol | Percent alcohol content of the wine |

The target variable is `quality`, a score between 0 and 10 based on sensory data.

## Dependencies
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```

## Model
The project implements a Random Forest Classifier to predict wine quality. Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the class that is the mode of the classes of the individual trees.

## Implementation
1. **Data Loading**: Loading the wine quality dataset from CSV
2. **Data Preprocessing**: Handling missing values and feature scaling if necessary
3. **Exploratory Data Analysis**: Visualizing relationships between features and quality
4. **Feature Selection**: Identifying the most influential features for prediction
5. **Model Training**: Using RandomForestClassifier with optimized hyperparameters
6. **Evaluation**: Measuring model performance using accuracy metrics

## Getting Started

### Prerequisites
- Python 3.6+
- Required libraries listed in the dependencies section

### Installation
```bash
# Clone the repository
git clone https://github.com/ab-tech-dev/wine-quality-prediction.git
cd wine-quality-prediction

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
# Run the training script
python train_model.py

# Make predictions with the trained model
python predict.py
```

## Results
The model's performance was assessed using the accuracy score on both the training and testing datasets. The model achieved an accuracy of 91.25%, indicating a strong ability to classify wine quality correctly. Feature importance analysis was conducted, revealing which chemical properties contribute the most to wine quality prediction. This analysis helps identify key factors such as volatile acidity, alcohol content, and sulphates, which have a significant impact on classification outcomes.

## Future Work
- Implement hyperparameter tuning for optimizing model performance
- Explore other classification algorithms for comparison
- Deploy the model as a web application for real-time predictions
- Expand the dataset to include other types of wines



## Contributors
[ab-tech-dev](https://github.com/ab-tech-dev)

## References

- P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. "Modeling wine      preferences by data mining from physicochemical properties." In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
- Breiman, L. (2001). "Random Forests". Machine Learning. 45 (1): 5–32.
scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
- McKinney, W. (2010). "Data Structures for Statistical Computing in Python." Proceedings of the 9th Python in Science Conference, 51-56.



## Acknowledgments
- UCI Machine Learning Repository for maintaining the original dataset
- The authors of the original research paper: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis
- Kaggle for hosting the dataset and making it accessible