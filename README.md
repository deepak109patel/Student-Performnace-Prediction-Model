# Student-Performnace-Prediction-Model

This project aims to predict student performance based on various features such as gender, race/ethnicity, parental level of education, lunch type, and test preparation course. The dataset used in this project is the **Students Performance in Exams** dataset, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams).

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Details](#model-details)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

---

## Project Overview
The goal of this project is to build a machine learning model that predicts student performance (e.g., math score) based on various features. The project involves:
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and evaluation
- Prediction on new data

---

## Dataset Description
The dataset contains the following features:
- **gender**: Student's gender (male/female)
- **race/ethnicity**: Student's race/ethnicity (group A, B, C, D, E)
- **parental level of education**: Parent's education level (e.g., some high school, bachelor's degree, etc.)
- **lunch**: Type of lunch (standard/free/reduced)
- **test preparation course**: Whether the student completed the test preparation course (none/completed)
- **math score**: Student's math score (target variable)
- **reading score**: Student's reading score
- **writing score**: Student's writing score

---

## Installation
To run this project, you need to have Python installed. Follow these steps to set up the environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/student-performance-prediction.git
   cd student-performance-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and place it in the `data/` directory.

---

## Usage
1. **Data Preprocessing**:
   Run the `data_preprocessing.py` script to clean and preprocess the data:
   ```bash
   python data_preprocessing.py
   ```

2. **Model Training**:
   Train the model using the `train_model.py` script:
   ```bash
   python train_model.py
   ```

3. **Prediction**:
   Use the trained model to make predictions on new data:
   ```bash
   python predict.py
   ```

---

## Model Details
The model used in this project is a **Random Forest Regressor**. The following steps are involved in the pipeline:
1. **Preprocessing**:
   - Numerical features are scaled using `StandardScaler`.
   - Categorical features are one-hot encoded using `OneHotEncoder`.

2. **Model Training**:
   - The dataset is split into training and testing sets (80% training, 20% testing).
   - The model is trained on the training set.

3. **Evaluation**:
   - The model's performance is evaluated using **Mean Squared Error (MSE)** and **R-squared**.

---

## Results
The model achieved the following performance metrics:
- **Mean Squared Error (MSE)**: 25.34
- **R-squared**: 0.89

These results indicate that the model performs well in predicting student performance.

---

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Submit a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Dataset: [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- Libraries: Pandas, Scikit-learn, NumPy

