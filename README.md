# Wine Quality Prediction using SVM Optimization

This project applies Support Vector Machine (SVM) classification to the Wine Quality dataset from the UCI Machine Learning Repository. It performs random hyperparameter tuning over 10 randomized train/test splits and visualizes the accuracy convergence over iterations.


##  Objective

- Build a **multi-class classification model** using SVM.
- Generate **10 randomized train-test splits** (70% train, 30% test).
- For each split:
  - Randomly sample SVM parameters over 100 iterations.
  - Track and store the best accuracy and corresponding parameters.
- Plot a **convergence graph** showing how the best accuracy improves during optimization.



##  Dataset

- **Source:** [Wine Quality - Red Wine Data Set](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- **Type:** Multi-class classification
- **Samples:** 1599 red wine instances
- **Features:** 11 physicochemical inputs (e.g., acidity, pH, alcohol)
- **Target:** Wine quality score (integer from 3 to 8)



##  Technologies Used

- **Python 3**
- **Libraries:**
  - `pandas`, `numpy` – data processing
  - `scikit-learn` – SVM model and evaluation
  - `matplotlib` – plotting convergence graph
  - `tqdm` – progress bars for loops



## How to Run

1. Open the `svm_wine_quality.ipynb` notebook using [Google Colab](https://colab.research.google.com/) or Jupyter Notebook.
2. Run the cells from top to bottom.
3. The notebook will:
   - Load and standardize the dataset.
   - Create 10 randomized train/test splits.
   - For each split, randomly tune `kernel`, `C`, and `gamma` over 100 iterations.
   - Save the best accuracy and parameters for each sample.
   - Plot a convergence graph of the best-performing sample.



##  Output
### 📋 Results Table
Displays best accuracy and parameters for each of the 10 data splits:

![image](https://github.com/user-attachments/assets/32ae4522-a57e-4fca-913f-8d363ee08957)


### 📊 Convergence Graph
Line chart showing improvement of best accuracy over 100 iterations for the best sample:

![image](https://github.com/user-attachments/assets/19fda0a6-df67-4828-8ac8-c3f6e37935c7)



## Conclusion


This project demonstrates how Support Vector Machines (SVM) can be effectively applied to a real-world multi-class classification problem, such as predicting wine quality. By generating multiple randomized data splits and using random hyperparameter tuning, we observed how model performance varies with different configurations. 

The convergence graph provided insight into how model accuracy can improve across iterations, helping to visualize the optimization process. While random search is simple and fast, more advanced techniques like grid search or genetic algorithms could further enhance performance.

Overall, this experiment highlights the importance of hyperparameter tuning and evaluation across multiple data samples for building reliable machine learning models.



