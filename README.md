![Project Banner](Project_banner.png)
![License](https://img.shields.io/github/license/Sanjai-Chippukutty/cfDNA-Lung-Cancer-ML)
![Made With](https://img.shields.io/badge/Made%20with-Python%20%7C%20ML-blue)
![Repo Size](https://img.shields.io/github/repo-size/Sanjai-Chippukutty/cfDNA-Lung-Cancer-ML)
![Last Commit](https://img.shields.io/github/last-commit/Sanjai-Chippukutty/cfDNA-Lung-Cancer-ML)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live--Demo-orange?logo=streamlit)](https://cfDNA-miRNA-lungcancer.streamlit.app/)
![Stars](https://img.shields.io/github/stars/Sanjai-Chippukutty/cfDNA-Lung-Cancer-ML?style=social)

### Integrative Analysis of cfDNA Methylation and miRNA Expression for Early Lung Cancer Detection Using Machine Learning
# Overview
This project focuses on the early detection of lung cancer using an integrated analysis of two powerful non-invasive biomarkers

##  Live Demo

 [Click here to try the Streamlit app](https://cfdna-lung-cancer-ml-gytw4zojirf3wkqxycoakp.streamlit.app/)

 This app uses a trained Random Forest model on cfDNA methylation + miRNA expression features (`gene1`, `gene2`, `gene3`, `miRNA_21`, `miRNA_34a`) to predict lung cancer probability.
 
Cell-free DNA (cfDNA) methylation profiles

# miRNA expression data

We built and evaluated multiple machine learning models to classify lung cancer vs normal samples, highlighting key biomarkers and achieving strong predictive performance. This project is suitable for clinical research, diagnostic development, and bioinformatics applications.
## Project Structure
cfDNA_LungCancer_ML/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 1_Data_Preprocessing.ipynb
│   ├── 2_Data_Integration_and_Labeling.ipynb
│   ├── 3_Model_Training_and_Evaluation.ipynb
│   ├── 4_Model_Evaluation_and_Visualization.ipynb
│   ├── 5_Model_Inference_and_Usage.ipynb
│   └── 6_Research_Report_and_Literature_Review.ipynb
│
├── results/
│   └── plots/
│
├── models/
├── streamlit_app/
│   └── app.py
│
├── requirements.txt
└── README.md

## Objectives

1. Integrate cfDNA methylation & miRNA expression data.

2. Train classification models for early lung cancer detection.

3. Identify top biomarkers using model interpretability.

4. Visualize model performance (ROC, confusion matrix).

5. Deploy prediction interface using Streamlit.

## Technologies Used

1. Python (NumPy, Pandas, Matplotlib, Seaborn)

2. Machine Learning (Scikit-learn, XGBoost)

3. Bioinformatics concepts (cfDNA, miRNA biomarker profiling)

4. Deployment: Streamlit

5. Model Persistence: joblib

6. Data Source: TCGA-LUAD datasets

##  Datasets

This project uses two primary datasets focused on early-stage lung cancer detection through blood-based biomarkers:

---

### 1. **cfDNA Methylation Dataset**

- **Type**: Cell-free DNA (cfDNA) methylation beta-values  
- **Source**: Public repository (GEO / published supplementary data)  
- **Shape**: 200+ samples × ~10,000 CpG sites  
- **Preprocessing**:
  - Removed low-variance CpGs  
  - Filtered NA/missing values  
  - Normalized values if necessary  
- **Target Labels**: `Benign` vs `Malignant` (based on clinical metadata)

---

### 2. **miRNA Expression Dataset**

- **Type**: Circulating miRNA expression profiles from patient plasma/serum  
- **Source**: Public domain / research publication  
- **Shape**: 200+ samples × ~300 miRNAs  
- **Preprocessing**:
  - Removed low-expression features  
  - Filtered missing values  
  - Log-transformation applied  
- **Target Labels**: Aligned to cfDNA samples via patient ID

---

###  Merged Dataset

After aligning both datasets by patient ID, a **merged matrix** was created containing:
- cfDNA methylation features  
- miRNA expression features  
- Combined target label (Benign vs Malignant)

This merged dataset was used for feature selection and model training.


## Models Trained

1. Logistic Regression

2. Random Forest Classifier

3. Support Vector Machine (SVM)

## Evaluation Metrics

1. Accuracy

2. Precision, Recall, F1-Score

3. Confusion Matrix

4. ROC-AUC Curve

## Key Results

| Model               | Accuracy | AUC Score |
| ------------------- | -------- | --------- |
| Logistic Regression | \~72%    | 0.81      |
| Random Forest       | \~75%    | 0.84      |
| SVM (RBF Kernel)    | \~73%    | 0.82      |

Random Forest highlighted top 20 important integrated features. All models showed balanced performance.

 ##  Project Workflow

This section outlines the complete workflow for our integrative analysis project on **cfDNA methylation and miRNA expression** for early lung cancer detection using machine learning:
##  Project Workflow

The following flowchart outlines the complete pipeline of this project:

<p align="center">
  <img src="Workflow_Diagram.png" alt="Project Workflow" width="700"/>
</p>


###  Overview Diagram

Workflow_Diagram.png README.md

###  Step-by-Step Workflow

1. **Data Collection**  
   - **cfDNA Methylation Data** and **miRNA Expression Data** were gathered from curated sources and stored in structured CSV formats.

2. **Data Preprocessing**  
   - Cleaned missing values, normalized features, and merged datasets using a common identifier.
   - Final processed file: \`merged_labeled_light.csv\`

3. **Feature Engineering**  
   - Selected most informative features from both cfDNA and miRNA matrices.
   - Removed non-informative or redundant columns.

4. **Label Assignment**  
   - Assigned binary labels:
     - \`0\`: Healthy/Control  
     - \`1\`: Cancer/Affected

5. **Model Training and Evaluation**  
   - Built multiple ML models (Random Forest, XGBoost, etc.).
   - Performed hyperparameter tuning and cross-validation.
   - Selected best model based on **accuracy**, **F1-score**, and **ROC-AUC**.

6. **Streamlit App Development**  
   - Developed an interactive web app using Streamlit.
   - Includes:
     - Visualizations
     - Model metrics
     - Live prediction area for user input
   - Integrated with GitHub and deployed via Streamlit Cloud.

7. **Documentation & Deployment**  
   - Full codebase documented and version-controlled on GitHub.
   - Results visualized and interpreted.
   - Repository structured for reuse and reproducibility.



## How to Run the Project

1. Install dependencies:
pip install -r requirements.txt

2. Run Jupyter Notebooks:

. 1_Data_Preprocessing.ipynb
. 2_Data_Integration_and_Labeling.ipynb
. 3_Model_Training_and_Evaluation.ipynb
. 4_Model_Evaluation_and_Visualization.ipynb
. 5_Model_Inference_and_Usage.ipynb

3. Launch Streamlit app:
 cd streamlit_app
 streamlit run app.py

4. Sample Input for Prediction
The app accepts scaled cfDNA and miRNA expression values and classifies the input as:

-> Normal

-> Lung Cancer


## References
 
1. TCGA-LUAD: The Cancer Genome Atlas – Lung Adenocarcinoma

2. GEO Datasets for cfDNA methylation and miRNA

3. Latest studies on non-invasive biomarkers in cancer detection

## Maintainer
Sanjai C.
MS Bioiformatics & Immunobiology
Amrita School of Boitechnology, Amrita Vishwa Vidyapeetham
University of Arizona

Email: sanjaichippukutty@gmail.com
##  Citation


```bibtex
@misc{chippukutty2025cfDNAmiRNA,
  author       = {Sanjai Chippukutty},
  title        = {Integrative Analysis of cfDNA Methylation and miRNA Expression for Early Cancer Detection Using Machine Learning},
  year         = {2025},
  url          = {https://github.com/Sanjai-Chippukutty/cfDNA-Lung-Cancer-ML},
  note         = {GitHub repository}
}


# cfDNA-Lung-Cancer-ML
Integrative analysis of cfDNA methylation and miRNA expression for early lung cancer detection using machine learning.

