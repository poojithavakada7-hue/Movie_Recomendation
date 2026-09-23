# 🎬 Movie Recommendation System Using Machine Learning

A Machine Learning-based Movie Recommendation System that uses **K-Nearest Neighbors (KNN)** and **Support Vector Machine (SVM)** to explore movie recommendations based on movie attributes and IMDb ratings.

This project allows users to discover movies using features such as genre, language, country, release year, director, lead actor, and weather. It also includes an interactive IMDb rating threshold slider for the SVM classification experiment.

## 📌 Project Overview

The project implements two Machine Learning approaches:

| Algorithm                    | Description                                                                         |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| KNN (K-Nearest Neighbors)    | Finds movies similar to the user's preferences using cosine distance.               |
| SVM (Support Vector Machine) | Classifies movies as recommended or not recommended using IMDb rating-based labels. |

The project demonstrates data preprocessing, feature engineering, similarity-based recommendation, supervised classification, and interactive widgets in Python.

## 🎯 Objectives

* Build a movie recommendation system using Machine Learning.
* Recommend movies based on user preferences.
* Apply KNN to find similar movies using cosine distance.
* Apply SVM to classify movies using IMDb rating thresholds.
* Perform data preprocessing on numerical and categorical features.
* Create an interactive IMDb rating slider using `ipywidgets`.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* colab
* IPyWidgets

## 📂 Project Structure

```text
Movie-Recommendation-System/
│
├── movie_recommendation_dataset_3600.csv
├── KNN_Movie_Recommendation.ipynb
├── SVM_Movie_Recommendation.ipynb
├── README.md
└── requirements.txt
```

## 📊 Dataset

The project uses a movie dataset containing movie details and IMDb ratings.

### Features Used

| Feature        | Description                               |
| -------------- | ----------------------------------------- |
| `title`        | Movie title                               |
| `genre`        | Movie genre                               |
| `language`     | Movie language                            |
| `country`      | Country associated with the movie         |
| `release_year` | Movie release year                        |
| `director`     | Movie director                            |
| `lead_actor`   | Lead actor                                |
| `weather`      | Weather category                          |
| `imdb_rating`  | IMDb rating used for SVM label generation |



## ⚙️ Machine Learning Algorithms

### 1. K-Nearest Neighbors (KNN)

KNN is used to identify movies that are most similar to the user's preferences.

**Working Process:**

1. Load the movie dataset.
2. Select relevant movie features.
3. Handle missing values using `SimpleImputer`.
4. Scale numerical features using `StandardScaler`.
5. Convert categorical features using `OneHotEncoder`.
6. Fit the `NearestNeighbors` model using cosine distance.
7. Accept movie preferences from the user.
8. Find the 5 nearest movies.
9. Display the recommended movie titles.

**Algorithm Configuration:**

```python
NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)
```

KNN is used as a similarity-search method and does not require labeled training data.

### 2. Support Vector Machine (SVM)

SVM is used as a binary classification approach to identify movies predicted as recommended or not recommended.

The project generates binary labels from IMDb ratings based on a user-selected threshold.

* IMDb rating greater than or equal to the threshold → `1` (Recommended)
* IMDb rating below the threshold → `0` (Not Recommended)

**Working Process:**

1. Load the movie dataset.
2. Select the movie attributes.
3. Preprocess the numerical and categorical features.
4. Generate binary recommendation labels using IMDb ratings.
5. Train an SVM classifier using `SVC`.
6. Predict recommendation labels.
7. Display up to 5 predicted recommended movie titles.

**Algorithm Configuration:**

```python
SVC(
    kernel="linear",
    probability=True
)
```

The SVM model learns a decision boundary from the movie features and the generated labels.

**Note:** The SVM notebook uses rating-derived labels and predicts on the same dataset used for training. Therefore, it demonstrates classification rather than a fully personalized recommendation system. Its predictions may differ from the original IMDb threshold labels.

## 🧹 Data Preprocessing

The project applies the following preprocessing techniques:

| Technique                | Purpose                                              |
| ------------------------ | ---------------------------------------------------- |
| Median Imputation        | Handles missing numerical values                     |
| Most-Frequent Imputation | Handles missing categorical values                   |
| StandardScaler           | Standardizes numerical features                      |
| OneHotEncoder            | Converts categorical values into numerical vectors   |
| ColumnTransformer        | Applies different preprocessing to different columns |
| Pipeline                 | Combines preprocessing steps into a workflow         |

## 💻 Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Movie-Recommendation-System.git
```

### Step 2: Navigate to the project folder

```bash
cd Movie-Recommendation-System
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

### Step 5: Run the notebooks

* Open `KNN_Movie_Recommendation.ipynb` to run the KNN recommendation system.
* Open `SVM_Movie_Recommendation.ipynb` to run the SVM classification system.

Make sure the dataset is in the correct location specified in the notebook.

## 📦 Requirements

Create a `requirements.txt` file with the following dependencies:

```text
pandas
numpy
scikit-learn
ipywidgets
notebook
```

## 🖥️ Project Features

* Content-based movie similarity search using KNN.
* SVM-based movie classification.
* Interactive IMDb rating threshold slider.
* Numerical and categorical data preprocessing.
* Dynamic recommendation output in Jupyter Notebook.
* Movie title display for recommended results.

## 🔮 Future Improvements

* Add user ratings and feedback to improve personalization.
* Implement a hybrid recommendation system.
* Include movie posters using a movie database API.
* Add a web interface using Streamlit or Flask.
* Evaluate models using suitable recommendation and classification metrics.
* Improve SVM recommendations using user-specific preference labels.
* Add filters for genre, language, release year, and IMDb rating.

## 🎓 Learning Outcomes

Through this project, I explored:

* Machine Learning algorithms such as KNN and SVM.
* Content-based recommendation techniques.
* Supervised and unsupervised learning concepts.
* Feature preprocessing and transformation.
* Cosine distance and classification.
* Interactive widgets using Python.

## 👩‍💻 Author

**Poojitha Vakada**

B.Tech – Computer Science and Engineering (AI & ML)

GitHub: [YOUR-GITHUB-USERNAME](https://github.com/YOUR-GITHUB-USERNAME)

---

⭐ If you find this project interesting, feel free to star the repository!
