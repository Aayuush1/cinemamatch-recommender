# CinemaMatch: Step-by-Step Implementation Guide

**Complete Tutorial for Building SVD-Powered Movie Recommender**

---

## 📚 Table of Contents

1. [Getting Started](#1-getting-started)
2. [Understanding the Mathematics](#2-understanding-the-mathematics)
3. [Data Generation](#3-data-generation)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Matrix Construction](#5-matrix-construction)
6. [SVD Decomposition](#6-svd-decomposition)
7. [Making Predictions](#7-making-predictions)
8. [Evaluation](#8-evaluation)
9. [Generating Recommendations](#9-generating-recommendations)
10. [Visualizations](#10-visualizations)
11. [Next Steps](#11-next-steps)

---

## 1. Getting Started

### 1.1 Prerequisites

**Required Knowledge:**
- Basic Python programming
- Linear algebra (matrices, vectors, dot products)
- Statistics fundamentals (mean, variance, correlation)
- NumPy basics

**Software Requirements:**
```bash
Python 3.8 or higher
```

### 1.2 Environment Setup

**Step 1: Create project directory**
```bash
mkdir cinemamatch-project
cd cinemamatch-project
```

**Step 2: Create virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Step 3: Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

**Step 4: Verify installation**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import svds

print("All imports successful!")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
```

### 1.3 Project Structure

```
cinemamatch-project/
│
├── cinemamatch_recommender.py    # Main implementation
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies
│
└── outputs/                       # Generated visualizations
    ├── eda_analysis.png
    ├── model_evaluation.png
    ├── latent_space.png
    └── algorithm_comparison.png
```

---

## 2. Understanding the Mathematics

### 2.1 What is SVD?

**Intuitive Explanation:**

Imagine you have a table of movie ratings:
- Rows = Users
- Columns = Movies
- Cells = Ratings (1-5 stars)

This table is mostly empty (sparse) because users haven't rated all movies.

**SVD magic:** Decomposes this messy table into three clean matrices:

```
R (users × movies) ≈ U (users × factors) × Σ (factors) × V^T (factors × movies)
```

**What each matrix represents:**

1. **U (User Matrix):**
   - Each row = one user's "preferences" across hidden factors
   - Factor might be: "likes action", "prefers depth", "enjoys humor"
   - Values indicate how much each factor matters to this user

2. **Σ (Singular Values):**
   - Diagonal matrix of importance weights
   - Larger values = more important factors
   - We only keep top k factors (dimensionality reduction!)

3. **V^T (Movie Matrix):**
   - Each column = one movie's "characteristics" across factors
   - Same factors as U: "action level", "depth", "humor"
   - Values indicate how much each factor applies to this movie

**Example with k=3 factors:**

```
User 1: [0.8, 0.2, 0.1]  → Loves action, neutral on depth, doesn't care about humor
Movie 5: [0.9, 0.1, 0.0] → High action, low depth, no humor
Predicted rating: 0.8×0.9 + 0.2×0.1 + 0.1×0.0 = 0.74 (then scale to 1-5)
```

### 2.2 Why SVD Works

**Key Insight:** Most rating patterns can be explained by a few hidden factors.

**Mathematical Property:**
SVD gives the **best** low-rank approximation (Eckart-Young theorem).

**Translation:** Among all ways to compress the data, SVD minimizes information loss!

### 2.3 Complexity Analysis

**Time Complexity:**
- Full SVD: O(m²n) where m=users, n=movies
- Truncated SVD: O(kmn) where k=factors

**Space Complexity:**
- Original: O(mn)
- Factorized: O(k(m+n))

**For our case (m=100, n=50, k=20):**
- Original: 5,000 values
- Factorized: 20×(100+50) = 3,000 values
- **Savings: 40%**

---

## 3. Data Generation

### 3.1 Why Synthetic Data?

**Advantages:**
- Known ground truth (we control the patterns)
- Reproducible (same seed = same data)
- Fast iteration (no download/cleaning)
- Controlled complexity (3 latent factors)

**Real-world alternative:** MovieLens dataset (will discuss in extensions)

### 3.2 Generation Strategy

**Step-by-step process:**

```python
import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Step 1: Define parameters
n_users = 100
n_movies = 50
n_latent_factors = 3  # True underlying factors
sparsity = 0.7        # 70% missing ratings

# Step 2: Create latent factors
# Each user has preferences across 3 dimensions
user_factors = np.random.randn(n_users, n_latent_factors)

# Each movie has characteristics across same 3 dimensions
movie_factors = np.random.randn(n_movies, n_latent_factors)

# Step 3: Generate "true" ratings via matrix multiplication
# Rating = how well user preferences match movie characteristics
true_ratings = np.dot(user_factors, movie_factors.T)

# Step 4: Scale to [1, 5] range
# Standardize to mean=0, std=1
mean_rating = true_ratings.mean()
std_rating = true_ratings.std()
true_ratings_standardized = (true_ratings - mean_rating) / std_rating

# Scale to [1, 5] with mean=3
true_ratings_scaled = 3 + 2 * true_ratings_standardized
true_ratings_scaled = np.clip(true_ratings_scaled, 1, 5)

# Step 5: Add user biases
# Some users rate consistently higher/lower
user_bias = np.random.randn(n_users, 1) * 0.5
true_ratings_scaled += user_bias
true_ratings_scaled = np.clip(true_ratings_scaled, 1, 5)

# Step 6: Round to realistic ratings (0.5 increments)
true_ratings_scaled = np.round(true_ratings_scaled * 2) / 2

# Step 7: Apply sparsity (randomly remove 70% of ratings)
mask = np.random.random((n_users, n_movies)) > sparsity

# Step 8: Convert to long format (userId, movieId, rating)
ratings_list = []
for i in range(n_users):
    for j in range(n_movies):
        if mask[i, j]:  # Only include observed ratings
            ratings_list.append({
                'userId': i + 1,      # 1-indexed
                'movieId': j + 1,     # 1-indexed
                'rating': true_ratings_scaled[i, j]
            })

ratings_df = pd.DataFrame(ratings_list)

print(f"Generated {len(ratings_df)} ratings")
print(f"Actual sparsity: {1 - len(ratings_df)/(n_users*n_movies):.1%}")
print(f"Average rating: {ratings_df['rating'].mean():.2f}")
```

**Expected output:**
```
Generated 1,461 ratings
Actual sparsity: 70.8%
Average rating: 3.01
```

### 3.3 Creating Movie Metadata

```python
# Create movie information
genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Romance', 'Horror', 'Thriller']

movies_data = {
    'movieId': range(1, n_movies + 1),
    'title': [f"Movie_{i}" for i in range(1, n_movies + 1)],
    'genre': np.random.choice(genres, n_movies),
    'year': np.random.randint(1990, 2024, n_movies)
}

movies_df = pd.DataFrame(movies_data)
print(movies_df.head())
```

**Expected output:**
```
   movieId    title   genre  year
0        1  Movie_1   Drama  2005
1        2  Movie_2  Sci-Fi  2012
2        3  Movie_3  Action  1998
3        4  Movie_4  Comedy  2019
4        5  Movie_5   Drama  2001
```

---

## 4. Exploratory Data Analysis

### 4.1 Basic Statistics

```python
print("Dataset Overview:")
print(f"Total ratings: {len(ratings_df):,}")
print(f"Unique users: {ratings_df['userId'].nunique()}")
print(f"Unique movies: {ratings_df['movieId'].nunique()}")
print(f"Rating range: {ratings_df['rating'].min():.1f} to {ratings_df['rating'].max():.1f}")

print("\nRating Statistics:")
print(f"Mean: {ratings_df['rating'].mean():.3f}")
print(f"Median: {ratings_df['rating'].median():.3f}")
print(f"Std Dev: {ratings_df['rating'].std():.3f}")
```

### 4.2 Distribution Analysis

```python
from scipy import stats

# Compute skewness and kurtosis
skewness = stats.skew(ratings_df['rating'])
kurtosis = stats.kurtosis(ratings_df['rating'])

print(f"Skewness: {skewness:.3f}")
print(f"Kurtosis: {kurtosis:.3f}")

# Interpretation
if skewness > 0:
    print("→ Right-skewed (more low ratings)")
elif skewness < 0:
    print("→ Left-skewed (more high ratings)")
else:
    print("→ Symmetric distribution")

if kurtosis > 0:
    print("→ Leptokurtic (heavy tails, outliers)")
elif kurtosis < 0:
    print("→ Platykurtic (light tails, no outliers)")
else:
    print("→ Normal distribution")
```

### 4.3 User and Movie Analysis

```python
# User activity
user_activity = ratings_df.groupby('userId').size()
print(f"\nUser Activity:")
print(f"Mean ratings per user: {user_activity.mean():.1f}")
print(f"Most active user: {user_activity.max()} ratings")
print(f"Least active user: {user_activity.min()} ratings")

# Movie popularity
movie_popularity = ratings_df.groupby('movieId').size()
print(f"\nMovie Popularity:")
print(f"Mean ratings per movie: {movie_popularity.mean():.1f}")
print(f"Most popular movie: {movie_popularity.max()} ratings")
print(f"Least popular movie: {movie_popularity.min()} ratings")
```

### 4.4 Statistical Tests

```python
# Test for normality
stat, p_value = stats.normaltest(ratings_df['rating'])
print(f"\nNormality Test:")
print(f"Statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("→ Ratings are NOT normally distributed (reject H0)")
else:
    print("→ Ratings appear normally distributed (fail to reject H0)")
```

### 4.5 Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Rating distribution
axes[0, 0].hist(ratings_df['rating'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(ratings_df['rating'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {ratings_df["rating"].mean():.2f}')
axes[0, 0].set_xlabel('Rating')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Rating Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. User activity
axes[0, 1].hist(user_activity, bins=20, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_xlabel('Number of Ratings')
axes[0, 1].set_ylabel('Number of Users')
axes[0, 1].set_title('User Activity Distribution')
axes[0, 1].grid(True, alpha=0.3)

# 3. Movie popularity
axes[1, 0].hist(movie_popularity, bins=20, edgecolor='black', alpha=0.7, color='orange')
axes[1, 0].set_xlabel('Number of Ratings')
axes[1, 0].set_ylabel('Number of Movies')
axes[1, 0].set_title('Movie Popularity Distribution')
axes[1, 0].grid(True, alpha=0.3)

# 4. Box plot
axes[1, 1].boxplot(ratings_df['rating'], vert=True)
axes[1, 1].set_ylabel('Rating')
axes[1, 1].set_title('Rating Box Plot')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization to 'eda_analysis.png'")
```

---

## 5. Matrix Construction

### 5.1 Creating the Utility Matrix

**Step 1: Pivot to wide format**

```python
# Convert from long format (userId, movieId, rating) 
# to wide format (users × movies matrix)

utility_matrix = ratings_df.pivot(
    index='userId',
    columns='movieId',
    values='rating'
)

print("Utility Matrix Shape:", utility_matrix.shape)
print(f"Sparsity: {utility_matrix.isna().sum().sum() / utility_matrix.size:.1%}")

# View sample
print("\nFirst 5 users × first 5 movies:")
print(utility_matrix.iloc[:5, :5])
```

**Expected output:**
```
Utility Matrix Shape: (100, 50)
Sparsity: 70.8%

First 5 users × first 5 movies:
movieId     1    2    3    4    5
userId                           
1         NaN  3.5  NaN  NaN  2.0
2         4.0  NaN  NaN  3.5  NaN
3         NaN  NaN  4.5  NaN  3.0
4         2.5  NaN  NaN  NaN  NaN
5         NaN  4.0  NaN  3.0  NaN
```

### 5.2 Mean Centering (Removing User Bias)

**Why?** Some users rate everything 4-5 stars, others rate 1-2 stars. We want to capture *relative* preferences.

```python
# Calculate each user's average rating
user_means = utility_matrix.mean(axis=1)

print("User Rating Means (first 10 users):")
print(user_means.head(10))

# Subtract user mean from each rating (de-meaning)
utility_matrix_demeaned = utility_matrix.sub(user_means, axis=0)

print("\nDemeaned Matrix (first 5 users × first 5 movies):")
print(utility_matrix_demeaned.iloc[:5, :5])
```

**Expected output:**
```
User Rating Means (first 10 users):
userId
1    3.12
2    3.45
3    2.89
...

Demeaned Matrix (first 5 users × first 5 movies):
movieId      1     2     3     4     5
userId                                
1          NaN  0.38  NaN  NaN -1.12
2         0.55  NaN  NaN  0.05  NaN
3          NaN  NaN  1.61  NaN  0.11
...
```

### 5.3 Handling Missing Values

```python
# For SVD, we need a complete matrix (no NaN)
# Fill NaN with 0 (neutral value after mean-centering)

utility_matrix_filled = utility_matrix_demeaned.fillna(0).values

print("Filled Matrix Shape:", utility_matrix_filled.shape)
print("Contains NaN?:", np.isnan(utility_matrix_filled).any())

# Create mask to remember which entries were observed
sparsity_mask = (~utility_matrix.isna()).values.astype(int)

print("Sparsity Mask Shape:", sparsity_mask.shape)
print("Number of observed ratings:", sparsity_mask.sum())
```

---

## 6. SVD Decomposition

### 6.1 Computing Truncated SVD

```python
from scipy.sparse.linalg import svds

# Specify number of latent factors
k = 20

print(f"Computing SVD with k={k} factors...")

# Perform truncated SVD
# Returns: U (m×k), sigma (k,), Vt (k×n)
U, sigma, Vt = svds(utility_matrix_filled, k=k)

print("✓ SVD computation complete")
print(f"U shape: {U.shape}")      # (100, 20)
print(f"sigma shape: {sigma.shape}")  # (20,)
print(f"Vt shape: {Vt.shape}")    # (20, 50)
```

### 6.2 Understanding the Decomposition

```python
# svds returns singular values in ASCENDING order
# Reverse them to get descending order
U = U[:, ::-1]
sigma = sigma[::-1]
Vt = Vt[::-1, :]

print("\nTop 5 Singular Values:")
for i, s in enumerate(sigma[:5], 1):
    print(f"σ_{i} = {s:.4f}")

# Compute variance explained
total_variance = np.sum(sigma**2)
explained_variance_ratio = sigma**2 / total_variance
cumulative_variance = np.cumsum(explained_variance_ratio)

print("\nVariance Explained:")
print(f"Factor 1: {explained_variance_ratio[0]:.1%}")
print(f"Top 5 factors: {cumulative_variance[4]:.1%}")
print(f"Top 10 factors: {cumulative_variance[9]:.1%}")
print(f"All {k} factors: {cumulative_variance[-1]:.1%}")
```

### 6.3 Interpreting the Matrices

**U Matrix (Users × Factors):**
```python
print("\nUser 1's latent preferences (first 5 factors):")
print(U[0, :5])

# Interpretation:
# Positive values = user likes this factor
# Negative values = user dislikes this factor
# Large magnitude = strong preference
```

**V^T Matrix (Factors × Movies):**
```python
print("\nMovie 1's latent characteristics (first 5 factors):")
print(Vt[:5, 0])

# Interpretation:
# Positive values = movie has this factor
# Negative values = movie lacks this factor
# Large magnitude = strong presence/absence
```

---

## 7. Making Predictions

### 7.1 Reconstructing the Rating Matrix

```python
# Prediction formula: R̂ = U Σ V^T
# First, create diagonal matrix from sigma
sigma_diag = np.diag(sigma)

# Matrix multiplication
predictions_demeaned = np.dot(np.dot(U, sigma_diag), Vt)

print("Predictions (demeaned) shape:", predictions_demeaned.shape)

# Add back user means
predictions = predictions_demeaned + user_means.values.reshape(-1, 1)

# Clip to valid rating range [1, 5]
predictions = np.clip(predictions, 1, 5)

print("Final predictions shape:", predictions.shape)
print(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
```

### 7.2 Example Predictions

```python
# Compare actual vs predicted for User 1, Movie 1
user_idx = 0  # User 1 (0-indexed)
movie_idx = 0  # Movie 1 (0-indexed)

actual_rating = utility_matrix.iloc[user_idx, movie_idx]
predicted_rating = predictions[user_idx, movie_idx]

print(f"\nUser 1, Movie 1:")
print(f"Actual rating: {actual_rating}")
print(f"Predicted rating: {predicted_rating:.2f}")

if pd.isna(actual_rating):
    print("(This was a missing rating - now we have a prediction!)")
```

---

## 8. Evaluation

### 8.1 Computing Metrics

```python
# Get actual ratings (only observed entries)
actual_ratings = utility_matrix.values[sparsity_mask == 1]
predicted_ratings = predictions[sparsity_mask == 1]

# 1. Root Mean Squared Error (RMSE)
rmse = np.sqrt(np.mean((actual_ratings - predicted_ratings)**2))

# 2. Mean Absolute Error (MAE)
mae = np.mean(np.abs(actual_ratings - predicted_ratings))

# 3. R-squared
ss_residual = np.sum((actual_ratings - predicted_ratings)**2)
ss_total = np.sum((actual_ratings - np.mean(actual_ratings))**2)
r_squared = 1 - (ss_residual / ss_total)

# 4. Correlation
correlation = np.corrcoef(actual_ratings, predicted_ratings)[0, 1]

print("Model Performance:")
print(f"RMSE:  {rmse:.4f} (lower is better)")
print(f"MAE:   {mae:.4f} (lower is better)")
print(f"R²:    {r_squared:.4f} (higher is better, max=1.0)")
print(f"Correlation: {correlation:.4f} (higher is better, max=1.0)")
```

### 8.2 Statistical Validation

```python
from scipy import stats

# Test for prediction bias
# H0: mean(error) = 0 (unbiased predictions)
errors = actual_ratings - predicted_ratings

t_stat, p_value = stats.ttest_1samp(errors, 0)

print("\nBias Test:")
print(f"Mean error: {errors.mean():.4f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("→ Predictions are BIASED (reject H0)")
else:
    print("→ Predictions are UNBIASED (fail to reject H0) ✓")
```

### 8.3 Visualization

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Predicted vs Actual scatter plot
axes[0].scatter(actual_ratings, predicted_ratings, alpha=0.3, s=20)
axes[0].plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Rating')
axes[0].set_ylabel('Predicted Rating')
axes[0].set_title(f'Predictions vs Actuals (R² = {r_squared:.3f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Error distribution
axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[1].axvline(errors.mean(), color='green', linestyle='--', 
                linewidth=2, label=f'Mean: {errors.mean():.3f}')
axes[1].set_xlabel('Prediction Error')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Error Distribution (RMSE = {rmse:.3f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization to 'model_evaluation.png'")
```

---

## 9. Generating Recommendations

### 9.1 Top-N Recommendations for a User

```python
def get_recommendations(user_id, predictions_matrix, movies_df, 
                       utility_matrix, n_recommendations=10):
    """
    Generate top-N movie recommendations for a user
    
    Parameters:
    -----------
    user_id : int (1-indexed)
    predictions_matrix : np.ndarray
    movies_df : pd.DataFrame
    utility_matrix : pd.DataFrame
    n_recommendations : int
    
    Returns:
    --------
    recommendations : pd.DataFrame
    """
    
    # Get user's predictions (convert to 0-indexed)
    user_predictions = predictions_matrix[user_id - 1, :]
    
    # Get movies user has already rated
    already_rated = utility_matrix.loc[user_id].dropna().index.tolist()
    
    # Create recommendations dataframe
    recommendations = pd.DataFrame({
        'movieId': range(1, len(user_predictions) + 1),
        'predicted_rating': user_predictions
    })
    
    # Merge with movie metadata
    recommendations = recommendations.merge(movies_df, on='movieId')
    
    # Filter out already-rated movies
    recommendations = recommendations[~recommendations['movieId'].isin(already_rated)]
    
    # Sort by predicted rating (descending)
    recommendations = recommendations.sort_values('predicted_rating', ascending=False)
    
    # Return top N
    return recommendations.head(n_recommendations)

# Example: Get recommendations for User 1
user_id = 1
recommendations = get_recommendations(
    user_id=user_id,
    predictions_matrix=predictions,
    movies_df=movies_df,
    utility_matrix=utility_matrix,
    n_recommendations=10
)

print(f"\nTop 10 Recommendations for User {user_id}:")
print(recommendations[['title', 'genre', 'predicted_rating']])
```

### 9.2 Explaining Recommendations

```python
def explain_recommendation(user_id, movie_id, U, sigma, Vt, k=3):
    """
    Explain why a movie was recommended using top factors
    
    Parameters:
    -----------
    user_id : int (1-indexed)
    movie_id : int (1-indexed)
    U : np.ndarray (user factors)
    sigma : np.ndarray (singular values)
    Vt : np.ndarray (movie factors)
    k : int (number of top factors to show)
    """
    
    # Get user and movie vectors
    user_vector = U[user_id - 1, :]
    movie_vector = Vt[:, movie_id - 1]
    
    # Compute contribution of each factor
    contributions = user_vector * sigma * movie_vector
    
    # Get top k factors
    top_factors_idx = np.argsort(np.abs(contributions))[::-1][:k]
    
    print(f"\nWhy User {user_id} might like Movie {movie_id}:")
    for i, factor_idx in enumerate(top_factors_idx, 1):
        contrib = contributions[factor_idx]
        user_pref = user_vector[factor_idx]
        movie_char = movie_vector[factor_idx]
        
        print(f"\n{i}. Factor {factor_idx + 1} (importance: {sigma[factor_idx]:.2f}):")
        print(f"   User preference: {user_pref:+.3f}")
        print(f"   Movie characteristic: {movie_char:+.3f}")
        print(f"   Contribution to rating: {contrib:+.3f}")

# Example
explain_recommendation(user_id=1, movie_id=5, U=U, sigma=sigma, Vt=Vt, k=3)
```

---

## 10. Visualizations

### 10.1 Latent Space Visualization

```python
def visualize_latent_space(Vt, movies_df, n_movies_to_show=30):
    """
    Visualize movies in 2D latent feature space
    """
    
    # Get first 2 latent factors for movies
    movie_features_2d = Vt[:2, :].T
    
    # Select random subset of movies
    n_movies_total = movie_features_2d.shape[0]
    indices = np.random.choice(n_movies_total, 
                              min(n_movies_to_show, n_movies_total), 
                              replace=False)
    
    plt.figure(figsize=(12, 8))
    
    # Plot movies
    scatter = plt.scatter(
        movie_features_2d[indices, 0],
        movie_features_2d[indices, 1],
        s=200,
        alpha=0.6,
        c=range(len(indices)),
        cmap='viridis',
        edgecolors='black',
        linewidth=1.5
    )
    
    # Annotate with movie IDs
    for idx in indices:
        plt.annotate(
            f'M{idx+1}',
            (movie_features_2d[idx, 0], movie_features_2d[idx, 1]),
            fontsize=8,
            ha='center',
            va='center',
            fontweight='bold'
        )
    
    plt.xlabel('Latent Factor 1', fontsize=12, fontweight='bold')
    plt.ylabel('Latent Factor 2', fontsize=12, fontweight='bold')
    plt.title('Movies in Latent Space\n(Closer = More Similar)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    plt.colorbar(scatter, label='Movie Index')
    
    plt.tight_layout()
    plt.savefig('latent_space.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved visualization to 'latent_space.png'")

visualize_latent_space(Vt, movies_df, n_movies_to_show=30)
```

### 10.2 Singular Value Spectrum

```python
plt.figure(figsize=(10, 6))

plt.plot(range(1, len(sigma) + 1), sigma, 'o-', linewidth=2, markersize=8)
plt.xlabel('Factor Index', fontsize=12)
plt.ylabel('Singular Value', fontsize=12)
plt.title('Singular Value Spectrum', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale for better visibility

plt.tight_layout()
plt.savefig('singular_values.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to 'singular_values.png'")
```

---

## 11. Next Steps

### 11.1 Extensions to Try

**1. Experiment with k (number of factors):**
```python
k_values = [5, 10, 15, 20, 25, 30]
rmse_values = []

for k in k_values:
    U, sigma, Vt = svds(utility_matrix_filled, k=k)
    # ... (repeat prediction and evaluation)
    rmse_values.append(rmse)

# Plot RMSE vs k
plt.plot(k_values, rmse_values, 'o-')
plt.xlabel('Number of Factors (k)')
plt.ylabel('RMSE')
plt.title('Model Performance vs Complexity')
plt.show()
```

**2. Try real-world data (MovieLens):**
```python
# Download MovieLens dataset
# https://grouplens.org/datasets/movielens/

import pandas as pd

# Load ratings
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Load movies
movies = pd.read_csv('ml-latest-small/movies.csv')

# Use same pipeline on real data!
```

**3. Implement cross-validation:**
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in kf.split(ratings_df):
    train_df = ratings_df.iloc[train_idx]
    test_df = ratings_df.iloc[test_idx]
    
    # Train on train_df, evaluate on test_df
    # ... (repeat pipeline)
```

**4. Add regularization:**
```python
# Instead of plain SVD, use regularized matrix factorization
# Prevents overfitting on sparse data

lambda_reg = 0.1  # Regularization strength

# Objective: minimize ||R - UV^T||² + λ(||U||² + ||V||²)
# Requires iterative optimization (Alternating Least Squares)
```

**5. Build an API:**
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json['user_id']
    n_recommendations = request.json.get('n', 10)
    
    recommendations = get_recommendations(
        user_id, predictions, movies_df, 
        utility_matrix, n_recommendations
    )
    
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
```

### 11.2 Learning Resources

**Books:**
1. *Introduction to Linear Algebra* by Gilbert Strang
2. *Recommender Systems: The Textbook* by Charu Aggarwal
3. *Machine Learning* by Tom Mitchell

**Online Courses:**
1. Coursera: *Machine Learning* by Andrew Ng
2. Fast.ai: *Practical Deep Learning*
3. MIT OpenCourseWare: *Linear Algebra*

**Papers:**
1. Netflix Prize documentation
2. "Matrix Factorization Techniques for Recommender Systems" (Koren et al.)

**Datasets:**
1. MovieLens (grouplens.org/datasets/movielens/)
2. Netflix Prize (if you can find it)
3. Amazon Product Reviews

### 11.3 Common Issues and Solutions

**Issue 1: SVD fails with NaN values**
```python
# Solution: Fill NaN before SVD
utility_matrix_filled = utility_matrix.fillna(0)
```

**Issue 2: Predictions outside [1, 5] range**
```python
# Solution: Clip predictions
predictions = np.clip(predictions, 1, 5)
```

**Issue 3: Memory error with large datasets**
```python
# Solution: Use sparse matrices
from scipy.sparse import csr_matrix

sparse_matrix = csr_matrix(utility_matrix_filled)
U, sigma, Vt = svds(sparse_matrix, k=20)
```

**Issue 4: Slow SVD computation**
```python
# Solution: Reduce k or use approximate methods
# Try k=10 instead of k=20
# Or use randomized SVD:

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=20, random_state=42)
U = svd.fit_transform(utility_matrix_filled)
sigma = svd.singular_values_
Vt = svd.components_
```

---

## 🎓 Congratulations!

You've successfully built a complete movie recommendation system from scratch using SVD!

**What you've learned:**
✅ Linear algebra in practice (SVD, matrix factorization)
✅ Statistical evaluation (RMSE, R², hypothesis testing)
✅ Data preprocessing (pivoting, mean-centering, imputation)
✅ Algorithm implementation (collaborative filtering)
✅ Visualization and communication

**Portfolio checklist:**
- [ ] Code runs without errors
- [ ] All visualizations generated
- [ ] README.md written
- [ ] Code well-commented
- [ ] Results documented
- [ ] GitHub repository created
- [ ] LinkedIn post drafted

**Next project ideas:**
- Image classifier using CNN
- Sentiment analysis with NLP
- Time series forecasting
- Clustering analysis
- Reinforcement learning game

---

*Happy learning and coding!* 🚀
