
# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# 2. Load Dataset

df = pd.read_csv("student_dropout_behavior_dataset.csv")

print("=" * 50)
print("STUDENT DROPOUT BEHAVIOR ANALYSIS")
print("=" * 50)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())


# 3. Exploratory Data Analysis (EDA)

plt.figure(figsize=(8, 5))

plt.scatter(
    df["previous_gpa"],
    df["final_marks"],
    s=60
)

plt.xlabel("Previous GPA")
plt.ylabel("Final Marks")
plt.title("Previous GPA vs Final Marks")
plt.tight_layout()
plt.show()


# 4. Select Features

features = [
    "quiz1_marks",
    "quiz2_marks",
    "quiz3_marks",
    "total_assignments",
    "midterm_marks",
    "final_marks",
    "previous_gpa",
    "lectures_attended",
    "labs_attended"
]

X = df[features].copy()

# Remove columns that contain only missing values

empty_columns = X.columns[X.isnull().all()].tolist()

if empty_columns:
    print("\nCompletely empty columns removed:", empty_columns)
    X = X.drop(columns=empty_columns)

# Replace infinite values with missing values

X = X.replace([np.inf, -np.inf], np.nan)

# Remove rows that have no usable feature values

X = X.dropna(how="all")

# Fill remaining missing values using column medians

X = X.fillna(X.median())

# Keep the original student records matching X

df_clean = df.loc[X.index].copy()

features = X.columns.tolist()

print("\nSelected Features:")
print(features)

print("\nFeature Data Shape:")
print(X.shape)


# 5. Standardize Data

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Data Shape:")
print(X_scaled.shape)


# 6. Apply PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA Shape:")
print(X_pca.shape)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Variance Explained:")
print(pca.explained_variance_ratio_.sum())


# 7. Visualize PCA

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    s=60
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - Student Data")
plt.tight_layout()
plt.show()


# 8. Elbow Method

inertia = []

k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(list(k_values))
plt.tight_layout()
plt.show()


# 9. Compare Silhouette Scores

results = []

for k in range(2, 7):
    kmeans_test = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans_test.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    results.append({
        "K": k,
        "Silhouette Score": score
    })

results_df = pd.DataFrame(results)

print("\nSilhouette Scores for Different K Values:")
print(results_df.to_string(index=False))

best_k = int(
    results_df.loc[
        results_df["Silhouette Score"].idxmax(),
        "K"
    ]
)

best_score = results_df["Silhouette Score"].max()

print("\nBest K:", best_k)
print("Best Silhouette Score:", best_score)


# 10. Final K-Means Model (K = 5)

final_kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

final_clusters = final_kmeans.fit_predict(X_scaled)

df_final = df_clean.copy()

df_final["Cluster"] = final_clusters

print("\nFinal Cluster Counts:")
print(
    df_final["Cluster"]
    .value_counts()
    .sort_index()
)


# 11. Final Cluster Summary

final_summary = df_final.groupby("Cluster")[features].mean()

performance_features = [
    col for col in [
        "quiz1_marks",
        "quiz2_marks",
        "quiz3_marks",
        "midterm_marks",
        "final_marks",
        "previous_gpa"
    ]
    if col in final_summary.columns
]

final_summary["Average_Performance"] = (
    final_summary[performance_features].mean(axis=1)
)

final_summary = final_summary.sort_values(
    "Average_Performance",
    ascending=False
)

print("\nFinal Cluster Summary:")
print(final_summary)


# 12. Visualize Final Clusters using PCA

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=final_clusters,
    s=60
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    "Final Student Segmentation using K-Means (K=5)"
)

plt.colorbar(scatter, label="Cluster")

plt.tight_layout()
plt.show()


# 13. Final Silhouette Score

final_silhouette = silhouette_score(
    X_scaled,
    final_clusters
)

print("\nFinal K:", 5)
print("Final Silhouette Score:", final_silhouette)


# 14. Save Final Results

df_final.to_csv(
    "student_dropout_clusters.csv",
    index=False
)

final_summary.to_csv("cluster_summary.csv")

print("\nFiles Saved Successfully:")
print("student_dropout_clusters.csv")
print("cluster_summary.csv")

print("\nProject Completed Successfully!")
