# Unsupervised Classification of Sentinel-3 Altimetry Echoes

This project performs unsupervised classification of Sentinel-3 radar altimetry echoes to discriminate between two surface types:
- Sea ice
- Leads (open water within sea ice)

## Contents

- [Project Background](#project-background)
- [Data](#data)
- [Methodology](#methodology)
  - [Unsupervised classification (K-means and GMM)](#unsupervised-classification-k-means-and-GMM)
  - [Physical waveform alignment](#physical-waveform-alignment)
- [Results](#results)
  - [Mean echo shapes](#mean-echo-shapes)
  - [ESA validation](#esa-validation)
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)

## Project Background 

### Why Collocation Matters

In Earth Observation, many applications link measurements from different sensors. This requires collocation, meaning matching observations in:

- **Space** (overlapping footprint)
- **Time** (same time or within an acceptable window)

Collocation is non-trivial because sensors differ in:

- Spatial resolution  
- Sampling geometry  
- Revisit time  

For example:

- **Sentinel-2 imagery** provides a 2D fixed grid (Eulerian frame) at ~10 m resolution.  
- **Sentinel-3 altimetry** samples along a 1D moving track (Lagrangian frame).  

This results in a mismatch in spatial dimensions when comparing datasets. Echo classification is therefore commonly combined with collocated datasets (such as satellite imagery) to support training, validation, and interpretation.

[Back to top](#unsupervised-classification-of-sentinel-3-altimetry-echoes)

## Data
The analysis uses Sentinel-3 Level-2 radar altimetry data (NetCDF format), comprising:
- **Radar waveforms** (256-bin echoes per observation)
- **ESA surface type classification** (`surf_type_class_20_ku`)
- **Backscatter coefficient** (Sigma0)
- **Echo stack** information used to derive Stack Standard Deviation (SSD)

More information on Sentinel-3 can be found here [Sentinel-3 mission overview](https://sentinels.copernicus.eu/web/sentinel/copernicus/sentinel-3)

**Binary Subset Selection**
To focus the classification task, the dataset is restricted to observations labelled by ESA as:
- Sea ice (ESA = 1)
- Lead (ESA = 2)
  
This reduction to a binary ice–lead subset simplifies the discrimination problem and should be considered when interpreting classification performance, as other surface types are excluded from the analysis.

[Back to top](#unsupervised-classification-of-sentinel-3-altimetry-echoes)

## Methodology

### Sentinel-3 Data Loading and Feature Construction

Rather than clustering raw 256-dimensional waveforms directly, the notebook constructs a compact, physically interpretable feature space. Each echo is described using three features:

**Pulse Peakiness (PP)** - Measures how sharp or specular the waveform peak is.
- Leads → specular reflection → high peakiness
- Sea ice → rougher scattering → broader return and lower peakiness

**Stack Standard Deviation (SSD)** - Represents variability across the echo stack (multiple viewing angles), reflecting structural variability in surface scattering.

**Sigma0 (Backscatter)** - Represents radar reflectivity intensity.

Each echo becomes a point in a 3-dimensional feature space. Features are standardised prior to clustering.

## Unsupervised classification (K-means and GMM)

Unsupervised classification means no labels are used to train the model. The algorithm identifies structure in the data without being given surface classes.

### K-means Clustering

K-means clustering is an unsupervised learning algorithm that partitions a dataset into a predefined number of clusters, 
𝑘, by grouping data points according to feature similarity (MacQueen, 1967). The method iteratively assigns each observation to the nearest centroid based on squared Euclidean distance and then updates the centroid positions to minimise within-cluster variance. This assignment-update process continues until convergence, typically reaching a local optimum. K-means is computationally efficient, straightforward to implement, and well suited to exploratory analysis when the underlying data structure is unknown. However, it assumes relatively simple cluster geometry and produces hard class assignments, which may limit flexibility for complex geophysical feature distributions.

Below is a basic code implementation for a K-means Clustering Model.

```
from matplotlib.colors import ListedColormap

# Sample data
X = np.random.rand(100, 2)

# K-means model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


# Plotting
cmap_discrete = ListedColormap(plt.cm.RdPu(np.linspace(0.4, 1.0, 4)))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap=cmap_discrete)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.6)
plt.title("K-means clustering")
plt.savefig("kmeans_model.png", dpi=300, bbox_inches="tight")
plt.show()
```

### Gaussian Mixture Models (GMMs)

Gaussian Mixture Models (GMMs) are probabilistic clustering methods that represent a dataset as a mixture of Gaussian distributions, each defined by its own mean, covariance, and mixing coefficient (Reynolds et al., 2009). Model parameters are estimated using the Expectation–Maximization (EM) algorithm, which iteratively alternates between estimating the probability of cluster membership (E-step) and maximising the likelihood of the data given those assignments (M-step). Unlike K-means, GMM produces soft probabilistic classifications and allows clusters to adopt elliptical shapes through flexible covariance modelling. This makes GMM particularly suitable for geophysical datasets where feature distributions may overlap or exhibit non-spherical structure, providing a more adaptable framework for unsupervised classification.

Below is a basic code implementation for a GMM Model.

```
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# GMM model
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_gmm = gmm.predict(X)

# Plotting
cmap_discrete = ListedColormap(plt.cm.RdPu(np.linspace(0.4, 1.0, 4)))
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap=cmap_discrete)
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Gaussian Mixture Model')
plt.savefig("GMM_model.png", dpi=300, bbox_inches="tight")
plt.show()
```




