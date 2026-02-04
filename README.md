# Custom ML Framework

A from-scratch implementation of some core machine learning algorithms designed to connect conceptual understanding with practical application. Numpy and Scipy were used exclusively for all data storage, manipulation, and calculations to maximize efficiency and readability.

## Usage

```
from preprocessing.standard import StandardScaler
from loss.mse import MSE
from models.linear_regression import LinearRegression

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression(loss_fn=MSE(), alpha=0.01)
model.fit(X_train_scaled, y_train, epochs=500, batch_size=32)

mse = model.evaluate(X_test_scaled, y_test)
print(f"MSE on test set: {mse:.4f}")
```

## Algorithms

### Linear Regression and MSE Loss
* Implemented vectorized Mini-Batch Gradient Descent ($\hat{y} = Xw + b$) with adjustable parameters for epochs, batch size, and learning rate.
* Utilized matrix-based partial derivatives for weight updates, avoiding iterative summation loops for O(n) performance gains.
* Validated accuracy against Scikit-Learn's LinearRegression implementation, achieving <0.001% variance in predicted weights.

### KNN (K-Nearest-Neighbors)
* Implemented inverse distance (supports both Euclidean and Manhattan) weighting to prioritize closer neighbors over farther ones.
* Utilized scipy.spatial for distance calculations to optimize prediction speed considering standard KNN's O(m•n) inference time.
* Achieved 96.67% accuracy (k=5, Euclidean distance) on MNIST handwritten digit dataset (preprocessed by MinMax scaler), comparable to industry-standard libraries (Scikit-Learn baseline ~97%).

### PCA (Principle Component Analysis)
* Supports both exact Full SVD (Singular Vector Decomposition) and Randomized Truncated SVD for flexible compute-accuracy trade-offs.
* Implemented cumulative explained variance logic, allowing for dimensionality selection based on the desired information retention threshold
* Supports data reconstruction applying the reduced dimensionality, enabling visualization of feature retention
* Mathematical foundation: 
  * Objective: Maximize Var(Xw), the projection of the data onto the weight line, subject to $||w||^2$, meaning the weight line must remain a unit vector
  * Calculation: SVD($X$) = $U\Sigma V^T$, where columns of V are the principal components

### Scalers (MinMax and Standard)
* StandardScaler: Centers data at $\mu = 0$ with $\sigma = 1$. Essentials for PCA and Gradient Descent convergence.
* MinMaxScaler: Normalizes features between 0 and 1, preserving relative distances between data points. Essential for KNN performance.

## Engineering Principles
* **Data Validation:** Implemented custom DataValidator utility to create clear, reusable condition checkers and standardized error handling
* **Modular Framework:** Utilized base class architecture to standardize methods (i.e. fit(), predict(), evaluate()) and facilitate future additions