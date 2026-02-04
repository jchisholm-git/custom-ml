import numpy as np
from keras.datasets import mnist, boston_housing
from sklearn.model_selection import train_test_split

from loss.mse import MSE
from models.linear_regression import LinearRegression
from models.knn import KNN
from preprocessing.standard import StandardScaler
from preprocessing.min_max import MinMaxScaler
from preprocessing.pca import PCA
from utils.data_helpers import flatten_features


def test_linear_regression_boston():
    print("Linear Regression || Boston Housing")
    scaler = StandardScaler()

    (X_train, y_train), (X_test, y_test) = boston_housing.load_data()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression(loss_fn=MSE(), alpha=0.01)
    model.fit(X_train, y_train, epochs=500, batch_size=32)
    mse = model.evaluate(X_test, y_test)

    print(f"MSE on test set: {mse:.4f}")
    assert mse > 0


def test_knn_mnist():
    print("\nKNN || MNIST (subset)")

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = flatten_features(X_train)
    X_test = flatten_features(X_test)

    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, stratify=y_train, random_state=42
    )
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNN(k=5, d="euclidean")
    knn.fit(X_train, y_train)
    accuracy = knn.evaluate(X_test, y_test)

    print(f"KNN accuracy: {accuracy:.2f}%")
    assert accuracy > 80


def test_pca_knn_pipeline():
    print("\nPCA + KNN || MNIST")

    (X_train, y_train), _ = mnist.load_data()
    X_train = flatten_features(X_train[:2000])
    y_train = y_train[:2000]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X_train)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_reduced, y_train, test_size=0.2, random_state=42
    )
    knn = KNN(k=3)
    knn.fit(X_tr, y_tr)
    accuracy = knn.evaluate(X_te, y_te)

    print(f"PCA + KNN accuracy: {accuracy:.2f}%")
    assert accuracy > 75


def test_loss_function():
    print("\nMSE Loss Check")

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 1.8, 2.7])

    loss = MSE()
    value = loss(y_true, y_pred)
    grad = loss.gradient(y_true, y_pred)

    print("MSE:", value)
    print("Gradient:", grad)

    assert value > 0
    assert grad.shape == y_true.shape


if __name__ == "__main__":
    np.random.seed(42)

    test_loss_function()
    test_linear_regression_boston()
    test_knn_mnist()
    test_pca_knn_pipeline()

    print("\nAll demo tests completed successfully")
