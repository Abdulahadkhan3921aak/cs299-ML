import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split


class LinearRegression:
    def __init__(self, lr=0.001, n_itters=10000, use_gradient_descent=False) -> None:
        self.lr = lr
        self.n_itters = n_itters
        self.use_gradient_descent = use_gradient_descent
        self.weights = None
        self.bias = (
            None  # Keep bias for the line calculation but will fix it to zero later
        )

    def fit(self, X: cp.array, y: cp.array):
        n_samples, n_features = X.shape
        self.weights = cp.zeros(n_features)
        self.bias = cp.zeros(1)  # Initialize bias but will set to zero during updates

        if self.use_gradient_descent:
            plt.ion()
            fig, (ax, ax_loss) = plt.subplots(2, 1, figsize=(10, 8))
            ax.set_xlim(cp.min(X.get()) - 1, cp.max(X.get()) + 1)
            ax.set_ylim(cp.min(y.get()) - 1, cp.max(y.get()) + 1)
            ax_loss.set_ylim(0, 10)  # Adjust as needed

            scatter = ax.scatter(X.get(), y.get(), color="blue", label="Data Points")
            (line,) = ax.plot([], [], color="red", label="Regression Line")
            losses = []

            # Annotate iteration and loss
            annotation = ax.annotate(
                "",
                xy=(0.1, 0.8),
                xycoords="axes fraction",
                fontsize=10,
                color="black",
            )
            for i in range(self.n_itters):
                input()
                # Update weights and bias
                y_predicted = cp.dot(X, self.weights) + self.bias
                dw = (1 / n_samples) * cp.dot(X.T, (y_predicted - y))

                # Update weights
                self.weights -= self.lr * dw

                # Keep bias at 0
                self.bias = 0  # Reset bias to 0 each iteration

                # Create a range for x values
                x_range = cp.linspace(cp.min(X) - 1, cp.max(X) + 1, 100).reshape(
                    -1, 1
                )  # Ensure it's 2D for dot product
                line.set_xdata(x_range.get())  # X-coordinates for the regression line
                line.set_ydata(
                    cp.dot(x_range, self.weights).get()
                )  # Update y-coordinates based on weights

                # Update loss
                loss = self.mean_squared_error(y, y_predicted)
                losses.append(loss.get())
                ax_loss.clear()
                ax_loss.plot(losses, color="purple", label="Loss over Iterations")
                ax_loss.set_title("Loss Function")
                ax_loss.set_xlabel("Iterations")
                ax_loss.set_ylabel("Mean Squared Error")
                ax_loss.legend()

                # Update scatter points for predictions
                # Remove existing scatter points
                for collection in ax.collections:
                    collection.remove()  # Clear all collections (scatter points)

                ax.scatter(
                    X.get(), y.get(), color="blue", label="Data Points", alpha=0.8
                )
                # Update the annotation text
                annotation.set_text(f"Iteration: {i + 1}\nLoss: {loss:.4f}")

                plt.draw()
                plt.pause(0.1)
                time.sleep(0.1)

            plt.ioff()
            plt.show()

        else:

            plt.ion()
            fig, (ax, ax_loss) = plt.subplots(2, 1, figsize=(10, 8))
            ax.set_xlim(cp.min(X.get()) - 1, cp.max(X.get()) + 1)
            ax.set_ylim(cp.min(y.get()) - 1, cp.max(y.get()) + 1)
            ax_loss.set_ylim(0, 10)  # Adjust as needed

            scatter = ax.scatter(X.get(), y.get(), color="blue", label="Data Points")
            (line,) = ax.plot([], [], color="red", label="Regression Line")
            losses = []

            # Annotate iteration and loss
            annotation = ax.annotate(
                "",
                xy=(0.1, 0.8),
                xycoords="axes fraction",
                fontsize=10,
                color="black",
            )

            # Update weights
            self.weights = cp.linalg.inv(X.T @ X) @ (X.T @ y)

            # Update weights and bias
            y_predicted = cp.dot(X, self.weights) + self.bias
            # Keep bias at 0
            self.bias = 0  # Reset bias to 0 each iteration

            # Create a range for x values
            x_range = cp.linspace(cp.min(X) - 1, cp.max(X) + 1, 100).reshape(
                -1, 1
            )  # Ensure it's 2D for dot product
            line.set_xdata(x_range.get())  # X-coordinates for the regression line
            line.set_ydata(
                cp.dot(x_range, self.weights).get()
            )  # Update y-coordinates based on weights

            # Update loss
            loss = self.mean_squared_error(y, y_predicted)
            losses.append(loss.get())
            ax_loss.clear()
            ax_loss.plot(losses, color="purple", label="Loss over Iterations")
            ax_loss.set_title("Loss Function")
            ax_loss.set_xlabel("Iterations")
            ax_loss.set_ylabel("Mean Squared Error")
            ax_loss.legend()

            # Update scatter points for predictions
            # Remove existing scatter points
            for collection in ax.collections:
                collection.remove()  # Clear all collections (scatter points)

            ax.scatter(X.get(), y.get(), color="blue", label="Data Points", alpha=0.8)
            # Update the annotation text
            annotation.set_text(f"Iteration: {1}\nLoss: {loss:.4f}")

            plt.draw()
            plt.pause(0.1)
            time.sleep(0.1)

            plt.ioff()
            plt.show()
            # Closed-form solution

    def predict(self, X):
        y_predict = cp.dot(X, self.weights) + self.bias
        return y_predict

    def mean_squared_error(self, y_true, y_pred):
        return cp.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":

    X, y = datasets.make_regression(
        n_samples=256, n_features=1, noise=8, random_state=12
    )
    X = cp.array(X)
    y = cp.array(y)

    model = LinearRegression(lr=0.01, n_itters=300, use_gradient_descent=True)
    model.fit(X, y)
