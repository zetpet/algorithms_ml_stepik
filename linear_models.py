import numpy as np
import pandas as pd
from random import seed, sample
from typing import Optional, Union, Callable, List


class MyLineReg:
    """MyLineReg is a custom linear regression class.

    :param n_iter: int
    :param learning_rate: float or callable
    :param random_state: int
    :param sgd_sample: None
    :param metric: str or None
    :param reg: str or None
    :param l1_coef: float
    :param l2_coef: float

    """

    def __init__(
            self,
            n_iter: int,
            learning_rate: Union[float, Callable],
            random_state: int = 42,
            sgd_sample: Optional[Union[int, float]] = None,
            metric: Optional[str] = None,
            reg: Optional[str] = None,
            l1_coef: float = 0,
            l2_coef: float = 0,
    ) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.sgd_sample = sgd_sample
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.weights = None
        self.best_score = None

    def __str__(self) -> str:
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = False) -> None:
        """Fit the linear regression model to the training data.

        :param X: pd.DataFrame:
        :param y: pd.Series:
        :param verbose: bool:  (Default value = False)

        """
        seed(self.random_state)
        X = self._add_intercept(X)
        num_features = X.shape[1]
        self.weights = np.ones(num_features)

        if verbose:
            print("start | loss:", self._mse_loss(X, y, self.weights))

        for iteration in range(1, self.n_iter+1):
            mini_batch_idx = self._sample_rows_idx(X, self.sgd_sample)
            y_pred = np.dot(X, self.weights)
            loss = self._mse_loss(X, y, self.weights) + \
                self._loss_reg(self.reg, self.weights)
            gradient = self._gradient(
                X[mini_batch_idx, :], y[mini_batch_idx], y_pred[mini_batch_idx])

            if callable(self.learning_rate):
                lr = self.learning_rate(iteration)
            else:
                lr = self.learning_rate

            self.weights -= lr * gradient

            if verbose and (iteration + 1) % 10 == 0:
                if self.metric is not None:
                    metric_value = self._calculate_metric(y, y_pred)
                    print(iteration, "| loss:", loss, "|",
                          self.metric + ":", metric_value)
                print("learning_rate:", lr)

        if self.metric is not None:
            y_pred = np.dot(X, self.weights)
            self.best_score = self._calculate_metric(y, y_pred)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the learned model.

        :param X: pd.DataFrame:
        :returns: np.ndarray: The predicted target values.

        """
        X = self._add_intercept(X)
        return np.dot(X, self.weights)

    def get_coef(self) -> float:
        """Get the coefficients of the linear regression model.

        :returns: The mean of the learned coefficients, excluding the intercept.

        :rtype: float

        """
        return np.mean(self.weights[1:])

    def get_best_score(self) -> Optional[float]:
        """Get the best score achieved during training based on the specified metric.

        :returns: The best score achieved during training, or None if no metric was specified.

        :rtype: float or None

        """
        return self.best_score

    @staticmethod
    def _add_intercept(X: pd.DataFrame) -> pd.DataFrame:
        """Add an intercept column to the input features.

        :param X: pd.DataFrame:
        :returns: pd.DataFrame: The input features with an intercept column added.

        """
        return pd.concat([pd.Series(1, index=X.index, name='intercept'), X], axis=1)

    @staticmethod
    def _mse_loss(X: pd.DataFrame, y: pd.Series, weights: np.ndarray) -> float:
        """Calculate the mean squared error loss.

        :param X: pd.DataFrame:
        :param y: pd.Series:
        :param weights: np.ndarray:
        :returns: float: The mean squared error loss.

        """
        y_pred = np.dot(X, weights)
        return np.mean((y_pred - y) ** 2)

    def _gradient(self, X: pd.DataFrame, y: pd.Series, y_pred: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the loss function.

        :param X: pd.DataFrame:
        :param y: pd.Series:
        :param y_pred: np.ndarray:
        :returns: np.ndarray: The gradient of the loss function.

        """
        gradient = (2 / len(y)) * np.dot(X.T, (y_pred - y))
        if self.reg == 'l1':
            return gradient + self.l1_coef * np.sign(self.weights)
        elif self.reg == 'l2':
            return gradient + 2 * self.l2_coef * self.weights
        elif self.reg == 'elasticnet':
            l1_term = self.l1_coef * np.sign(self.weights)
            l2_term = 2 * self.l2_coef * self.weights
            return gradient + l1_term + l2_term
        else:
            return gradient

    def _sample_rows_idx(self, X: pd.DataFrame, sgd_sample: Union[int, float, None]) -> Union[List[int], range]:
        """Sample rows for stochastic gradient descent.

        :param X: pd.DataFrame:
        :param sgd_sample: Union[int, float, None]:
        :returns: List[int] or range: The indices of the sampled rows.

        """
        if self.sgd_sample is None:
            return range(X.shape[0])
        elif isinstance(self.sgd_sample, int):
            return sample(range(X.shape[0]), sgd_sample)
        elif isinstance(self.sgd_sample, float):
            sample_size = int(X.shape[0] * sgd_sample)
            return sample(range(X.shape[0]), sample_size)

    def _loss_reg(self, reg: Optional[str], weights: np.ndarray) -> float:
        """Calculate the regularization loss.

        :param reg: Optional[str]:
        :param weights: np.ndarray:
        :returns: float: The regularization loss.

        """
        if reg == 'l1':
            return self.l1_coef * np.sum(np.abs(weights))
        elif reg == 'l2':
            return self.l2_coef * np.sum(weights ** 2)
        elif reg == 'elasticnet':
            l1_term = self.l1_coef * np.sum(np.abs(weights))
            l2_term = self.l2_coef * np.sum(weights ** 2)
            return l1_term + l2_term
        else:
            return 0

    def _calculate_metric(self, y: pd.Series, y_pred: np.ndarray) -> Optional[float]:
        """Calculate the specified evaluation metric.

        :param y: pd.Series:
        :param y_pred: np.ndarray:
        :returns: float or None: The calculated metric value, or None if no metric was specified.

        """
        if self.metric == 'mae':
            return np.mean(np.abs(y - y_pred))
        elif self.metric == 'mse':
            return np.mean((y - y_pred) ** 2)
        elif self.metric == 'rmse':
            return np.sqrt(np.mean((y - y_pred) ** 2))
        elif self.metric == 'mape':
            return np.mean(np.abs((y - y_pred) / y)) * 100
        elif self.metric == 'r2':
            sst = np.sum((y - y_pred) ** 2)
            ssr = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (sst / ssr)
            return r2
        else:
            return None