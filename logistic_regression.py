import pandas as pd
import numpy as np
from typing import Union, List
from random import seed, sample


class MyLogReg:

    def __init__(
            self,
            n_iter=10,
            learning_rate=0.1,
            weights=None,
            metric=None,
            reg=None,
            l1_coef=0,
            l2_coef=0,
            random_state=42,
            sgd_sample=None
    ):
        """Initialize the MyLogReg class.

        Params:
            n_iter (int): Number of iterations for training (default=10).
            learning_rate (float): Learning rate for gradient descent (default=0.1).
            weights (np.ndarray): Initial weights for the model. If not provided, initialized to ones.
            metric (str): Evaluation metric for model quality (e.g., 'accuracy', 'precision', 'recall', 'f1', 'roc_auc').
            reg (str): Type of regularization ('l1', 'l2', or None).
            l1_coef (float): L1 regularization coefficient (default=0).
            l2_coef (float): L2 regularization coefficient (default=0).
            random_state (int): Seed for random number generator (default=42).
            sgd_sample (Union[int, float, None]): Number of samples to use in each iteration of stochastic gradient descent.
                Can be an integer or a float from 0.0 to 1.0. If None, uses all data for each iteration (default).

        """
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.sgd_sample = sgd_sample
        self.weights = weights
        self.metric = metric
        self.best_score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def __str__(self) -> str:
        """Returns the string representation of the MyLogReg class.

        Returns:
            str: The string representation of the MyLogReg class.
        """
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}, metric={self.metric}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        """Fit the logistic regression model on the provided data.

        Params:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target vector (0 or 1).
            verbose (bool): Flag to print intermediate results (default=False).

        """
        seed(self.random_state)
        X = self._add_intercept(X)
        num_features = X.shape[1]
        self.weights = np.ones(num_features)

        start_loss = self._logloss(y, np.dot(X, self.weights))
        if self.metric:
            start_metric = self._calculate_metric(y, self.predict(X))
            if verbose:
                print("start | loss:", start_loss,
                      f"| {self.metric}: {start_metric:.2f}")

        for iteration in range(1, self.n_iter+1):
            mini_batch_idx = self._sample_rows_idx(X, self.sgd_sample)
            y_pred = np.dot(X, self.weights)
            y_proba = self._sigmoid(y_pred)
            log_loss = self._logloss(y, y_proba) + \
                self._loss_reg(self.reg, self.weights)

            gradient = self._gradient(X.to_numpy()[mini_batch_idx, :],
                                      y.to_numpy()[mini_batch_idx],
                                      y_proba[mini_batch_idx])

            if callable(self.learning_rate):
                lr = self.learning_rate(iteration)
            else:
                lr = self.learning_rate

            self.weights -= lr * gradient

            if self.metric and verbose and (iteration + 1) % 10 == 0:
                y_pred_train = self.predict(X)
                train_metric = self._calculate_metric(y, y_pred_train)
                print(iteration, "| loss:", log_loss,
                      f"| {self.metric}: {train_metric:.2f}")

        if self.metric and self.metric != 'roc_auc':
            y_pred_train = self.predict(X)
            self.best_score = self._calculate_metric(y, y_pred_train)
        elif self.metric == 'roc_auc':
            self.best_score = self._calculate_metric(y, self.predict_proba(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the probabilities for the positive class.

        Params:
            X (pd.DataFrame): The feature matrix.

        Returns:
            np.ndarray: The predicted probabilities for the positive class.

        """
        X = self._add_intercept(X)
        y_pred = np.dot(X, self.weights)
        y_proba = self._sigmoid(y_pred)
        return y_proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the binary class labels based on the given data.

        Params:
            X (pd.DataFrame): The feature matrix.

        Returns:
            np.ndarray: The predicted binary class labels (0 or 1).

        """
        y_proba = self.predict_proba(X)
        y_pred = np.where(y_proba > 0.5, 1, 0)
        return y_pred

    def _logloss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the log-loss for the predicted probabilities.

        Params:
            y (np.ndarray): The true target labels (0 or 1).
            y_pred (np.ndarray): The predicted probabilities for the positive class.

        Returns:
            float: The log-loss.

        """
        epsilon = 1e-15
        y_proba = self._sigmoid(y_pred)
        loss = -np.mean(y * np.log(y_proba + epsilon) + (1 - y)
                        * np.log(1 - y_proba + epsilon))
        return loss

    def _loss_reg(self, reg: str, weights: np.ndarray) -> float:
        """Calculate the regularization loss.

        Params:
            reg (str): Type of regularization ('l1', 'l2', or None).
            weights (np.ndarray): Model weights.

        Returns:
            float: The regularization loss.

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

    def _sample_rows_idx(self, X: pd.DataFrame, sgd_sample: Union[int, float, None]) -> Union[List[int], range]:
        """Sample rows for stochastic gradient descent.

        Params:
            X (pd.DataFrame): The feature matrix.
            sgd_sample (Union[int, float, None]): Number of samples to use in each iteration of stochastic gradient descent.

        Returns:
            Union[List[int], range]: The indices of the sampled rows.

        """
        if self.sgd_sample is None:
            return range(X.shape[0])
        elif isinstance(self.sgd_sample, int):
            return sample(range(X.shape[0]), sgd_sample)
        elif isinstance(self.sgd_sample, float):
            sample_size = int(X.shape[0] * sgd_sample)
            return sample(range(X.shape[0]), sample_size)

    def get_coef(self) -> np.ndarray:
        """Get the coefficients of the trained model.

        Returns:
            np.ndarray: The coefficients of the trained model (excluding the intercept).

        """
        return self.weights[1:]

    def _add_intercept(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add an intercept column of ones to the feature matrix.

        Params:
            X (pd.DataFrame): The feature matrix.

        Returns:
            pd.DataFrame: The feature matrix with an intercept column.

        """
        if 'intercept' not in X.columns:
            return pd.concat([pd.Series(1, index=X.index, name='intercept'), X], axis=1)
        return X

    @staticmethod
    def _sigmoid(z: np.ndarray) -> float:
        """Calculate the sigmoid function.

        Params:
            z (np.ndarray): Input to the sigmoid function.

        Returns:
            float: The output of the sigmoid function.

        """
        return 1 / (1 + np.exp(-z))

    def _gradient(self, X: pd.DataFrame, y: pd.Series, y_pred: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the loss function.

        Params:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The true target labels (0 or 1).
            y_pred (np.ndarray): The predicted probabilities for the positive class.

        Returns:
            np.ndarray: The gradient of the loss function.

        """
        gradient = (1 / len(y)) * np.dot((y_pred - y), X)
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

    def _calculate_metric(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate the specified evaluation metric for the model.

        Params:
            y_true (pd.Series): The true target labels (0 or 1).
            y_pred (pd.Series): The predicted target labels (0 or 1).

        Returns:
            float: The calculated evaluation metric value.

        """
        if self.metric == 'accuracy':
            return self._accuracy(y_true, y_pred)
        elif self.metric == 'precision':
            return self._precision(y_true, y_pred)
        elif self.metric == 'recall':
            return self._recall(y_true, y_pred)
        elif self.metric == 'f1':
            return self._f1(y_true, y_pred)
        elif self.metric == 'roc_auc':
            return self._roc_auc(y_true, y_pred)
        else:
            raise ValueError(f"Invalid metric: {self.metric}")

    def _accuracy(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate the accuracy metric.

        Params:
            y_true (pd.Series): The true target labels (0 or 1).
            y_pred (pd.Series): The predicted target labels (0 or 1).

        Returns:
            float: The accuracy metric.

        """
        return np.mean(y_true == y_pred)

    def _precision(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate the precision metric.

        Params:
            y_true (pd.Series): The true target labels (0 or 1).
            y_pred (pd.Series): The predicted target labels (0 or 1).

        Returns:
            float: The precision metric.

        """
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / predicted_positives if predicted_positives != 0 else 0

    def _recall(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate the recall metric.

        Params:
            y_true (pd.Series): The true target labels (0 or 1).
            y_pred (pd.Series): The predicted target labels (0 or 1).

        Returns:
            float: The recall metric.

        """
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / actual_positives if actual_positives != 0 else 0

    def _f1(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate the F1 score metric.

        Params:
            y_true (pd.Series): The true target labels (0 or 1).
            y_pred (pd.Series): The predicted target labels (0 or 1).

        Returns:
            float: The F1 score metric.

        """
        precision = self._precision(y_true, y_pred)
        recall = self._recall(y_true, y_pred)
        return 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    def _roc_auc(self, y_true: pd.Series, y_score: pd.DataFrame) -> float:
        """Calculate the ROC AUC metric.

        Params:
            y_true (pd.Series): The true target labels (0 or 1).
            y_score (pd.DataFrame): The predicted probabilities for the positive class.

        Returns:
            float: The ROC AUC metric.

        """
        y_score = np.round(y_score, decimals=10)
        combined_array = np.column_stack((y_score, y_true))
        combined_array = combined_array[np.argsort(combined_array[:, 0])[::-1]]

        positives = combined_array[combined_array[:, 1] == 1]
        negatives = combined_array[combined_array[:, 1] == 0]

        total = 0
        for row in negatives:
            score_higher = (positives[:, 0] > row[0]).sum()
            score_equal = (positives[:, 0] == row[0]).sum()
            total += score_higher + 0.5 * score_equal

        return total / (positives.shape[0] * negatives.shape[0])

    def get_best_score(self) -> float:
        """Get the best evaluation score achieved during training.

        Returns:
            float: The best evaluation score.

        """
        if self.best_score is None:
            raise ValueError("The model has not been trained yet. Call the 'fit' method first.")
        return self.best_score
