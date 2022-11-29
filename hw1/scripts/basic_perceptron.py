from typing import NoReturn

import numpy as np


class Perceptron:
    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения),
        w[0] должен соответстовать константе,
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.

        """

        self.w = None
        self.fst = None
        self.snd = None
        self.n = iterations
        self.inc = 1
        self.dec = -1

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает простой перцептрон.
        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.

        """
        self.w = [0] * (X.shape[1] + 1)
        (self.fst, self.snd) = np.unique(y)
        for j in range(self.n):

            for i in range(len(X)):
                cur_sum = self.w[0]

                for xi in range(1, len(X[i]) + 1):
                    xe = X[i][xi - 1]
                    cur_sum += self.w[xi] * xe

                if y[i] != ((self.fst, self.snd)[bool(cur_sum < 0)]):
                    for wi in range(len(self.w)):
                        self.w[wi] += (self.inc if y[i] == self.fst else self.dec) *\
                                      (1 if wi == 0 else X[i][wi - 1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.

        Return
        ------
        labels : np.ndarray
            Вектор индексов классов
            (по одной метке для каждого элемента из X).

        """
        ans = []
        for x in X:
            cur_sum = self.w[0]
            for xi in range(len(x)):
                cur_sum += self.w[xi + 1] * x[xi]
            ans.append((self.fst, self.snd)[bool(cur_sum < 0)])
        return ans
