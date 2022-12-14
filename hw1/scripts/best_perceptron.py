from typing import NoReturn
import numpy as np

from scripts.basic_perceptron import Perceptron


class PerceptronBest(Perceptron):

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

        super().__init__(iterations)

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает перцептрон.

        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.

        При этом в конце обучения оставляет веса,
        при которых значение accuracy было наибольшим.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.

        """
        self.w = [0] * (X.shape[1] + 1)
        (self.fst, self.snd) = np.unique(y)
        min_err = float('inf')
        nw = []
        nw[:] = self.w
        for j in range(self.n):
            en = 0
            for i in range(len(X)):
                cur_sum = self.w[0]

                for xi in range(1, len(X[i]) + 1):
                    xe = X[i][xi - 1]
                    cur_sum += self.w[xi] * xe

                if y[i] != ((self.fst, self.snd)[bool(cur_sum < 0)]):
                    en += 1
                    for wi in range(len(self.w)):
                        self.w[wi] += (self.inc if y[i] == self.fst else self.dec) * \
                                      (1 if wi == 0 else X[i][wi - 1])
            if en < min_err:
                min_err = en
                nw[:] = self.w
        self.w[:] = nw
        # print(self.w)
