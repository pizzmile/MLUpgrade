import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn import metrics


#  Class to model parameters search grid with simple train-test split
class GridSearchTV:
    model = None
    param_grid: ParameterGrid = None
    train_scores: np.ndarray = None
    train_time: np.ndarray = None
    val_scores: np.ndarray = None
    val_time: np.ndarray = None
    best_params: dict = None
    best_score: float = None
    worst_params: dict = None
    worst_score: float = None

    def __init__(self, model, params):
        self.model = model
        self.param_grid = ParameterGrid(params)
        self.train_scores = np.zeros(len(self.param_grid))
        self.train_time = np.zeros(len(self.param_grid))
        self.val_scores = np.zeros(len(self.param_grid))
        self.val_time = np.zeros(len(self.param_grid))

    # Train and validate model for each parameter combination
    def fit(self, x_train, y_train, x_val, y_val, verbose: bool = False):
        if verbose:
            print('Testing {} parameters combinations'.format(len(self.param_grid)))

        for i, params in enumerate(self.param_grid):
            self.model.set_params(**params)
            self.model.fit(x_train, y_train)

            # Compute the inference time
            start = time.time()
            self.train_scores[i] = self.model.score(x_train, y_train)
            end = time.time()
            self.train_time[i] = end - start
            #  Compute inference time
            start = time.time()
            self.val_scores[i] = self.model.score(x_val, y_val)
            end = time.time()
            self.val_time[i] = end - start

            # Plot step results
            if verbose:
                print(
                    '{}{}/{} | Training time: {}{}, Inference time: {}{} | T {}{}, V {}{} | Parameters: {}'.format(
                        ' ' * (len(str(len(self.param_grid))) - len(str(i + 1))),
                        i + 1,
                        len(self.param_grid),
                        round(self.train_time[i], 6),
                        '0' * (6 - len(str(round(self.train_time[i], 6)).split('.')[1])),
                        round(self.val_time[i], 6),
                        '0' * (6 - len(str(round(self.val_time[i], 6)).split('.')[1])),
                        round(self.train_scores[i], 4),
                        '0' * (6 - len(str(round(self.train_scores[i], 4)))),
                        round(self.val_scores[i], 4),
                        '0' * (6 - len(str(round(self.val_scores[i], 4)))),
                        params
                    )
                )
        # Store best parameters, score and time
        self.best_score = np.max(self.val_scores)
        self.best_params = self.param_grid[np.argmax(self.val_scores)]
        # Store worse parameters, score and time
        self.worst_score = np.min(self.val_scores)
        self.worst_params = self.param_grid[np.argmin(self.val_scores)]

        # Plot results
        if verbose:
            print(
                'Best parameters: {} | Score: {} | Training time: {}{} ; Inference time: {}{}'.format(
                    self.best_params,
                    self.best_score,
                    round(self.train_time[np.argmax(self.train_scores)], 6),
                    '0' * (6 - len(str(round(self.train_time[np.argmax(self.train_scores)], 6)).split('.')[1])),
                    round(self.val_time[np.argmax(self.val_scores)], 6),
                    '0' * (6 - len(str(round(self.val_time[np.argmax(self.val_scores)], 6)).split('.')[1]))
                )
            )
            print(
                'Worse parameters: {} | Score: {} | Training time: {}{} ; Inference time: {}{}'.format(
                    self.worst_params,
                    self.worst_score,
                    round(self.train_time[np.argmin(self.val_scores)], 6),
                    '0' * (6 - len(str(round(self.train_time[np.argmin(self.val_scores)], 6)).split('.')[1])),
                    round(self.val_time[np.argmin(self.val_scores)], 6),
                    '0' * (6 - len(str(round(self.val_time[np.argmin(self.val_scores)], 6)).split('.')[1]))
                )
            )

    # Plot scores
    def plot_scores_2d(self, show_best: bool = True, show_worse: bool = True):
        plt.figure()
        plt.plot(self.train_scores, label='train')
        plt.plot(self.val_scores, label='validation')
        if show_best:
            plt.scatter(np.argmax(self.val_scores), self.best_score, label='best', c='g')
        if show_worse:
            plt.scatter(np.argmin(self.val_scores), self.worst_score, label='worse', c='r')
        plt.legend()
        plt.show()

    def plot_scores_3d(self, show_best: bool = True, show_worse: bool = True):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.train_scores, self.val_scores, self.train_time, label='train', mode='markers')
        ax.plot(self.train_scores, self.val_scores, self.val_time, label='validation', mode='markers')
        if show_best:
            ax.scatter(self.train_scores[np.argmax(self.val_scores)], self.best_score,
                       self.train_time[np.argmax(self.val_scores)], label='best', c='g')
        if show_worse:
            ax.scatter(self.train_scores[np.argmin(self.val_scores)], self.worst_score,
                       self.train_time[np.argmin(self.val_scores)], label='worse', c='r')
        ax.legend()
        plt.show()

    #  Plot confusion matrix between the best model and worse one
    def plot_confusion_matrix(self, X_train, y_train, X_test, y_test):
        #  Best model
        self.model.set_params(**self.best_params)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        cm = metrics.confusion_matrix(y_test, y_pred)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix - Best model')
        plt.colorbar()
        tick_marks = np.arange(40)
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

        #  Worse model
        self.model.set_params(**self.worst_params)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        cm = metrics.confusion_matrix(y_test, y_pred)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix - Worse model')
        plt.colorbar()
        tick_marks = np.arange(40)
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def get_best_model(self):
        tmp_model = self.model
        tmp_model.set_params(**self.best_params)
        return tmp_model, self.best_params

    def get_worst_model(self):
        tmp_model = self.model
        tmp_model.set_params(**self.worst_params)
        return tmp_model, self.worst_params
