from pyiron_base import PythonTemplateJob
import pyiron

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from scipy.optimize import differential_evolution
import optuna
import numpy as np
import pickle
import os

# The current implementation of the application of the optuna optimizer is hard-coded and only works if the number 
# of hyperparameters is 2.

def optuna_optimizer(obj_func, initial_theta, bounds, nTrials):
    # obj_func: function handle for the cost function
    # initial_theta: array containing the initial values of the hyperparameters
    # bounds: array containing the bounds for the hyperparameters
    # nTrials: number of function evaluations per process
    nHyperParams = 2 
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1234))
    study.optimize(lambda trial: objective(trial,obj_func,bounds), n_trials=nTrials)
    best_params = study.best_params
    found_x = np.array([0.]*nHyperParams)
    found_x[0] = best_params["alpha"]
    found_x[1] = best_params["beta"]
    return found_x, obj_func(found_x, eval_gradient=False)

def de_optimizer(obj_func, initial_theta, bounds):
    res = differential_evolution(lambda x: obj_func(x, eval_gradient=False), bounds, maxiter=50, disp=False, polish=False, seed=1234)
    return res.x, obj_func(res.x, eval_gradient=False)

# The following is an auxiliary function which returns the value of the cost function and obtains a trial object 
# as argument, see the documentation of optuna for more details on trial objects.

def objective(trial, objFunc, bounds):
    nHyperParams = 2
    alpha = trial.suggest_float("alpha", bounds[0,0], bounds[0,1])
    beta = trial.suggest_float("beta", bounds[1,0], bounds[1,1])
    x = np.array([0.]*nHyperParams)
    x[0] = alpha
    x[1] = beta
    return objFunc(x, eval_gradient=False)


class GaussianProcessRegressionJob(PythonTemplateJob):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.input.xSamples = None
        self.input.ySamples = None
        self.input.kernel = "Gaussian"
        self.input.initialLengthScale = 1.0
        self.input.initialLengthScaleBounds = [0.00001, 100000]
        self.input.initialPreFactor = 1.0
        self.input.nu = 1.5
        self.input.optimizer = "fmin_l_bfgs_b"
        self.input.nTrials = 20
        self.input.nRestartsOptimizer = 9
        self.input.normalize_y_data = False
        self.input.noise = 5e-7

    def run_static(self):
        # Kernel
        if self.input.kernel == "Gaussian":
            initialLengthScale = self.input.initialLengthScale
            initialLengthScaleBounds = self.input.initialLengthScaleBounds
            kernel = self.input.initialPreFactor*RBF(length_scale=initialLengthScale,length_scale_bounds=(initialLengthScaleBounds[0],initialLengthScaleBounds[1]))
        elif self.input.kernel == 'Matern':
                initialLengthScale = self.input.initialLengthScale
                initialLengthScaleBounds = self.input.initialLengthScaleBounds
                nu = self.input.nu
                kernel = self.input.initialPreFactor*Matern(length_scale=initialLengthScale,length_scale_bounds=(initialLengthScaleBounds[0],initialLengthScaleBounds[1]),nu=nu)
        else:
            raise Exception(f"Kernel {self.input.kernel} not understood, should be 'Matern' or 'Gaussian'.")

        # Optimizer
        if self.input.optimizer == 'de_optimizer':
            Optimizer = de_optimizer
        elif self.input.optimizer == 'fmin_l_bfgs_b':
            Optimizer = self.input.optimizer
        elif self.input.optimizer == 'none':
            Optimizer = None
        elif self.input.optimizer == 'optuna':
            Optimizer = lambda obj_func,initial_theta,bounds: optuna_optimizer(obj_func,initial_theta,bounds,self.input.nTrials)
        else:
            print('Error: Unknown hyperparameter optimizer.')
            exit()

        # Gaussian Process Regressor
        self.gaussian_process = GaussianProcessRegressor(kernel=kernel,
                                                         optimizer=Optimizer,
                                                         n_restarts_optimizer=self.input.nRestartsOptimizer,
                                                         normalize_y=self.input.normalize_y_data,
                                                         random_state=1234,
                                                         alpha=self.input.noise)

        # Start the training
        self.gaussian_process.fit(self.input.xSamples, self.input.ySamples)

        self.to_hdf()
        self.status.finished = True

    def predict(self, xTestData):
        self.gaussian_process.predict(xTestData, return_std=True)

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf, group_name)
        if hasattr(self, "gaussian_process"):
            with open(f'gaussian_process_{self.name}.pkl', 'wb') as open_file:
                pickle.dump(self.gaussian_process, open_file)

    def from_hdf(self, hdf=None, group_name=None):
        return_value = super().from_hdf(hdf, group_name)
        if os.path.isfile(f'gaussian_process_{self.name}.pkl'):
            with open(f'gaussian_process_{self.name}.pkl', 'rb') as open_file:
                self.gaussian_process = pickle.load(open_file)
        else:
            pass
        return return_value

def main():
    pr = pyiron.Project("gaussian_progress_regression")
    pr.remove_jobs(silently=True, recursive=True)
    job = pr.create_job(GaussianProgressRegressionJob, job_name="test", delete_existing_job=False)

    X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
    Y = np.squeeze(X * np.sin(X))

    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(Y.size), size=20, replace=False)

    X_train, Y_train = X[training_indices], Y[training_indices]
    noise_std = 0.75
    Y_train_noisy = Y_train + rng.normal(loc=0.0, scale=noise_std, size=Y_train.shape)

    job.input.xSamples = X_train
    job.input.ySamples = Y_train_noisy
    job.run()


if __name__ == "__main__":
    main()