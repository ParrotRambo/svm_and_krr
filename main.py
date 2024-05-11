
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tqdm import tqdm

from models import *


def mse(predictions, targets):
    return np.mean((predictions - targets)**2)


def read_sine():
    df = pd.read_csv('sine.csv')
    train = df.head(int(len(df.index) * 0.8))
    test = df.head(len(df.index) - int(len(df.index) * 0.8))
    return df, train['x'].to_numpy(), train['y'].to_numpy(), test['x'].to_numpy(), test['y'].to_numpy()


def test_sine():
    Xy, X_train, y_train, X_test, y_test = read_sine()

    Xy = Xy.sort_values('x')

    X = Xy['x'].to_numpy()[np.newaxis].T
    y = Xy['y'].to_numpy()

    X_train = X_train[np.newaxis].T
    X_test = X_test[np.newaxis].T

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X = scaler.transform(X)

    ridge_rbf = KernelizedRidgeRegression(kernel=RBF(sigma=0.5), lambda_=0.0001)
    ridge_rbf = ridge_rbf.fit(X_train_scaled, y_train)

    ridge_pol = KernelizedRidgeRegression(kernel=Polynomial(M=15), lambda_=0.001)
    ridge_pol = ridge_pol.fit(X_train_scaled, y_train)

    svr_rbf = SVR(kernel=RBF(sigma=0.6), lambda_=0.001, epsilon=1)
    svr_rbf = svr_rbf.fit(X_train_scaled, y_train)

    svr_pol = SVR(kernel=Polynomial(M=10), lambda_=0.01, epsilon=1)
    svr_pol = svr_pol.fit(X_train_scaled, y_train)

    print('SVR_RBF: {rbf}     SVR_POL: {pol}'.format(rbf=mse(svr_rbf.predict(X_test_scaled), y_test), pol=mse(svr_pol.predict(X_test_scaled), y_test)))
    print('RIDGE_RBF: {rbf}     RIDGE_POL: {pol}'.format(rbf=mse(ridge_rbf.predict(X_test_scaled), y_test), pol=mse(ridge_pol.predict(X_test_scaled), y_test)))

    a_rbf = svr_rbf.get_alpha()
    a_rbf = np.subtract(a_rbf[:, 0], a_rbf[:, 1])
    a_rbf[np.abs(a_rbf) < 1e-5] = 0
    a_pol = svr_pol.get_alpha()
    a_pol = np.subtract(a_pol[:, 0], a_pol[:, 1])
    a_pol[np.abs(a_pol) < 1e-5] = 0

    print('X_length: {}     SVR_RBF SVs: {}     SRV_POL SVs: {}'.format(X_train_scaled.shape[0], np.count_nonzero(a_rbf), np.count_nonzero(a_pol)))

    rbf_ind = a_rbf != 0
    pol_ind = a_pol != 0

    #X = X_train_scaled
    X = X_train
    y = y_train
    X_sorted = scaler.transform(Xy['x'].to_numpy()[np.newaxis].T)
    #X_sorted = Xy['x'].to_numpy()[np.newaxis].T

    plt.scatter(X[~rbf_ind, :], y[~rbf_ind], color = 'red', label='data')
    plt.scatter(X[rbf_ind, :], y[rbf_ind], color = 'green', label='SVs')
    plt.plot(scaler.inverse_transform(X_sorted), svr_rbf.predict(X_sorted), color = 'blue', label='fit')
    plt.title('Support Vector Regression Model (RBF)')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('svr_rbf.svg', format='svg', dpi=1200)
    plt.clf()

    plt.scatter(X[~pol_ind, :], y[~pol_ind], color = 'red', label='data')
    plt.scatter(X[pol_ind, :], y[pol_ind], color = 'green', label='SVs')
    plt.plot(scaler.inverse_transform(X_sorted), svr_pol.predict(X_sorted), color = 'blue', label='fit')
    plt.title('Support Vector Regression Model (Polynomial)')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('svr_poly.svg', format='svg', dpi=1200)
    plt.clf()

    plt.scatter(X, y, color = 'red', label='data')
    plt.plot(scaler.inverse_transform(X_sorted), ridge_rbf.predict(X_sorted), color = 'blue', label='fit')
    plt.title('Ridge Regression Model (RBF)')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('rr_rbf.svg', format='svg', dpi=1200)
    plt.clf()

    plt.scatter(X, y, color = 'red', label='data')
    plt.plot(scaler.inverse_transform(X_sorted), ridge_pol.predict(X_sorted), color = 'blue', label='fit')
    plt.title('Ridge Regression Model (Polynomial)')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('rr_poly.svg', format='svg', dpi=1200)
    plt.clf()
    
    
if __name__ == "__main__":
    test_sine()