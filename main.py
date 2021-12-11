# noinspection PyUnresolvedReferences
from models.LogisticRegression import LogisticRegression
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OrdinalEncoder
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd

if __name__ == "__main__":

    raw_data = pd.read_csv("./data/train.csv")
    print(f"Shape={raw_data.shape}")
    print(f"Columns={raw_data.columns}")
    y_data = raw_data[["Survived"]]
    x_data = raw_data[["Sex","Age"]]
    print(x_data.isnull().any())

    ordinal_encoder = OrdinalEncoder()
    imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

    x_data["Age"] = imputer.fit_transform(x_data["Age"].values.reshape(-1,1))
    x_data["Sex"] = ordinal_encoder.fit_transform(x_data["Sex"].values.reshape(-1,1))
    x_data = x_data["Sex"]
    #x_data = x_data["Age"]

    logistic_model = LogisticRegression(100,0.1,0.5,True)
    x = logistic_model.theta_vals
    logistic_model.fit_and_transform(x_data.values,y_data.values)

    # theta_vals = logistic_model.theta_vals[:]
    # errors = logistic_model.theta_vals[:,-1]

    theta_vals = np.array(logistic_model.theta_vals)
    q1_vals = theta_vals[:,0]
    q2_vals = theta_vals[:,1]
    q3_vals = theta_vals[:,2]
    errors = theta_vals[:,-1]

    print(logistic_model.coeff_)
    print(logistic_model.min_cost_)

    print("F")

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # surf = ax.scatter(q1_vals,errors)
    # plt.show()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # surf = ax.scatter(q2_vals, errors)
    # plt.show()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_trisurf(q1_vals,q2_vals,errors, cmap=cm.jet, linewidth=0)
    # fig.colorbar(surf)
    #
    # ax.set_xlabel('Q_S')
    # ax.set_ylabel('Q_Age')
    # ax.set_zlabel('Cost')
    #
    # plt.show()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_trisurf(q1_vals, q3_vals, errors, cmap=cm.jet, linewidth=0)
    # fig.colorbar(surf)
    #
    # ax.set_xlabel('Q_S')
    # ax.set_ylabel('Q_int')
    # ax.set_zlabel('Cost')
    #
    # plt.show()
    #
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_trisurf(q2_vals, q3_vals, errors, cmap=cm.jet, linewidth=0)
    # fig.colorbar(surf)
    #
    # ax.set_xlabel('Q_Age')
    # ax.set_ylabel('Q_int')
    # ax.set_zlabel('Cost')

    #plt.show()

    probabilities = logistic_model.get_hypothesis(x_data.values)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(x_data["Age"].values,x_data["Sex"].values,probabilities, cmap=cm.jet, linewidth=0)
    ax.scatter(x_data["Age"].values,x_data["Sex"].values,logistic_model.predict(x_data.values))
    fig.colorbar(surf)
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # surf = ax.scatter(x_data["Age"].values,probabilities)
    # plt.show()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # surf = ax.scatter(x_data["Sex"].values,probabilities)
    # plt.show()