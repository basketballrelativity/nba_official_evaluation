"""
official_quality.py

Main script to assess NBA
official quality
"""
import random

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

import matplotlib.pyplot as plt

import utils


def lr_confidence_intervals(x_data, y_data, model_number, C, p):
    """ lr_confidence_intervals calculates the confidence intervals
    for model coefficients through bootstrapping the data model_number
    times

    @param x_data (pd.DataFrame): Input data
    @param y_data (list): Output data
    @param model_number (int): Number of models,
        and therefore coefficient values, from which the confidence
        intervals are calculated
    @param C (int): Regularization parameter for the
        logistic regression model
    @param p (int): Significance level for coefficients (divided by 2
        within the function)

    Returns:

        - interval_df (pd.DataFrame): DataFrame of confidence intervals
            and Z scores for each feature
    """

    random.seed(2398)

    store_coef = []
    for mod in range(model_number):
        x_samp, y_samp = resample(x_data, y_data, replace=True,
                                  random_state=np.random.randint(10000, size=1)[0])
        lr_here = LogisticRegression(C=C, solver='lbfgs', penalty='l2', random_state=93)
        f_here = lr_here.fit(x_samp, y_samp)

        store_coef.append(list(lr_here.coef_[0]))

    coef_df = pd.DataFrame(store_coef, columns=list(x_data))
    mean_df = pd.DataFrame(coef_df.mean(axis=0)).T
    std_df = pd.DataFrame(coef_df.std(axis=0)).T

    t_statistic = stats.t.ppf(1 - p/2, model_number - 1)
    lower_bound = lambda x: mean_df[x][0] - t_statistic*std_df[x][0]
    upper_bound = lambda x: mean_df[x][0] + t_statistic*std_df[x][0]

    lower_interval = [lower_bound(x) for x in list(x_data)]
    upper_interval = [upper_bound(x) for x in list(x_data)]

    z_score = mean_df/std_df
    interval_df = pd.DataFrame([upper_interval, lower_interval, list(z_score.loc[0])],
                               columns=list(x_data),
                               index=['upper', 'lower', 'Z'])

    return interval_df


def main():
    """ Read in and format data

    Returns:
        - official_df (DataFrame): DataFrame containing
            official IDs and foul/challenge data
        - l2m_df (DataFrame): DataFrame containing
            Last-Two Minute report data
    """


    # L2M and box score data
    l2m_df_18, box_df_18 = utils.read_files(2018)
    l2m_df_19, box_df_19 = utils.read_files(2019)
    l2m_df_20, box_df_20 = utils.read_files(2020)
    l2m_df_21, box_df_21 = utils.read_files(2021)
    # Challenges
    challenge_df = utils.read_challenges()

    # Concatenate dataframes
    l2m_df = pd.concat([l2m_df_18, l2m_df_19], axis=0)
    box_df = pd.concat([box_df_18, box_df_19], axis=0)

    l2m_df = pd.concat([l2m_df, l2m_df_20], axis=0)
    box_df = pd.concat([box_df, box_df_20], axis=0)

    l2m_df = pd.concat([l2m_df, l2m_df_21], axis=0)
    box_df = pd.concat([box_df, box_df_21], axis=0)

    # Format data
    l2m_df['Foul'] = [1 if 'foul' in str(x).lower() else 0
                           for x in l2m_df['Call Type']]
    challenge_df['Foul'] = [1 if x == 'Foul' else 0
                            for x in challenge_df['Challenge Type']]

    official_df = box_df.merge(l2m_df[['game_id', 'Review Decision', 'Foul']],
                               left_on='GAME_ID',
                               right_on='game_id')
    off_chal_df = box_df.merge(challenge_df[['GAME_ID', 'Result', 'Foul']],
                               left_on='GAME_ID',
                               right_on='GAME_ID')

    # One-hot encoding
    official_df = utils.officials_to_one_hot(official_df)
    off_chal_df = utils.officials_to_one_hot(off_chal_df)
    official_df = official_df[(pd.notnull(official_df['Review Decision']))
                              & (official_df['Review Decision'] != 'Undetectable')]

    official_df['output'] = [1 if x in ['CNC', 'CC', 'NCC']
                             else 0 for x in official_df['Review Decision']]

    off_chal_df['output'] = [1 if x == 'Successful' else 0 for x in off_chal_df['Result']]

    # Clean up
    del official_df['Official 1'], official_df['Official 2'], official_df['Official 3']
    del official_df['Review Decision'], official_df['game_id'], official_df['GAME_ID']
    del official_df['Official 1 Name'], official_df['Official 2 Name'], official_df['Official 3 Name']

    del off_chal_df['Official 1'], off_chal_df['Official 2'], off_chal_df['Official 3']
    del off_chal_df['Result'], off_chal_df['GAME_ID']

    for col in list(official_df):
        if col not in list(off_chal_df):
            off_chal_df[col] = [0]*len(off_chal_df)

    off_chal_df = off_chal_df[list(official_df)]
    official_df = pd.concat([official_df, off_chal_df])

    return official_df, l2m_df


def build_model(official_df):
    """ This function constructs the predictive models

    @param official_df (pd.DataFrame): DataFrame containing
            official IDs and foul/challenge data

    Returns:
        - X_train (DataFrame): DataFrame of training data
            predictors
        - y_train (pd.Series): Series of output features
        - lr_best (LogisticRegression): Chosen logistic
            regression model
        - int_df (DataFrame): DataFrame of referee effect
            confidence intervals
    """

    # Adding official effects
    predictors = [x for x in list(official_df) if x != 'output']

    # Train/test split
    X_train, X_test, y_train, y_test = \
        train_test_split(official_df[predictors],
                         official_df['output'],
                         test_size=0.3,
                         random_state=2089) 

    # Hyperparameter tuning
    params = {'C': [0.001, 0.01, 0.1, 1, 10]}
    lr = LogisticRegression(penalty='l2', solver='lbfgs', random_state=2389)
    clf = GridSearchCV(lr, params, cv=5, scoring='neg_log_loss')

    clf.fit(X_train, y_train)
    lr_best = clf.best_estimator_
    print('Best lr with C = ' + str(lr_best.C) + ' for a CV negative log loss score of ' + str(clf.best_score_))

    # Store referee effects
    off_df = pd.DataFrame({'Official': list(X_train), 'Coef': list(lr_best.coef_[0])})
    off_df.sort_values('Coef')

    # Training set calibration
    preds = lr_best.predict_proba(X_train)[:, 1]
    train_df = X_train.copy()
    train_df['output'] = list(y_train)
    train_df['preds'] = preds
    for val in np.linspace(0.5, 0.95, 10):
        num = sum(train_df[(train_df['preds']>=val)
                           & (train_df['preds']<val+0.05)]['output'])
        den = len(train_df[(train_df['preds']>=val)
                           & (train_df['preds']<val+0.05)]['output'])
        print(str(val) + " <= x < " + str(val + 0.05))
        print(str(num) + "/" + str(den))
        if den != 0:
            print(num/den)

    # Test set calibration
    preds = lr_best.predict_proba(X_test)[:, 1]
    test_df = X_test.copy()
    test_df['output'] = list(y_test)
    test_df['preds'] = preds
    for val in np.linspace(0.5, 0.95, 10):
        num = sum(test_df[(test_df['preds']>=val)
                           & (test_df['preds']<val+0.05)]['output'])
        den = len(test_df[(test_df['preds']>=val)
                           & (test_df['preds']<val+0.05)]['output'])
        print(str(val) + " <= x < " + str(val + 0.05))
        print(str(num) + "/" + str(den))
        if den != 0:
            print(num/den)

    # Construct confidence intervals
    int_df = lr_confidence_intervals(X_train, y_train, 1000, lr_best.C, 0.1)
    int_df = int_df.T
    int_df['mean'] = lr_best.coef_[0]
    int_df = int_df.T

    return X_train, y_train, lr_best, int_df


def visualize_model(clf, X_test, y_test):
    """ This function visualizes model calibration

    @param clf (LogisticRegression): Classification model
        for call accuracy
    @param X_test (pd.DataFrame): DataFrame of test data
            predictors
    @param y_test (pd.Series): Series of output features
    """

    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    prob_pos = clf.predict_proba(X_test)[:, 1]

    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",)

    ax2.hist(prob_pos, range=(0, 1), bins=20,
             histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    plt.show()


def visualize_coefficients(coef_df):
    """ This function visualizes model coeficients

    @param coef_df (pd.DataFrame): Mean model coefficients
        and bootstrapped upper and lower bounds
    """
    fig = plt.figure(figsize=(10, 10))
    axis = fig.add_subplot(111, aspect='auto')
    axis.axvline(0, 0, len(list(coef_df)) - 1, linestyle='--', color='k')

    coef_df = coef_df.T.sort_values('mean').T
    coef_df.to_csv("coef.csv", index=False)
    y_val = 1.5
    for col in list(coef_df):
        if col != 'Foul':
            lower = coef_df[col]['lower']
            upper = coef_df[col]['upper']
            coef = coef_df[col]['mean']
            if col != list(coef_df)[-2]:
                axis.plot([lower, upper], [y_val, y_val], 'k-', lw=2, alpha=0.5)
                axis.plot(coef, y_val, 'ko', alpha=0.5)
            else:
                axis.plot([lower, upper], [y_val, y_val], 'k-', lw=2, alpha=0.5, label='Confidence Interval')
                axis.plot(coef, y_val, 'ko', alpha=0.5, label='Mean')
            y_val += 1

    plt.xlabel('Model Coefficients', fontsize=32)
    plt.ylabel('Official', fontsize=32)
    plt.yticks([])
    plt.xticks(fontsize=24)
    plt.legend(fontsize=18)
    plt.savefig('interval.png')
    plt.show()


def visualize_playoff_officials(coef_df):
    """
    """

    off_df = coef_df.T.reset_index()
    off_df = off_df[off_df['index']!='Foul']
    off_df['index'] = [int(x) for x in off_df['index']]

    count_df = utils.playoff_assignments()
    off_df = off_df.merge(count_df,
                          left_on='index',
                          right_on='official_id',
                          how='left')

    print(np.mean(list(off_df[off_df['index'].isin(list(count_df["official_id"]))]['mean'])))
    print(np.mean(list(off_df[~off_df['index'].isin(list(count_df["official_id"]))]['mean'])))
    data = [list(off_df[off_df['index'].isin(list(count_df["official_id"]))]['mean']),
            list(off_df[~off_df['index'].isin(list(count_df["official_id"]))]['mean'])]

    fig = plt.figure(figsize=(10, 10))
    axis = fig.add_subplot(111, aspect='auto')
    axis.boxplot(data)
    axis.axhline(0, 0, 1, linestyle='--', color='k', alpha=0.5)
    plt.xticks([1, 2], labels=['Playoff Assignment', 'No Playoff Assignment'], fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel('Official Model Coefficients', fontsize=32)
    plt.savefig('playoff.png')
    plt.show()


if __name__ == '__main__':
    official_df, l2m_df = main()
    x_train, y_train, lr_best, interval_df = build_model(official_df)
    visualize_coefficients(interval_df)
    visualize_playoff_officials(interval_df)