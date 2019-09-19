### Test file for linear_regression ###

import numpy as np
from linear_regression import *


def test_simulate_data():
    """
    Ensures that the simulate_data method returns data.
    """
    data = simulate_data()

    # make sure the appropriate keys exist
    assert 'X' in data.keys()
    assert 'y' in data.keys()
    assert 'beta' in data.keys()

    # make sure the first column is a column of ones
    assert np.allclose(data['X'][:,0], np.ones_like(data['X'][:,0]))


def test_compare_models():
    """
    Test compare_models function.
    """
    data = simulate_data()
    X = data['X']
    y = data['y']
    beta = data['beta']

    results = compare_models(X, y, beta)

    # statsmodels is accurate
    assert np.linalg.norm(results['statsmodels']-results['truth']) < .5

    # sklearn is accurate
    assert np.linalg.norm(results['sklearn']-results['truth']) < .5

    # sklearn method == statsmodels method
    assert np.allclose(results['sklearn'], results['statsmodels'])


def test_load_hospital_data():
    """
    Test load_hospital_data function.
    """
    df = load_hospital_data("./hospital_charge_sample.csv")

    # make sure all variables we want are accounted for
    assert 'average covered charges' in df.columns
    assert 'total discharges' in df.columns
    assert 'average total payments' in df.columns
    assert 'provider state' in df.columns
    assert 'average medicare payments' in df.columns

    # make sure the data types are correct
    df_dtypes = df.dtypes
    assert df_dtypes['average covered charges'] == np.float64
    assert df_dtypes['total discharges'] == np.int64
    assert df_dtypes['average total payments'] == np.float64
    assert df_dtypes['average medicare payments'] == np.float64
    assert df_dtypes['provider state'] == object

    # ensure non-negativity
    assert df['average total payments'].all() >= 0
    assert df['average medicare payments'].all() >= 0
    assert df['average covered charges'].all() >= 0
    assert df['total discharges'].all() >= 0

    # no more than 50 states plus the District of Columbia
    assert len(df['provider state'].unique()) <= 51

    # make sure states line up with provider ids
    assert df[df.index==10023]['provider state'].all() == 'AL'
    assert df[df.index==520103]['provider state'].all() == 'WI'
    assert df[df.index==460047]['provider state'].all() == 'UT'



def test_prepare_data():
    """
    Test prepare_data function.
    """
    df = load_hospital_data("./hospital_charge_sample.csv")
    data = prepare_data(df)

    # make sure independent and dependent variables exist in the dictionary
    assert 'X' in data.keys()
    assert 'y' in data.keys()

    # make sure first column is all ones
    assert np.allclose(data['X'][:,0], np.ones_like(data['X'][:,0]))

    assert data['X'].shape[0] == data['y'].shape[0]


# Tests on simulated data sets
print("testing...")
test_simulate_data()
test_compare_models()
test_load_hospital_data()
test_prepare_data()
print("\nSUCCESS!\n")


### END ###