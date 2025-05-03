import os

import numpy as np
import pandas as pd


PREDICTED = 'predicted'
ACTUAL = 'gt'


def calculate_mae(data):
    '''Calculate the mean average error

    Parameters
    ----------
    data : DataFrame containing raw data

    Returns
    -------
    MAE

    '''
    return np.mean(np.abs(data[PREDICTED] - data[ACTUAL]))


def calculate_balanced_mae(data):
    '''Calculate mean average error per year before averaging over those
    results

    Parameters
    ----------
    data : DataFrame containing raw data

    Returns
    -------
    Balanced MAE

    '''
    maes = []

    years = data[ACTUAL].value_counts()
    for year, _ in years.items():
        subset = data[data[ACTUAL] == year]
        maes.append(calculate_mae(subset))

    return np.mean(maes)


def calculate_accuracy(data):
    '''Calculate the accuracy

    Parameters
    ----------
    data : DataFrame containing raw data

    Returns
    -------
    Accuracy

    '''
    predicted = np.round(data[PREDICTED])
    return np.sum(predicted == data[ACTUAL]) / len(data)


def main(csv_folder, output_csv):
    '''Calculate summary statistics for result CSVs in given folder

    Parameters
    ----------
    csv_folder : Folder containing CSV files
    output_csv : CSV file to write results to

    '''
    results = {'fold': [], 'mae': [], 'bmae': [], 'acc': []}
    csv_files = [os.path.join(csv_folder, f)
                 for f in os.listdir(csv_folder)
                 if f.endswith('.csv')]

    csv_files = sorted(csv_files)
    for i, csv_file in enumerate(csv_files):
        data = pd.read_csv(csv_file)

        assert f'{i}' in csv_file, f'Mismatch: {i} != {csv_file}'
        results['fold'].append(f'Fold {i}')
        results['mae'].append(calculate_mae(data))
        results['bmae'].append(calculate_balanced_mae(data))
        results['acc'].append(calculate_accuracy(data))

    if output_csv is None:
        for fold, mae, bmae, acc in zip(results['fold'], results['mae'],
                                        results['bmae'], results['acc']):
            print(f'{fold}\t{mae}\t{bmae}\t{acc}')

        print('-------------------------------')
        print(f'all\t{np.mean(results["mae"])}\t{np.mean(results["bmae"])}\t'
              f'{np.mean(results["acc"])}')
    else:
        pd.DataFrame(results).to_csv(output_csv, index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                        description='Calculate summary statistics for result '
                                    'CSVs in given folder',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('csv_folder', help='Folder containing CSV files')
    parser.add_argument('--output_csv', help='CSV file to write results to',
                        default=None)

    args = vars(parser.parse_args())
    main(**args)
