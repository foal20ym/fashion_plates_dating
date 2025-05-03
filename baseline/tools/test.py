import os

import utils.generator as gen
import models.models as mod


def main(model_path, data_dir, log_dir, test_fold):
    '''Test model on given test fold

    Parameters
    ----------
    model_path : Path to saved model
    data_dir : Folder containing CSV files with labeled data
    log_dir : Folder in which test results are saved
    test_fold : Fold number used for testing

    '''
    test_gen = gen.FoldGenerator.create_test_generator(data_dir, test_fold)
    model = mod.Model(test_gen.image_size, model_path)

    results = model.test(test_gen)

    results.to_csv(os.path.join(log_dir, f'results_{test_fold}.csv'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                        description='Test model for given fold',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_path', help='Path to trained model')
    parser.add_argument('data_dir', help='Folder containing CSV files')
    parser.add_argument('log_dir', help='Folder to log test results')
    parser.add_argument('test_fold', help='Fold number used for testing',
                        type=int)

    args = vars(parser.parse_args())
    main(**args)
