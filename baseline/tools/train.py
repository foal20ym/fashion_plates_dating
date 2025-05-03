import utils.generator as gen
import models.models as mod

MODELS = {'NASNetMobile': mod.NASNetModel,
          'ResNet101': mod.ResNet101Model,
          'InceptionV3': mod.InceptionModel}


def main(model_type, data_dir, test_fold, log_dir, seed):
    '''Train model for given fold

    Parameters
    ----------
    model_type : Model configuration to be trained
    data_dir : Folder containing CSV files with labeled data
    test_fold : Fold number used for testing
    log_dir : Folder to which models and training progress is logged to
    seed : Seed value to use

    '''
    train_gen, valid_gen = gen.FoldGenerator.create_training_generators(
                                data_dir, test_fold, seed)

    model = MODELS[model_type](train_gen.image_size)
    model.train(train_gen, valid_gen)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                        description='Train model for given fold.',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_type', help='Model to train',
                        choices=MODELS.keys())
    parser.add_argument('data_dir', help='Folder containing CSV files')
    parser.add_argument('log_dir', help='Folder to log models and training '
                        'progress to')
    parser.add_argument('test_fold', help='Fold number used for testing',
                        type=int)
    parser.add_argument('seed', help='Seed value to use', type=int)

    args = vars(parser.parse_args())
    main(**args)
