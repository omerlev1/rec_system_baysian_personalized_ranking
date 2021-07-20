import bpr, data_bpr
import config


def infer_triple(model, u, i1, i2):
    """
    return 0 if the first item is the pne preferable by thw user, 1 otherwise.
    """
    pred = model.predict_u(u, i1, i2)
    if pred > 0:
        return 0
    else:
        return 1


if __name__ == '__main__':
    '''
    load test
    load best config and best epochs.
    create the data set based on best epochs - for each 
    '''
    best_epoch = 20
    best_config_params = {
            # Model parameters
            'n_users': 6040,
            'n_items': 3705,
            'k': 20,
            'lr_u': 0.01,
            'lr_i': 0.01,
            'lr_j': 0.01,
            'regularizers': dict(au=1e-1, av=1e-1),
            # Model parameters
            'sample_method': 'Uniform',
            # training loop parameters
            'max_epochs': 20,
            'early_stop_threshold': 0.001,
            'early_stopping_lag': 0}
    model = bpr.BPR(**best_config_params)
    rd = data_bpr.prep_data()
    train_list_uniform = rd.get_training_list(rd.data.reset_index(), 'uniform')
    model.fit_early_stop(train_list_uniform, best_epoch)

    test_random = rd.load_sessions_file(config.config.RANDOM_TEST_PATH)
    test_random['bitClassification'] = test_random.apply(lambda row: infer_triple(model, row['UserID'],
                                                                                  row['Item1'],
                                                                                  row['Item2']), axis=1)
    test_random.to_csv(config.config.RANDOM_TEST_OUT)

    model = bpr.BPR(**best_config_params)
    train_list_popularity = rd.get_training_list(rd.data.reset_index(), 'distribution')
    model.fit_early_stop(train_list_popularity, best_epoch)
    test_popularity = rd.load_sessions_file(config.config.POPULARITY_TEST_PATH)
    test_popularity['bitClassification'] = test_popularity.apply(lambda row: infer_triple(model, row['UserID'],
                                                                                          row['Item1'],
                                                                                          row['Item2']), axis=1)
    test_popularity.to_csv(config.config.POPULARITY_TEST_OUT)
