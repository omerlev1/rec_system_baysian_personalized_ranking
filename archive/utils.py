from config import TRAIN_PATH, VALIDATION_PATH, USER_COL_NAME_IN_DATAEST, \
    ITEM_COL_NAME_IN_DATASET, USER_COL, ITEM_COL, RATING_COL, \
    RATING_COL_NAME_IN_DATASET, TEST_PATH
from scipy.sparse import csr_matrix
import pandas as pd


def turn_to_sparse(rows_num, cols_num, row_indxs, col_indxs, vals):
    mtx = csr_matrix((rows_num, cols_num))
    mtx[[i for i in row_indxs], [j for j in col_indxs]] = vals
    return mtx


def get_data(small_data=False, train_size=0.05, valid_size=0.4):
    """
    reads train, validation to python indices so we don't need to deal with it in each algorithm.
    of course, we 'learn' the indices (a mapping from the old indices to the new ones) only on the train set.
    if in the validation set there is an index that does not appear in the train set then we can put np.nan or
     other indicator that tells us that.
    """
    train_df = pd.read_csv(TRAIN_PATH) 
    train_df[USER_COL] = train_df[USER_COL_NAME_IN_DATAEST] - 1
    train_df[ITEM_COL] = train_df[ITEM_COL_NAME_IN_DATASET] - 1
    train_df[RATING_COL] = train_df[RATING_COL_NAME_IN_DATASET]
    valid_df = pd.read_csv(VALIDATION_PATH)
    valid_df[USER_COL] = valid_df[USER_COL_NAME_IN_DATAEST] - 1
    valid_df[ITEM_COL] = valid_df[ITEM_COL_NAME_IN_DATASET] - 1
    valid_df[RATING_COL] = valid_df[RATING_COL_NAME_IN_DATASET]
    # work with less data for testing and hyperparams search 
    train_df = train_df.iloc[:int(train_df.shape[0]*train_size), :] if small_data else train_df
    valid_df = valid_df.iloc[:int(valid_df.shape[0]*valid_size), :] if small_data else valid_df
    return train_df[[USER_COL, ITEM_COL, RATING_COL]], valid_df[[USER_COL, ITEM_COL, RATING_COL]]


def get_test():
    test_df = pd.read_csv(TEST_PATH)
    test_df[USER_COL] = test_df[USER_COL_NAME_IN_DATAEST] - 1
    test_df[ITEM_COL] = test_df[ITEM_COL_NAME_IN_DATASET] - 1

    return test_df[[USER_COL, ITEM_COL]]


class Config:
    def __init__(self, **kwargs):
        self._set_attributes(kwargs)

    def _set_attributes(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
