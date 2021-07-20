
from modules.bpr import *
from modules.data_bpr import prep_data
from config.config import BPR_PARAMS


if __name__ == '__main__':
    subsetting= False #run on less data just to test the code
    sample_users = 500
    sample_items = 1000
    rd = prep_data()
    if subsetting:
        rd.subset_train(n_users=sample_users, n_items=sample_items)
        BPR_PARAMS['n_users']=sample_users
        BPR_PARAMS['n_items']=sample_items

    train_list, val_list = rd.get_train_val_lists(neg_method='distribution',val_type='leave_one_out')
    # train_list, val_list = rd.get_train_val_lists(neg_method='distribution',val_type='normal',val_quant=0.2)

    ###
    model = BPR(**BPR_PARAMS)

    print('Starting point: ')
    print('---------------------')
    # print(model.auc_val(val_list))
    print(model.loss_log_likelihood(val_list))
    print(model.loss_log_likelihood(train_list))
    print(model.precision_at_n(n=5,val_list=val_list,train_list=train_list))
    print(f" MPR: {model.mpr(sess_list=val_list):.3f}")

    print('Training phase: ')
    trial_auc = model.fit(train_list, val_list)
    print(f" MPR: {model.mpr(sess_list=val_list):.3f}")
    print(model.precision_at_n(n=5, val_list=val_list,train_list=train_list))
    fig=model.plot_learning_curve()
    plt.show()

