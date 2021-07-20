import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(33)
import datetime
import time
import os


def nested_dict_to_df(dict_x):
    output = pd.DataFrame(index=[1])
    for k, v in dict_x.items():
        if type(v) == dict:
            df_i = pd.DataFrame.from_dict(v, orient='index').T
            df_i.columns = k + '_' + df_i.columns
            output = pd.concat([output, df_i], axis=1)
        else:
            output[k] = v
    return output

def time_string():
    t=datetime.datetime.now()
    return t.strftime("%Y%m%d_%H%M%S")

class hyper_search_optimizer(object):
    def __init__(self,models_params,candidate_params):
        """
        models models_params must include all candidate params not vice versa
        :param models_params: which parameters we run for all candidates
        :param candidate_params: which parameters we search in (uniform)
        """
        self.models_params=models_params #the dictionary with 1 instance choice of all parameters
        self.candidate_params=candidate_params #the dictionary with the search options

    def select_random_option(self,x):
        i = np.random.randint(0, len(x))
        return x[i]

    def get_candidate(self):
        candidate = {}
        for param_type in self.candidate_params.keys():
            if type(self.models_params[param_type]) == dict:
                #this case our parameter has a dictionary for with sub parameters
                param_sub = {}
                for sub_val in self.models_params[param_type].keys():
                    x = self.select_random_option(self.candidate_params[param_type])
                    param_sub[sub_val] = x
                candidate[param_type] = param_sub
            else:
                x = self.select_random_option(self.candidate_params[param_type])
                candidate[param_type] = x
        #adding the parameters that did not have parameters
        fixed_keys = list(set(self.models_params.keys()) - set(candidate.keys()))
        for param in fixed_keys:
            candidate[param] = self.models_params[param]
        return candidate

    def evaluate_hyper_random_search(self,model,
                                     models_params,
                                     fit_params,
                                     n_candidates=1,
                                     dir_results_tags=None):
        """
        model :parameter must provide model object with methods fit,plot_learning_curve
        """
        if dir_results_tags==None:
            dir_results = f"export/time_{time_string()}"
        else:
            dir_results = f"export/time_{time_string()}"
            #TODO implement tags

        output = pd.DataFrame()
        os.mkdir(dir_results)
        ## ---- loop over candidates---- ##
        export_filename = f"{dir_results}/_{time_string()}.csv"

        for i in range(n_candidates):
            candidate = self.get_candidate()
            c_model=model(**models_params,**candidate)
            start_run = time.time()
            c_model.fit(**fit_params)
            end_run = time.time()
            candidate['model'] = i

        ## this section needs some refactoring to generalize it ##
        ##---- update relevant metrics ----#
            #TODO add more metrics that we need to show improvments and sensitivity analysis
            candidate['naive_error']=0.5
            candidate['max_AUC'] = max(c_model.loss_curve['validation_auc'])
            candidate['best_epoch'] = np.argmax(c_model.loss_curve['validation_auc'])
            candidate['% improvement']=candidate['max_AUC']/candidate['naive_error']-1
            candidate['elapsed']=str(datetime.timedelta(seconds=end_run-start_run))

        ##---- done relevant metrics ----#
            fig=c_model.plot_learning_curve()
            plt.savefig(f'{dir_results}/model_{i}_{time_string()}.png')

        ## ---- saving run results --- ##
            output=output.append(nested_dict_to_df(candidate))
            plt.close('all')
            if os.path.exists(export_filename):
                append_write = 'a'  # append if already exists
            else:
                append_write = 'w'  # make a new file if not
            #printing to file
            output.to_csv(export_filename,mode=append_write)

if __name__=='__main__':
    from config.config import *
    from modules.bpr import *
    from modules.data_bpr import prep_data

    sampling = True
    sample_users = 100
    sample_items = 50

    if sampling:
        rd = prep_data(sample_users=100, sample_items=50)
    else:
        rd = prep_data()
    train_list, val_list = rd.get_train_val_lists(neg_method='uniform')

    hp=hyper_search_optimizer(models_params=BPR_PARAMS,candidate_params=BPR_CANDIDATE_PARAMS)
    hp.evaluate_hyper_random_search(model=BPR,
                                    models_params={},
                                    fit_params=dict(train_list=train_list,val_list= val_list),
                                    n_candidates=2,
                                    dir_results_tags=None)

