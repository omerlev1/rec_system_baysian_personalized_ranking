import numpy as np
import pandas as pd
from tqdm import tqdm
import functools
from joblib import Parallel, delayed
# from config import TRAIN_BPR_PATH, BPR_PARAMS, BPR_CANDIDATE_PARAMS,U_BEST_MODEL_FIT, \
#     U_BEST_MODEL_TRIAL, I_BEST_MODEL_FIT, I_BEST_MODEL_TRIAL, \
#     RANDOM_TEST_PATH, POPULARITY_TEST_PATH
from operator import itemgetter
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_auc_score


class BPR:
    def __init__(self,
                 k=15,
                 lr_u=0.01, lr_i=0.01,lr_j=0.01,
                 regularizers=dict(au=1e-1,av=1e-1),
                 n_users=0, n_items=0,
                 sample_method='Uniform',
                 max_epochs=10,early_stop_threshold=0.001,early_stopping_lag=10):
        self.k = k  # dimension to represent user/item vectors
        self.lr_u = lr_u
        self.lr_i = lr_i
        self.lr_j = lr_j
        self.users = np.random.rand(n_users + 1, self.k) * 1 #TODO speak with Karen, the vectors dimentation are opposite to the formulas
        self.items = np.random.rand(n_items + 1, self.k) * 1
        self.scores = self.predict() # the scores for all users and items
        self.early_stop_epoch = None
        self.max_epochs=max_epochs
        self.current_epoch = 0
        self.item_popularity = []
        self.sample_method = sample_method
        self.early_stop_threshold=early_stop_threshold
        self.early_stopping_lag = early_stopping_lag
        self.regularizers=regularizers
        self.positive_array= np.array([])


    def step(self, u, i, j, e_u_i_j):
        self.users[u] += self.lr_u * (
                    e_u_i_j * (self.items[i] - self.items[j])) -self.lr_u*self.regularizers['au']*self.users[u]
        self.items[i] += self.lr_i * (e_u_i_j * self.users[u]) -self.lr_i*self.regularizers['av']*self.items[i]
        self.items[j] += -self.lr_j * (e_u_i_j * self.users[u]) -self.lr_j*self.regularizers['av']*self.items[j]

    def predict(self):
        return self.users.dot(self.items.T)

    def predict_u(self, u, i, j):
        pred = self.users[u].dot(self.items[i].T) - self.users[u].dot(
            self.items[j].T)
        return pred

    def run_epoch(self,mspu=1,train_list=[]):
        trained = []
        train_list_step=train_list.copy()
        random.shuffle(train_list_step) # we shuffle the order of the update
        for u,pos,neg in tqdm(train_list): #we iterate on the users
            # some edge cases:
            # 1. we can have a case where len(neg) < len(pos), here we extend the list
            if len(neg)<len(pos):
                neg=self._extend_neg(neg,len(pos))
            # 2. I defined the mspu to 1 to have a fair run time check -NOT Needed anymore
            # we align to the list size if the number of postive session is lower than the max
            random.shuffle(neg) #should help in convergence
            rmspu=min(mspu,len(pos)) # we shuffle the order of triples

            for idx,i in enumerate(pos[:rmspu]): # we iterate on the positive samples
                # print(i)
                j=neg[idx]
                trained.append((u, i, j))
                pred = self.sigmoid(self.predict_u(u, i, j))
                e_u_i_j=1-pred
                self.step(u, i, j, e_u_i_j)

        likelihood_u_lst=0

        return likelihood_u_lst

    def sigmoid(self,x):
        return  1 / (1 + np.exp(-x))

    def _extend_neg(self,neg,n):
        """if the neg list is shorter than the positive we just extend it"""
        k=int(n/len(neg))+1
        return neg*k

    def auc_val(self,val_list):
        """
        we average the score for all users in the val set.
        the score for each goes as follows
        the number of times the prediction chance for the positive is higher than rest of negatives:
            sigmoid(xuT*vi) > sigmoid(XuTvj) for every j element in val set
            note, we dont apply sigmoid since when x1>x2 then sigmoid(x1)>sigmoid(x2)
        """
        total_auc=0
        for u,pos,neg in val_list:
            # vi=self.items[pos,:]
            # vjs=self.items[neg,:]
            # pos_score=np.dot(vi,self.users[[u],:].T)
            # negs_scores=np.dot(vjs,self.users[[u],:].T)
            pos_score=self.scores[u,pos]
            negs_scores=self.scores[u,neg]
            if len(pos)>1:
                p=self.sigmoid(np.concatenate((pos_score,negs_scores)))
                t=np.concatenate((np.ones((len(pos_score),1)),np.zeros((len(negs_scores),1))))
                user_auc=roc_auc_score(t,p)
            else:
                # we dont use sigmoid here since it is monotonic increasing function
                user_auc = (negs_scores <= pos_score).sum() / len(neg)
            total_auc += user_auc
        return total_auc / len(val_list)

    def loss_log_likelihood(self,train_list):
        total_loss=0
        count_items=0
        for u,pos,neg in train_list:
            # print (u)
            vis=pos
            if len(neg)<len(pos):
                neg=self._extend_neg(neg,len(pos))
            end_j=len(pos)
            count_items+=len(pos)
            vjs=neg[:end_j]
            vis_scores=self.scores[u,vis]
            vjs_scores=self.scores[u,vjs]
            total_loss+=np.log(self.sigmoid(vis_scores-vjs_scores)).sum()
        return total_loss/count_items

    def precision_at_n(self,n,val_list,train_list):
        """
        1. we rank the validation set best on the best scores
        2. Removing positive samples from the training set
        3. we check how many validated true session are in top n
        :parameter
        """
        #Todo: it is not fair to compute precision on validation set unless we zero the training set positives
        if self.positive_array.size == 0:
            self.positive_array = self.lookup_positive(train_list)
        precision=0
        for u,pos,neg in val_list:
            #we predict all items
            pred_u=self.scores[u,:]
            #zero'ing the pos from the train (or in this case minimizing)
            u_idx_t=self.positive_array[u]
            _,pos_item_t,_=train_list[u_idx_t]
            pred_u[pos_item_t]=pred_u.min()-1
            #TODO when we validate against the frequent - we need to remove the negative cases not in the validation set
            topn=np.argsort(pred_u)[-n:][::-1]
            precision+=len(set(topn).intersection(set(pos))) / n
        return precision/len(val_list)

    def mpr(self,sess_list):
        """"
        what is the rank of one positive prediction among all other negative rank
        if there are more than 1 negative, we choose only the first 1"""
        total_mpr=0
        for u, pos, neg in sess_list:
            if len(pos)>1:
                pos=pos[0]
            # we dont use sigmoid here since it is monotonic increasing function
            pos_score=self.scores[u,pos]
            negs_scores=self.scores[u,neg]
            #if we had the best prediction, the pos score would be the highest
            #best mpr is equal to 0
            total_mpr+=( (negs_scores>pos_score).sum() ) /(len(negs_scores))
        return total_mpr/len(sess_list)

    def classification_accuracy(self,val_list):
        accuracy = 0
        for u, pos, neg in val_list:
            # vis = self.items[pos, :]
            if len(neg) < len(pos):
                neg = self._extend_neg(neg, len(pos))
            end_j = len(pos)
            # vjs = self.items[neg[:end_j], :]
            vis = pos
            vjs = neg[:end_j]
            vis_scores=self.scores[u,vis]
            vjs_scores=self.scores[u,vjs]
            # correct=np.dot(vis, self.users[[u], :].T) > np.dot(vjs, self.users[[u], :].T)
            correct=vis_scores> vjs_scores
            accuracy+=correct.flatten().sum()/len(pos)
        return accuracy/len(val_list)

    def save_params(self, path_out_u, path_out_i):
        with open(path_out_u, 'wb') as f:
            np.save(f, self.users)
        with open(path_out_i, 'wb') as f:
            np.save(f, self.items)

    def load_params(self, path_out_u, path_out_i):
        with open(path_out_u, 'rb') as f:
            self.users = np.load(f, self.users)
        with open(path_out_i, 'rb') as f:
            self.items = np.load(f, self.users)

    def lookup_positive(self,train_list):
        lookup_arr=np.arange(0,len(train_list))
        for i,(u,p,n) in enumerate(train_list):
            lookup_arr[i]=u
        return lookup_arr

    def fit(self,train_list,val_list):
        self.positive_array=self.lookup_positive(train_list)
        best_auc_valid = 0
        self.loss_curve = dict(training_loglike=[],
                               validation_loglike=[],
                               validation_auc=[],
                               val_accuracy=[],
                               mpr=[],
                               precision_at_1=[],
                               precision_at_5=[],
                               precision_at_10=[])
        while True and self.current_epoch<=self.max_epochs:
            print('epoch:', self.current_epoch)
            train_likelihood  = self.run_epoch(mspu=4000,train_list=train_list)
            # ----  updating losses and scores ---- #
            #TODO: consider create a prediction matrix and have all losses use those scores instead of having each one calculating it

            self.scores= self.predict()

            self.loss_curve['training_loglike'].append(self.loss_log_likelihood(train_list))
            self.loss_curve['validation_loglike'].append(self.loss_log_likelihood(val_list))
            self.loss_curve['validation_auc'].append(self.auc_val(val_list))
            self.loss_curve['val_accuracy'].append(self.classification_accuracy(val_list))
            self.loss_curve['mpr'].append(self.mpr(val_list))
            self.loss_curve['precision_at_1'].append(self.precision_at_n(n=1, val_list=val_list,train_list=train_list))
            self.loss_curve['precision_at_5'].append(self.precision_at_n(n=5, val_list=val_list, train_list=train_list))
            self.loss_curve['precision_at_10'].append(self.precision_at_n(n=10, val_list=val_list, train_list=train_list))

            print(f"calc evaluation AUC: {self.loss_curve['validation_auc'][self.current_epoch]:.3f}")
            print(f"total train log likelihood: {self.loss_curve['training_loglike'][self.current_epoch]:.3f}")

            # ----  early stopping ---- #
            # Early stopping
            if self.current_epoch > self.early_stopping_lag:
                if self.loss_curve['validation_auc'][self.current_epoch] - self.early_stop_threshold < \
                        self.loss_curve['validation_auc'][self.current_epoch - self.early_stopping_lag]:
                    print(f"Reached early stopping in epoch {self.current_epoch}")
                    break

            self.current_epoch += 1

        return best_auc_valid

    def plot_learning_curve(self):
        # ---- plotting the validation and training ---- #
        fig, ax = plt.subplots(2,3,figsize=(12, 8))

        epochs = epochs = range(1, len(self.loss_curve['training_loglike'])+ 1)

        #left side
        tr_mse = ax[0,0].plot(epochs, self.loss_curve['training_loglike'], 'b', label='Training Loss (Normalized log-likelihood)')
        val_mse = ax[0,0].plot(epochs, self.loss_curve['validation_loglike'], 'g', label='Validation Loss (Normalized log-likelihood)')
        ax[0,0].legend()
        #left side
        tr_mse = ax[0,1].plot(epochs, self.loss_curve['validation_auc'], 'g', label='Validation AUC')
        ax[0,1].legend()
        #bottom
        tr_mse = ax[0, 2].plot(epochs, self.loss_curve['mpr'], 'g', label='mpr')
        ax[0, 2].legend()
        tr_mse = ax[1, 0].plot(epochs, self.loss_curve['precision_at_1'], 'g', label='val precision_at_1')
        ax[1, 0].legend()
        tr_mse = ax[1, 1].plot(epochs, self.loss_curve['precision_at_5'], 'g', label='val precision_at_5')
        ax[1, 1].legend()
        tr_mse = ax[1, 2].plot(epochs, self.loss_curve['precision_at_10'], 'g', label='val precision_at_10')
        ax[1, 2].legend()

        return fig.axes

    def fit_early_stop(self, train_list, best_epoch):
        """
        fit the model initialized based on best configuration until its early stop epoch. The fit is on all full
        dataset.
        """

        self.loss_curve = dict(training_loglike=[])
        for epoch in range(best_epoch):
            self.current_epoch = epoch
            print('epoch:', self.current_epoch)
            train_likelihood = self.run_epoch(mspu=4000, train_list=train_list)

            self.scores = self.predict()

            self.loss_curve['training_loglike'].append(self.loss_log_likelihood(train_list))
            print(f"total train log likelihood: {self.loss_curve['training_loglike'][self.current_epoch]:.3f}")
