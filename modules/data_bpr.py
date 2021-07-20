import numpy as np
import pandas as pd
from config.config import *
import random
from tqdm import tqdm
import pickle


class prep_data(object):
    def __init__(self,sample_users=None,sample_items=None,train_path=TRAIN_BPR_PATH):
        self.items=None
        self.users = None
        self.sample_set = (sample_users, sample_items)
        self.data=self.load_sessions_file(train_path).set_index(["UserID"])
        self.get_items_users()
        self.train_list=None
        self.val_list=None

    def get_items_users(self):
        file_list=[RANDOM_TEST_PATH,POPULARITY_TEST_PATH]
        df_tests=pd.DataFrame()
        for file in file_list:
            # print(self.load_sessions_file(file).head())
            df_tests=df_tests.append(self.load_sessions_file(file))
        #we add the tests users alias to the train data
        self.users=np.unique(np.concatenate((self.data.index.values,df_tests['UserID'].values)))
        self.items=np.unique(np.concatenate((self.data['ItemID'].values,df_tests['Item1'].values,df_tests['Item2'].values)))
        print(f"Loaded {len(self.users):,} users and {len(self.items):,} items")

    def load_sessions_file(self,path):
        df=pd.read_csv(path)
        for col in df.columns:
            df[col] = df[col] - 1
        return df

    def leave_one_out(self,sess_df):
        # np.random.seed(9)
        sess_df_1 = sess_df.groupby('UserID').ItemID.apply(
            lambda x: x.sample(n=1)).reset_index()[['UserID', 'ItemID']]
        idx_to_remove=list(zip(sess_df_1['UserID'],sess_df_1['ItemID']))
        sess_df_all=sess_df.set_index(['UserID', 'ItemID']).drop(index=idx_to_remove).reset_index()
        sess_df_all.head()

        #sanity check:
        assert(len(sess_df_1)+len(sess_df_all)==len(sess_df))
        return sess_df_1, sess_df_all

    def split_train_test(self, sess_df, val_quant=0.2):
        np.random.seed(9)
        sess_df_val = sess_df.groupby('UserID').ItemID.apply(
            lambda x: x.sample(n=int(val_quant * len(x)))).reset_index()[['UserID', 'ItemID']]
        idx_to_remove = list(zip(sess_df_val['UserID'], sess_df_val['ItemID']))
        sess_df_train = sess_df.set_index(['UserID', 'ItemID']).drop(index=idx_to_remove).reset_index()
        sess_df_train.head()

        # sanity check:
        assert (len(sess_df_val) + len(sess_df_train) == len(sess_df))
        print(f"training size: {len(sess_df_train)}, val size: {len(sess_df_val)},\
         ratio: {len(sess_df_val) / len(sess_df_train)}")
        return sess_df_val, sess_df_train

    def get_train_val_lists(self,neg_method='uniform',val_type='leave_one_out',val_quant=0.2):
        if val_type=='leave_one_out':
            sess_df_val, sess_df_train =self.leave_one_out(self.data.reset_index())
        else:
            sess_df_val,sess_df_train = self.split_train_test(self.data.reset_index(),val_quant)
        self.train_list=self.get_training_list(sess_df_train,neg_method=neg_method).copy()
        self.val_list = self.get_training_list(sess_df_val, neg_method=neg_method).copy()

        return self.train_list,self.val_list

    def get_training_list(self,sess_df_v,neg_method):
        """
        returns as training list consisted of tuples of sessions [(user_id,[positive items],[random negative items]),...,]
        """

        session_list = []
        sess_df=sess_df_v.set_index('UserID')
        sess_df.sort_index(inplace=True)

        if neg_method == 'distribution':
            item_dist=self.data.groupby('ItemID').size()/len(sess_df)
            item_dist.name = 'weight'
            item_dist=item_dist.sort_values(ascending=False)#.reset_index().rename(columns={0:'weight'})
            # t=item_dist.sample(10,weights='weight',replace=True)
            # t.groupby('ItemID').size().sort_values(ascending=False)

        print(f"constructing users info in list, negative selections are by: {neg_method} \n")
        for u in tqdm(sess_df.index.unique()):
            # print(u)
            pos = sess_df.loc[u].ItemID
            if type(pos)==pd.core.series.Series:
                pos=pos.to_list()
            else:
                pos=[pos]
            if len(pos)==0:
                continue
            if neg_method=='uniform':
                #regardless if we look on training set, or leave on out set, we never wish to pick a negative this user liked
                original_pos=self.data.loc[u].ItemID.values
                neg = list(set(self.items) - set(original_pos))
                random.shuffle(neg)
            elif neg_method=='distribution':# this is sampled by priority
                #we need to remove the <original> positive items from the item_dist before we sample
                original_pos = self.data.loc[u].ItemID.values
                neg = list(item_dist.drop(original_pos).sample(len(pos), weights=item_dist, replace=True).index.values)
                #trick to remove pos element from the documentation:  index values in sampled object not in weights will be assigned weights of zero
                """
                #test example 
                #pos1=[813,404]
                #t=item_dist.drop(pos1).sample(1000000, weights=item_dist, replace=True)
                #t.groupby(t.index).size().sort_values(ascending=False)
                """
            session_list.append((u, pos, neg))
        return session_list

    def subset_train(self,n_users=10,n_items=10):
        """
        The major challange is to reindex the items and users to start from zero
        :param n_users:
        :param n_items:
        :return:
        """
        sess_df=self.data.reset_index()
        items_sample=np.random.choice(a=sess_df['ItemID'],size=n_items,replace=False)
        new_train=sess_df[sess_df['ItemID'].isin(items_sample)].copy()

        users_sample = np.random.choice(a=new_train['UserID'], size=n_users, replace=False)
        new_train=new_train[new_train['UserID'].isin(users_sample)].copy()

        u = new_train['UserID'].unique()
        new_user_idx=dict(zip(u,np.arange(len(u))))
        new_train['UserID']=new_train['UserID'].map(new_user_idx)
        self.users=np.arange(len(u))

        i=new_train['ItemID'].unique()
        new_item_idx = dict(zip(i, np.arange(len(i))))
        new_train['ItemID']=new_train['ItemID'].map(new_item_idx)
        self.items=np.arange(len(i))

        self.data=new_train.set_index('UserID')
        return new_train

    def save_local_train_val_list(self,pkl_path):
        # save to local file
        outfile = open(pkl_path, 'wb')
        print(f"saving lists to disk {pkl_path}")
        pickle.dump( (self.train_list,self.val_list), outfile)
        outfile.close()

    def load_local_train_val_list(self,pkl_path):
        print(f"loading lists from disk {pkl_path}")
        self.train_list, self.val_list = pickle.load(open(pkl_path, 'rb'))
        return self.train_list, self.val_list

if __name__ == '__main__':
    # rd=prep_data(sample_users=100,sample_items=50)
    # train_list,val_list=rd.get_train_val_lists(neg_method='uniform')

    rd=prep_data()

    # rd.subset_train(n_users=10,n_items=50)

    #20% split
    train_list, val_list = rd.get_train_val_lists(neg_method='uniform',val_type='normal',val_quant=0.2)

    #leave one out
    # train_list, val_list = rd.get_train_val_lists(neg_method='uniform', val_type='leave_one_out', val_quant=0.2)

    #short viz of the data model
    for ses in val_list:
        u, p, n = ses
        print(u, p[0:5], n[0:10])
    for ses in train_list:
        u, p, n = ses
        print(u, p[0:5], n[0:10])



    u1,p1,n=train_list[6039]
    u2,p2,n2=val_list[6038]
    u1,p1, n1 = val_list[6039]

    #
    # print(u1,p1)
    # print(u2,p2)
    # #print to file
    # with open('test_train.txt','w') as f:
    #     for ses in train_list:
    #         u, p, n = ses
    #         for p_i in p:
    #             f.write(f"{u} {p_i}\n")
    #         print(u, p[0:5], n[0:10])
    # with open('test_val.txt','w') as f:
    #     for ses in val_list:
    #         u, p, n = ses
    #         for p_i in p:
    #             f.write(f"{u} {p_i}\n")
    #         print(u, p[0:5], n[0:10])

