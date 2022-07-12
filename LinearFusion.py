import pandas as pd
import lossSet as ls
import cal_IR as cr
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV


class LinearFusion:
    def __init__(self, true_series, models_train_res_list, models_test_res_list=None):
        self.train_res = models_train_res_list
        self.test_res = models_test_res_list
        self.true_res = true_series
        self.na_weight = None
        self.er_weight = None
        self.ic_weight = None
        self.ir_weight = None
        self.lss_weight = None
        self.train_res_df = pd.DataFrame()
        if self.test_res is None:
            self.have_test_res = False
        else:
            self.have_test_res = True
            self.test_res_df = pd.DataFrame()

    def add_model(self, new_train_res, new_test_res=None):
        self.train_res.append(new_train_res)
        if (new_test_res is None) | (self.have_test_res == False):
            return
        else:
            self.test_res.append(new_test_res)
        return

    def models_merge(self):  # merge all model's res_series as an df
        all_train = pd.DataFrame()
        model_nm = []
        for i in range(len(self.train_res)):
            all_train = pd.concat([all_train, self.train_res[i]], axis=1)
            model_nm.append('model' + str(i + 1))
        all_train.columns = model_nm
        all_train.index = self.train_res[0].index
        self.train_res_df = all_train
        if self.have_test_res:
            all_test = pd.DataFrame()
            for i in range(len(self.test_res)):
                all_test = pd.concat([all_test, self.test_res[i]], axis=1)
            all_test.columns = model_nm
            all_test.index = self.test_res[0].index
            self.test_res_df = all_test
        return

    def models_corr(self, set_nm):
        self.models_merge()
        if set_nm == 'train':
            corr_df = self.train_res_df.corr()
        elif self.have_test_res:
            corr_df = self.test_res_df.corr()
        else:
            print('No test set data to calculate correlation!')
            return
        return corr_df

    def naive_avg_weight(self):
        self.models_merge()
        self.na_weight = [1 / len(self.train_res)] * len(self.train_res)
        return

    def error_norm_weight(self, loss_func='mse'):
        self.models_merge()
        model_loss_list = []
        for i in range(len(self.train_res_df.columns)):
            if loss_func == 'mae':
                loss = ls.mae(self.train_res_df.iloc[:, i], self.true_res[list(self.train_res_df.index)])  # val
            elif loss_func == 'mape':
                loss = ls.mape(self.train_res_df.iloc[:, i], self.true_res[list(self.train_res_df.index)])
            elif loss_func == 'mse':
                loss = ls.mse(self.train_res_df.iloc[:, i], self.true_res[list(self.train_res_df.index)])
            else:
                loss = ls.smape(self.train_res_df.iloc[:, i], self.true_res[list(self.train_res_df.index)])
            model_loss_list.append(loss)  # 一维list

        # weight_i=(loss_max-loss_i)/sum_i(loss_max-loss_i)
        loss_max = max(model_loss_list)
        loss_sum = sum([loss_max - loss for loss in model_loss_list])

        self.er_weight = [(loss_max - loss) / loss_sum for loss in model_loss_list]
        return

    def ic_norm_weight(self):
        self.models_merge()
        self.true_res.name = 'true_res'
        train_df1 = pd.merge(self.train_res_df, self.true_res, right_index=True, left_index=True, how='inner')
        model_ic_list = []
        for i in range(len(self.train_res_df.columns)):
            cur_corr = train_df1.iloc[:, i].corr(train_df1.iloc[:, -1])
            model_ic_list.append(abs(cur_corr))

        corr_min = min(model_ic_list)
        corr_sum = sum([corr - corr_min for corr in model_ic_list])

        self.ic_weight = [(corr - corr_min) / corr_sum for corr in model_ic_list]
        return

    def ir_norm_weight(self):
        self.models_merge()
        self.true_res.name = 'true_res'
        train_df1 = pd.merge(self.train_res_df, self.true_res, right_index=True, left_index=True, how='inner')

        ir_list = []
        for i in range(len(self.train_res_df.columns)):
            cur_ir = cr.cal_ir(train_df1.iloc[:, i], train_df1.iloc[:, -1])
            ir_list.append(cur_ir)

        ir_min = min(ir_list)
        ir_sum = sum(ir - ir_min for ir in ir_list)

        self.ir_weight = [(ir - ir_min) / ir_sum for ir in ir_list]
        return

    def lasso_weight(self):
        self.models_merge()
        self.true_res.name = 'true_res'
        train_df1 = pd.merge(self.train_res_df, self.true_res, right_index=True, left_index=True, how='inner')

        reg_X = train_df1.iloc[:, :-1]
        reg_y = train_df1.iloc[:, -1]

        lassocv = LassoCV()
        lassocv.fit(reg_X, reg_y)
        alpha = lassocv.alpha_
        lasso = Lasso(alpha=alpha)
        lasso.fit(reg_X, reg_y)
        lasso_coefs = list(lasso.coef_)

        lasso_min = min(lasso_coefs)
        lasso_sum = sum([lss - lasso_min for lss in lasso_coefs])

        self.lss_weight = [(lss - lasso_min) / lasso_sum for lss in lasso_coefs]
        return

    def get_pred(self, set, weight_list):  # train: set=1 test: set=0
        if set:
            pred = self.train_res_df.copy()
            for i in range(len(self.train_res_df.columns)):
                pred.iloc[:, i] = self.train_res_df.iloc[:, i] * weight_list[i]
            return pred.sum(axis=1)
        elif self.have_test_res:
            pred = self.test_res_df.copy()
            for i in range(len(self.test_res_df.columns)):
                pred.iloc[:, i] = self.test_res_df.iloc[:, i] * weight_list[i]
            return pred.sum(axis=1)
        else:
            print('No test set data!')
            return

    def na_train_pred(self):
        self.naive_avg_weight()
        return self.train_res_df.mean(axis=1)

    def ew_train_pred(self):
        self.error_norm_weight()
        cur_pred =self.get_pred(1,self.er_weight)
        return cur_pred

    def ic_train_pred(self):
        self.ic_norm_weight()
        cur_pred = self.get_pred(1,self.ic_weight)
        return cur_pred

    def ir_train_pred(self):
        self.ir_norm_weight()
        cur_pred = self.get_pred(1,self.ir_weight)
        return cur_pred

    def lasso_train_pred(self):
        self.lasso_weight()
        cur_pred=self.get_pred(1,self.lss_weight)
        return cur_pred

    def na_test_pred(self):
        self.naive_avg_weight()
        return self.test_res_df.mean(axis=1)

    def ew_test_pred(self):
        self.error_norm_weight()
        cur_pred=self.get_pred(0,self.er_weight)
        return cur_pred

    def ic_test_pred(self):
        self.ic_norm_weight()
        cur_pred=self.get_pred(0,self.ic_weight)
        return cur_pred

    def ir_test_pred(self):
        self.ir_norm_weight()
        cur_pred=self.get_pred(0,self.ir_weight)
        return cur_pred

    def lasso_test_pred(self):
        self.lasso_weight()
        cur_pred=self.get_pred(0,self.lss_weight)
        return cur_pred





