# -*- coding：utf-8 -*-
# Author: Ethan
# Time: 2021/8/31 14:20

import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
import os
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import gc
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
import xgboost as xgb
warnings.filterwarnings('ignore')

def load_dataset(path):
    train_df = pd.read_csv(path+'train.csv')
    test_df = pd.read_csv(path+'test.csv')
    return train_df, test_df

def gen_his_orders_stats(data, stride):
    orders_mean_feats = []
    orders_cols = ['author', 'level1', 'level2', 'level3', 'level4',
                   'brand', 'mall', 'url', 'baike_id_1h', 'baike_id_2h']
    for date in tqdm(range(4, 124, 7)):
        his_data = data[(data['date']<=date)&(data['date']>date-stride)].copy()
        df = data[(data['date']>date)&(data['date']<=date+7)][['article_id', 'date']+orders_cols].copy()
        for col in orders_cols:
            gp = his_data.groupby(col)['orders_3h_15h'].agg([[f'{col}_orders_{stride}d_mean', 'mean'],
                                                             [f'{col}_orders_{stride}d_std', 'std'],
                                                             [f'{col}_orders_{stride}d_min', 'min'],
                                                             [f'{col}_orders_{stride}d_max', 'max'],
                                                             [f'{col}_orders_{stride}d_med', 'median']]).reset_index()
            df = df.merge(gp, on=col, how='left')
        orders_mean_feats.append(df)
    orders_mean_feats = pd.concat(orders_mean_feats, axis=0, ignore_index=True)
    orders_mean_feats.drop(orders_cols, axis=1, inplace=True)
    return orders_mean_feats

def gen_magic_features(data):
    data['magic_rank'] = data.groupby(['date'])['article_id'].rank()

    for f in tqdm([['author'],
                   ['brand'],
                   ['mall'],
                   ['price'],
                   ['level1'],
                   ['level2'],
                   ['price'],
                   ['author', 'brand']]):
        g = data.groupby(f)
        name = '_'.join(f)
        for n in [1, 5, 10, 20]:
            data[f'magic_{name}_aft_gap{n}'] = g['article_id'].shift(-n) - g['article_id'].shift(0)
            data[f'magic_{name}_bef_gap{n}'] = g['article_id'].shift(0) - g['article_id'].shift(n)
        for stat in ['max', 'min', 'mean']:
            data[f'magic_{name}_id_{stat}'] = g['article_id'].transform(stat)

    data = data.sort_values(by=['date', 'magic_rank']).reset_index(drop=True)
    for key in ['author']:
        for col in ['orders_1h', 'orders_2h', 'magic_rank']:
            data[f'{key}_{col}_bef'] = data.groupby([key, 'date'])[col].shift(1)
            data[f'{key}_{col}_aft'] = data.groupby([key, 'date'])[col].shift(-1)
            data[f'{key}_{col}_bef_diff'] = data[col] - data[f'{key}_{col}_bef']
            data[f'{key}_{col}_aft_diff'] = data[f'{key}_{col}_aft'] - data[col]

    for key in ['author', 'level1', 'level2', 'level3', 'level4',
                'brand', 'mall', 'url', 'baike_id_2h']:
        data[f'{key}_rank_mean'] = data.groupby([key, 'date'])['magic_rank'].transform('mean')
        data[f'{key}_rank_std'] = data.groupby([key, 'date'])['magic_rank'].transform('std')
        data[f'{key}_rank_min'] = data.groupby([key, 'date'])['magic_rank'].transform('min')
        data[f'{key}_rank_max'] = data.groupby([key, 'date'])['magic_rank'].transform('max')

    data['baike_id_group'] = data.groupby(['date', 'baike_id_2h'])['article_id'].rank(method='first')
    data['baike_id_group'] = data['magic_rank'] - data['baike_id_group']
    data['baike_id_group'] = data.groupby(['date', 'baike_id_2h'])['baike_id_group'].rank(ascending=False, method='dense')
    return data

def feature_extractor(train_df, test_df):
    data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    data['week'] = (data['date'] - 5) // 7 # 这里把valid和test放入到各自相同的周内
    data['price_diff_ratio'] = data['price_diff'] / (data['price'] + data['price_diff'])

    data['zhi_1h_cnt'] = data['zhi_1h'] + data['buzhi_1h']
    data['zhi_2h_cnt'] = data['zhi_2h'] + data['buzhi_2h']
    data['zhi_1h_ratio'] = data['zhi_1h'] / data['zhi_1h_cnt']
    data.loc[data['zhi_1h_cnt'] == 0, 'zhi_1h_ratio'] = 0
    data['buzhi_1h_ratio'] = data['buzhi_1h'] / data['zhi_1h_cnt']
    data.loc[data['zhi_1h_cnt'] == 0, 'buzhi_1h_ratio'] = 0
    data['zhi_2h_ratio'] = data['zhi_2h'] / data['zhi_2h_cnt']
    data.loc[data['zhi_2h_cnt'] == 0, 'zhi_2h_ratio'] = 0
    data['buzhi_2h_ratio'] = data['buzhi_2h'] / data['zhi_2h_cnt']
    data.loc[data['zhi_2h_cnt'] == 0, 'buzhi_2h_ratio'] = 0

    ### 1. magic特征
    data = gen_magic_features(data)

    ### 2. 历史销量滑窗统计特征
    strides = [7, 150] # 150是指历史全部
    for d in strides:
        feats = gen_his_orders_stats(data, d)
        data = data.merge(feats, on=['article_id', 'date'], how='left')
        del feats
        gc.collect()

    ### 3. 当日、全局count统计特征
    stats_col = ['author', 'level1', 'level2', 'level3', 'level4',
                  'brand', 'mall', 'url', 'baike_id_2h']
    for col in tqdm(stats_col):
        data[f'{col}_day_cnt'] = data.groupby([col, 'date'])['article_id'].transform('count')
        data[f'{col}_global_cnt'] = data.groupby([col])['article_id'].transform('count')
        data[f'{col}_day_cnt_ratio'] = data[f'{col}_day_cnt'] / data[f'{col}_global_cnt']

    ### 4. 当日、当周、全局orders_1h、orders_2h统计特征
    stats_col = ['author', 'level1', 'level2', 'level3', 'level4',
                 'brand', 'mall', 'url', 'baike_id_2h']
    for label in ['orders_1h', 'orders_2h']:
        for col in tqdm(stats_col):
            data[f'{col}_day_{label}_sum'] = data.groupby([col, 'date'])[label].transform('sum')
            data[f'{col}_global_{label}_sum'] = data.groupby([col])[label].transform('sum')
            data[f'{col}_day_{label}_ratio'] = data[f'{col}_day_{label}_sum'] / data[f'{col}_global_{label}_sum']
            data[f'{col}_day_{label}_mean'] = data.groupby([col, 'date'])[label].transform('mean')
            data[f'{col}_vs_day_{label}'] = data[label] / data[f'{col}_day_{label}_mean']
            data[f'{col}_week_{label}_sum'] = data.groupby([col, 'week'])[label].transform('sum')
            data[f'{col}_day_in_week_{label}_ratio'] = data[f'{col}_day_{label}_sum'] / data[f'{col}_week_{label}_sum']
            data[f'{col}_week_{label}_ratio'] = data[f'{col}_week_{label}_sum'] / data[f'{col}_global_{label}_sum']

    ### 5. 当日、当周、全局收藏、点赞、评论统计特征
    stats_col = ['author', 'url', 'baike_id_1h', 'baike_id_2h']
    for label in ['comments_1h', 'zhi_1h', 'buzhi_1h', 'favorite_1h',
                  'comments_2h', 'zhi_2h', 'buzhi_2h', 'favorite_2h']:
        for col in tqdm(stats_col):
            data[f'{col}_day_{label}_sum'] = data.groupby([col, 'date'])[label].transform('sum')
            data[f'{col}_week_{label}_sum'] = data.groupby([col, 'week'])[label].transform('sum')
            data[f'{col}_global_{label}_sum'] = data.groupby([col])[label].transform('sum')
            data[f'{col}_day_in_week_{label}_ratio'] = data[f'{col}_day_{label}_sum'] / data[f'{col}_week_{label}_sum']
            data[f'{col}_week_{label}_ratio'] = data[f'{col}_week_{label}_sum'] / data[f'{col}_global_{label}_sum']
            data[f'{col}_day_{label}_ratio'] = data[f'{col}_day_{label}_sum'] / data[f'{col}_global_{label}_sum']

    ### 6. 商品价格、降价幅度统计特征
    stats_col = ['baike_id_2h', 'url']
    for key in tqdm(stats_col):
        for col in ['price', 'price_diff', 'price_diff_ratio']:
            data[f'{key}_{col}_mean'] = data.groupby([key])[col].transform('mean')
            data[f'{key}_{col}_std'] = data.groupby([key])[col].transform('std')
            data[f'{key}_{col}_vs_mean'] = data[col] / data[f'{key}_{col}_mean']

    ### 7. 当日、当周、全局nunique统计特征
    stats_col = [['author', 'url'], ['author', 'baike_id_2h'], ['baike_id_2h', 'url']]
    for col in tqdm(stats_col):
        data[f'{col[0]}_{col[1]}_nunique_day'] = data.groupby([col[0], 'date'])[col[1]].transform('nunique')
        data[f'{col[0]}_{col[1]}_nunique_week'] = data.groupby([col[0], 'week'])[col[1]].transform('nunique')
        data[f'{col[0]}_{col[1]}_nunique_global'] = data.groupby([col[0]])[col[1]].transform('nunique')

        data[f'{col[0]}_{col[1]}_nunique_day'] = data[f'{col[0]}_{col[1]}_nunique_day'] / data[f'{col[0]}_{col[1]}_nunique_global']
        data[f'{col[0]}_{col[1]}_nunique_week'] = data[f'{col[0]}_{col[1]}_nunique_week'] / data[f'{col[0]}_{col[1]}_nunique_global']

        data[f'{col[1]}_{col[0]}_nunique_day'] = data.groupby([col[1], 'date'])[col[0]].transform('nunique')
        data[f'{col[1]}_{col[0]}_nunique_week'] = data.groupby([col[1], 'week'])[col[0]].transform('nunique')
        data[f'{col[1]}_{col[0]}_nunique_global'] = data.groupby([col[1]])[col[0]].transform('nunique')

        data[f'{col[1]}_{col[0]}_nunique_day'] = data[f'{col[1]}_{col[0]}_nunique_day'] / data[f'{col[1]}_{col[0]}_nunique_global']
        data[f'{col[1]}_{col[0]}_nunique_week'] = data[f'{col[1]}_{col[0]}_nunique_week'] / data[f'{col[1]}_{col[0]}_nunique_global']

    return data

def split_dataset(data, valid_start_date, gap):
    train = data[data['date']<valid_start_date].copy()
    valid = data[(data['date']>=valid_start_date)&(data['date']<valid_start_date+gap)].copy()
    test = data[data['date']>=valid_start_date+gap].copy()
    print(f'训练集样本数: {train.shape[0]}, 验证集样本数: {valid.shape[0]}, 测试集样本数: {test.shape[0]}')
    return train, valid, test

def lgb_reg_model(train, valid, test, target):
    params = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'l2',
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'min_data_in_leaf': 200,
        'verbose': -1,
        'nthread': 16,
    }
    feats = [f for f in train.columns if f not in ['article_id', 'date', 'orders_3h_15h', 'week']]

    print('线下训练集样本数：%d' % (train.shape[0]))
    print('线下验证集样本数：%d' % (valid.shape[0]))
    print('特征个数：%d' % (len(feats)))
    print('开始线下验证......')

    dtrain = lgb.Dataset(train[feats],
                         label=train[target])
    dvalid = lgb.Dataset(valid[feats],
                         label=valid[target])
    lgb_model = lgb.train(params,
                          dtrain,
                          num_boost_round=10000,
                          valid_sets=[dvalid],
                          early_stopping_rounds=100,
                          verbose_eval=100)
    valid_preds = lgb_model.predict(valid[feats].values, num_iteration=lgb_model.best_iteration)
    print('--------------------------特征重要性----------------------------')
    feature_importance = pd.DataFrame()
    feature_importance['features'] = feats
    feature_importance['importance'] = lgb_model.feature_importance(importance_type='gain')
    print(feature_importance.sort_values(by='importance', ascending=False).head(15))
    del dtrain, dvalid
    gc.collect()

    preds = 0
    print('开始线上预测...')
    best_iters = int(lgb_model.best_iteration*1.2)
    all_train = pd.concat([train, valid], axis=0, ignore_index=True)
    dtrain = lgb.Dataset(all_train[feats],
                         label=all_train[target])
    del all_train
    gc.collect()
    lgb_model = lgb.train(params,
                          dtrain,
                          num_boost_round=best_iters,
                          verbose_eval=100,
                          )
    preds = lgb_model.predict(test[feats])
    return valid_preds, preds

def cbt_reg_model(train, valid, test, target):
    catboost_params = {
        'iterations': 10000,
        'learning_rate': 0.05,
        'eval_metric': 'RMSE',
        'task_type': 'GPU',
        'early_stopping_rounds': 200,
        'use_best_model': True,
        'verbose': 100,
        'depth': 10,
        'bootstrap_type': 'Bernoulli',
        'boosting_type': 'Plain',
        'subsample': 0.8
    }
    feats = [f for f in train.columns if f not in ['article_id', 'date', 'orders_3h_15h', 'week']]

    print('线下训练集样本数：%d' % (train.shape[0]))
    print('线下验证集样本数：%d' % (valid.shape[0]))
    print('特征个数：%d' % (len(feats)))
    print('开始线下验证......')
    dtrain = Pool(train[feats].values,
                  label=train[target])
    dvalid = Pool(valid[feats].values,
                  label=valid[target])
    cbt_model = CatBoostRegressor(**catboost_params)
    cbt_model.fit(dtrain, eval_set=dvalid)
    #     print(cbt_model.best_score_['validation'])
    score = cbt_model.best_score_['validation']['RMSE']
    valid_preds = cbt_model.predict(valid[feats].values)
    del dtrain, dvalid
    gc.collect()

    print('开始线上预测......')
    best_iters = int(cbt_model.best_iteration_ * 1.2)
    catboost_params = {
        'iterations': best_iters,
        'learning_rate': 0.05,
        'eval_metric': 'RMSE',
        'task_type': 'GPU',
        'early_stopping_rounds': 200,
        #         'use_best_model': True,
        'verbose': 100,
        'depth': 10,
        'bootstrap_type': 'Bernoulli',
        'boosting_type': 'Plain',
        'subsample': 0.8
    }
    all_train = pd.concat([train, valid], axis=0, ignore_index=True)

    dtrain = Pool(all_train[feats],
                  label=all_train[target])
    del all_train
    gc.collect()
    cbt_model = CatBoostRegressor(**catboost_params)
    cbt_model.fit(dtrain)
    del dtrain
    gc.collect()
    preds = cbt_model.predict(test[feats].values)
    return valid_preds, preds

def xgb_reg_model(train, valid, test, target):
    feats = [f for f in train.columns if f not in ['article_id', 'date', 'orders_3h_15h',
                                                   'week', 'baike_id_2h_price_diff_vs_mean', 'baike_id_2h_price_diff_ratio_vs_mean',
                                                   'url_price_diff_vs_mean', 'url_price_diff_ratio_vs_mean']]

    print('线下训练集样本数：%d' % (train.shape[0]))
    print('线下验证集样本数：%d' % (valid.shape[0]))
    print('特征个数：%d' % (len(feats)))
    print('开始线下验证......')
    clf = xgb.XGBRegressor(
        n_estimators=10000,
        max_depth=12,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.4,
        missing=-1,
        eval_metric='rmse',
        # USE CPU
        # nthread=4,
        # tree_method='hist'
        # USE GPU
        tree_method='gpu_hist'
    )
    xgb_model = clf.fit(train[feats], train[target],
                        eval_set=[(valid[feats], valid[target])],
                        verbose=100,
                        early_stopping_rounds=200)
    valid_preds = xgb_model.predict(valid[feats].values)
    #     del dtrain, dvalid
    gc.collect()

    print('开始线上预测......')
    best_iters = int(xgb_model.best_iteration * 1.2)
    clf = xgb.XGBRegressor(
        n_estimators=best_iters,
        max_depth=12,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.4,
        missing=-1,
        eval_metric='rmse',
        # USE CPU
        # nthread=4,
        # tree_method='hist'
        # USE GPU
        tree_method='gpu_hist'
    )
    all_train = pd.concat([train, valid], axis=0, ignore_index=True)

    xgb_model = clf.fit(all_train[feats],
                        all_train[target],
                        verbose=100)
    del all_train
    gc.collect()
    preds = xgb_model.predict(test[feats].values)
    return valid_preds, preds

def gen_submission(df, preds, sub_path):
    sub = df[['article_id']].copy()
    sub['orders_3h_15h'] = preds
    sub['orders_3h_15h'] = sub['orders_3h_15h'].apply(lambda x: 0 if x < 0 else x)
    sub.to_csv(sub_path+'sub.csv', index=False)

def main(data_path, sub_path):
    train_df, test_df = load_dataset(data_path) # 读取数据
    data = feature_extractor(train_df, test_df) # 生成特征
    train, valid, test = split_dataset(data=data, valid_start_date=110, gap=7) # 划分训练\验证集
    del train_df, test_df
    gc.collect()
    lgb_oof, lgb_preds = lgb_reg_model(train, valid, test, 'orders_3h_15h') # lightgbm
    cbt_oof, cbt_preds = cbt_reg_model(train, valid, test, 'orders_3h_15h')  # catboost
    xgb_oof, xgb_preds = xgb_reg_model(train, valid, test, 'orders_3h_15h')  # xgboost
    preds = 0.3*lgb_preds + 0.5*cbt_preds + 0.2*xgb_preds # 加权融合
    gen_submission(test, preds, sub_path) # 生成提交文件

if __name__ == '__main__':
    data_path = '../data/'
    sub_path = '../submission/'
    main(data_path, sub_path)
