# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
from IPython.display import display
import warnings

from copy import deepcopy as dcopy
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import math


class Binning():
    MISSING = "Missing"
    OTHERS = "Others"

    def __init__(self, max_cats=6, x=None, y=None, cuts=None, min_pct=0.01, postfix='woe', max_bins=8, dtype=None,
                 min_iv=0.01, q=150, **kwargs):
        """

        :param max_cats:
        :param x:
        :param y:
        :param cuts:
        :param min_pct:
        :param postfix: new column name to store feature_engineering x
            e.g. x="age",postfix ="bin" -> age_bin
            note: postfix ="" / None will inplace df[x] directly!
        :param max_bins:
        :param dtype:
        # :param b_woe:
        :param min_iv:
        :param kwargs:
        """
        self.dtype = dtype
        self.x = x
        self.y = y
        self.max_cats = max_cats
        self.min_pct = min_pct
        self.max_bins = max_bins
        self.postfix = postfix
        self.cuts = cuts
        # self.b_woe = b_woe
        self.min_iv = min_iv
        self.q = q

        bin_name = self.x
        if self.postfix:
            bin_name = bin_name + "_" + self.postfix
        self.bin_name = bin_name

        self.kwargs = kwargs
        if kwargs:
            print("unused params:", kwargs)

        self.iv_table = None

        self.woe_map = dict()
        self.missing_ratio = None
        self.unique_cnt = None
        self.ignore_reason = None
        self.bin_labels = None
        self.min_bin_cnt = None

        self.desc = None
        self.min_v = None
        self.max_v = None

    def cal_cuts(self, df, x, y, dtype=None):
        """

        :param df:
        :param x:
        :param y:
        :param dtype:
        :return:
        """
        self.ignore_reason = None

        # only one value except nan
        if 2 == self.unique_cnt and self.missing_ratio > 0:
            cuts = []
        else:
            if 'factor' == self.dtype:
                cuts = self.cal_factor_cuts(df=df, x=x, y=y, max_cats=self.max_cats, min_pct=self.min_pct)
            else:
                cuts = self.cal_numeric_cuts(df=df, x=x, y=y, max_bins=self.max_bins)

        # final hope
        # Maybe missing is meaningful,so we try cuts: missing and no missing
        # Whether missing is meanful is
        if len(cuts) == 0:
            if self.min_pct <= self.missing_ratio <= (1 - self.min_pct):
                cuts = [self.MISSING]
            else:
                self.ignore_reason = "No significant cuts"

        cuts_simple = []
        for v in cuts:
            if isinstance(v, list):
                v_ = []
                for vv in v:
                    if str(vv).endswith(".0"):
                        v_.append(int(float(vv)))
                    else:
                        v_.append(vv)
                cuts_simple.append(v_)
            else:
                if str(v).endswith(".0"):
                    cuts_simple.append(int(float(v)))
                else:
                    cuts_simple.append(v)

        self.cuts = cuts_simple
        return cuts_simple

    def get_cuts(self):
        """
        :return:
        """
        return self.cuts

    def fit(self, df, cuts=None, min_iv=None):
        """
        get cuts and iv-table
        :param df:
        :param x:
        :param y:
        :param dtype: 'numeric'/'factor'/None
            numeric: continue variable
            factor:  category variable
            None:    set dtype by type of type(x) automatically!
                object ->  factor
                non object -> numeric
        :param cuts:
            manual setting cuts or calculate cuts automatic
            support formats:
                numeric:
                    e.g. [0,4,6] -> <=0    <=6   >6
                    e.g. [0,4,[6,'Missing']] -> <=0 <=6 >6,Missing
                    e.g. [0,4,['Missing',6]] -> <=0 <=6,Missing >6
                factor:
                    e.g. [0,[1,2],[3,4,5,6,7,8],'Missing','Others']
                    e.g.[[1102,1109],110,[119,'Missing']] -> 1102,1109  110 119,Missing
                    e.g. ['男','女'] -> '男'   '女'
                    e.g.[['男','Missing'],'女'] -> '男,Missing'  '女'
            advice: you should not set it here,you'd best set it in the init func!
        :return:
        """

        print("fit 【 {} 】 ......".format(self.x))

        x = self.x
        y = self.y

        df = df[[x, y]].copy()
        if x != y and isinstance(df[x].dtype, pd.CategoricalDtype):
            df[x] = df[x].astype(str)

        if x not in df.columns:
            self.ignore_reason = "Not found in columns!"
            return

        if x == y:
            self.ignore_reason = "{} is Y".format(x)
            return

        self.missing_ratio = df[x].isna().mean()

        self.min_bin_cnt = self.min_pct if self.min_pct > 1 else self.min_pct * len(df)
        self.min_bin_cnt = round(self.min_bin_cnt)

        self.min_pct = max(round(self.min_bin_cnt / len(df), 4), 0.0001)

        if self.min_pct > 0.3:
            self.min_pct = 0.3
            warnings.warn("min_pct = {:.2%} is too large".format(self.min_pct))

        # determine dtype
        if self.dtype is None:
            self.dtype = 'factor' if 'object' == df[x].dtype else 'numeric'
        else:
            pass

        # statistic
        if 'numeric' == self.dtype:
            self.min_v = df[x].min()
            self.max_v = df[x].max()
        else:

            oc = df[x].value_counts().index
            if len(oc) > 0:
                self.max_v, self.min_v = oc[0], oc[-1]

        self.unique_cnt = df[x].unique().size
        if self.unique_cnt == 1:
            self.ignore_reason = "Only one unique value!"
            return

        # calculate cuts automatic

        if cuts:  # manual cuts
            self.cuts = cuts
        elif self.cuts:  # with old cuts
            pass
        else:  # no cuts
            self.cuts = self.cal_cuts(df, x, y, dtype=self.dtype)
        if self.ignore_reason:
            return

        if not isinstance(self.cuts, list):
            self.ignore_reason = 'No significant split!'
            return

        # feature_engineering
        self.binning(df)

        # failed to feature_engineering
        if self.ignore_reason:
            return 'failed to feature_engineering'

        # calculate woe
        self.cal_iv_table(df)
        iv = self.iv_table.iloc[-1, -1]
        if min_iv and iv < min_iv:
            self.ignore_reason = 'iv = {:.4f} <{:.4f}'.format(iv, min_iv)
            return

    def transform(self, df, df_out, b_woe=True):
        print("transform 【 {} 】...".format(self.x))
        if self.ignore_reason:
            raise Exception("can't transform {} for {} ".format(self.x, self.ignore_reason))
            # return
        self.binning(df, df_out)
        if b_woe:
            self.woe(df_out)

    def woe(self, df):
        # print("woe...")

        if not self.woe_map:
            # self.cal_iv_table(df, x=self.x, y=self.y)
            self.cal_iv_table(df)

        df[self.bin_name] = df[self.bin_name].map(self.woe_map)
        # display(df[self.bin_name].value_counts())
        return df[self.bin_name]

    def to_woe_sql(self):

        def to_int(vs):
            l = []
            for v in vs:
                try:
                    l.append(int(float(v)))
                except:
                    None
            return set(l)

        variable = self.x
        variable_trs = variable.strip("\t\t' ") + "_woe"

        c_l = []

        is_ratio = variable.strip("\t\n' ").startswith("r_")

        max_v = -math.inf
        min_v = math.inf
        max_woe = None
        min_woe = None
        max_bad_rate = None
        min_bad_rate = None

        null_woe = None
        null_bad_rate = None

        type_numeric = False
        cuts = self.cuts

        sql_whens = []
        for i, r in self.iv_table.iloc[:-1].iterrows():
            woe_i = r['WoE']
            bad_rate_i = r['BadRate']
            cut = r['Cutpoint']

            # numeric:
            # e.g.[0, 4, 6] -> <= 0 <= 6 > 6
            # e.g.[0, 4, [6, 'Missing']] -> <= 0 <= 6 > 6, Missing
            # e.g.[0, 4, 'Missing'] -> <= 0, <= 4, > 4, Missing
            # e.g.[0, 4, ['Missing', 6]] -> <= 0 <= 6, Missing > 6
            sql_when = ""
            if self.dtype == 'numeric':
                cut_cond = cut.split(",")
                cut_cond =[x for x in cut_cond if x != self.MISSING]

                if "<=" in cut or ">" in cut:
                    sql_when = "{:>20}  {:>8}".format(variable,cut_cond[0] )
                # if "<=" in cut:
                #     sql_when += "{:>20} <= {:>8}".format(variable,  )
                # elif ">" in cut:
                #     sql_when += "{:>20} >  {:>8}".format(variable, cuts[i - 1])

                if "Missing" == cut:
                    sql_when = "{:>20} is null or {} in ('','null','NULL','XY_RC_DATA_NULL','XY_RC_DATA_\\\\N')".format(
                        variable, variable)
                elif "Missing" in cut:
                    sql_when += " or " + "{:>20} is null or {} in ('','null','NULL','XY_RC_DATA_NULL','XY_RC_DATA_\\\\N')".format(
                        variable, variable)


            # factor:
            # e.g.[0, [1, 2], [3, 4, 5, 6, 7, 8], 'Missing', 'Others']
            # e.g.[[1102, 1109], 110, [119, 'Missing']] -> 1102, 1109
            # 110
            # 119, Missing
            # e.g.['男', '女'] -> '男'   '女'
            # e.g.[['男', 'Missing'], '女'] -> '男,Missing'  '女'
            else:
                if cut == self.OTHERS:
                    continue
                cut = ["'{}'".format(x) for x in cut.split(",") if x != 'Missing']
                if "Missing" == cut:
                    sql_when = "{:>20} is null or {} in ('','null','NULL','XY_RC_DATA_NULL','XY_RC_DATA_\\\\N')".format(
                        variable, variable)
                elif "Missing" in cut:
                    sql_when = "{:>20} in ({:<20})".format(variable, ",".join(cut))
                    sql_when += " or " + "{:>20} is null or {} in ('','null','NULL','XY_RC_DATA_NULL','XY_RC_DATA_\\\\N')".format(
                        variable, variable)
                else:
                    sql_when = "{:>20} in ({:<20})".format(variable, ",".join(cut))

            sql_when = "\twhen {} then  {:<20}  --{:.2%}\n".format(sql_when, woe_i, bad_rate_i)

            sql_whens.append(sql_when)

        if self.dtype != 'numeric' and self.OTHERS in cuts:
            mask_ = self.iv_table['Cutpoint'] == self.OTHERS
            woe_i = self.iv_table.loc[mask_, 'WoE'].values[0]
            bad_rate_i = self.iv_table.loc[mask_, 'BadRate'].values[0]
            # print(woe_i, bad_rate_i)
            sql_whens.append("\telse {:>20}  --{:.2%}".format(woe_i, bad_rate_i))
        else:
            sql_whens.append("\telse null  -- 异常值")
        sql = "case\n" + "".join(sql_whens) + "\nend as {}_woe, ".format(variable)

        return sql

    def binning(self, df, df_out=None):
        x = self.x
        cuts = self.cuts
        bin_name = self.bin_name
        dtype = self.dtype

        if df_out is None:
            df_out = df

        # missing is the only cuts
        if [self.MISSING] == cuts:
            mask_miss = df[x].isna()
            df_out[bin_name] = "~" + self.MISSING
            df_out.loc[mask_miss, bin_name] = self.MISSING
            return

        # check whether need to merge bins
        merge_bins = [x for x in cuts if isinstance(x, list)]

        # print(self.dtype)
        if 'numeric' == dtype:
            cuts_numeric = [x for x in cuts if not isinstance(x, list)]

            if merge_bins:
                assert 1 == len(merge_bins), "{} is not support! at most one merge group for numeric variable! ".format(
                    merge_bins)
                cuts_numeric.extend(
                    [x for x in merge_bins[0] if
                     x != self.MISSING])  # numeric feature_engineering can have at most one group bin
            cuts_numeric = list(set(cuts_numeric))
            cuts_numeric.sort()

            # filter valid x range
            # cuts_numeric = [x for x in cuts_numeric if min_v <= x <= max_v]
            if not cuts_numeric:
                self.ignore_reason = "nnmeric data cuts,must have a numeric".format(self.cuts)
                # self.ignore_reason = "cuts:{} not in [{},{}]".format(self.cuts, min_v, max_v)
                return
            # cuts  must increase monotonically 妈蛋，会有无穷大的情况
            min_v, max_v = df[x].min() - 1, df[x].max()

            if min_v < cuts_numeric[0]:
                cuts_numeric.insert(0, min_v)

            if max_v > cuts_numeric[-1]:
                cuts_numeric.append(self.max_v)
            # cuts_numeric.append(max(self.max_v, cuts_numeric[-1]) + 1)
            # cuts_numeric.sort()

            bin_labels = ["<={}".format(cuts_numeric[1])]

            if len(cuts_numeric) > 2:
                bin_labels.extend(["<={}".format(x) for i, x in enumerate(cuts_numeric[2:-1])])
                bin_labels.append(">{}".format(cuts_numeric[-2]))

            self.bin_labels = bin_labels

            # cut
            df_out[bin_name] = pd.cut(df[x], bins=cuts_numeric, labels=bin_labels, retbins=False).astype(str)

            if self.missing_ratio > 0:
                df_out[bin_name] = df_out[bin_name].replace('nan', self.MISSING)  # Nan have converted to 'nan'

        else:
            df_out[bin_name] = df[x].fillna(self.MISSING)

        # merge bins
        if merge_bins:
            if 'numeric' == dtype:
                map_dict = {x: x for x in bin_labels}
                # merge missing with v_merge
                v_merge = [x for x in merge_bins[0] if x != self.MISSING][0]

                v_merge_bin = bin_labels[cuts_numeric.index(v_merge) + (-1 if merge_bins[0][0] == self.MISSING else 0)]
                m_bin_name = v_merge_bin + "," + self.MISSING

                self.bin_labels[self.bin_labels.index(v_merge_bin)] = m_bin_name

                map_dict[self.MISSING] = m_bin_name
                map_dict[v_merge_bin] = m_bin_name
            # factor
            else:
                # single value bins
                map_dict = {cc: cc for cc in [c for c in cuts if not isinstance(c, list)]}

                # multi-value bins
                for m in merge_bins:
                    # truncation bin name with two many items
                    if False and len(m) > 5:
                        m_bin_name = ",".join([str(x) for x in m[:5]] + ["etc"])
                    else:
                        m_bin_name = ",".join([str(x) for x in m])

                    for c in m:
                        map_dict[c] = m_bin_name

                if self.OTHERS in cuts:
                    # Note: wo don't warning anything if new value occurred that not seeing in the train data
                    for c in set(df_out[bin_name].unique()) - set(map_dict.keys()):
                        map_dict[c] = self.OTHERS

            # replace with woe value
            df_out[bin_name] = df_out[bin_name].map(map_dict)

    def cal_iv_table(self, df):
        """
        :param df:
        :param x:
        :param y: 1:bad,0:good
        :return:
         Cutpoint CntRec CntGood CntBad CntCumRec CntCumGood CntCumBad PctRec GoodRate BadRate    Odds LnOdds     WoE     IV
        1   '女'  29143   28366    777     29143      28366       777 0.2953   0.9733  0.0267 36.5071 3.5975  0.3404 0.0293
        2   '男'  69533   66652   2881     98676      95018      3658 0.7047   0.9586  0.0414 23.1350 3.1413 -0.1158 0.0100
        3  Missing      0       0      0     98676      95018      3658 0.0000      NaN     NaN     NaN    NaN     NaN    NaN
        4    Total  98676   95018   3658        NA         NA        NA 1.0000   0.9629  0.0371 25.9754 3.2571  0.0000 0.0393
        """

        x = self.bin_name
        y = self.y

        df_total = df[y].agg(['count', 'sum', 'mean'])
        df_total[x] = 'Total'

        df = df.groupby(x)[y].agg(['count', 'sum', 'mean'])

        if 'numeric' == self.dtype and [self.MISSING] != self.cuts:
            sort_labels = dcopy(self.bin_labels)
            if self.MISSING in df.index:
                sort_labels.append(self.MISSING)

            sort_labels_exists = [x for x in sort_labels if x in df.index]

            # if len(sort_labels_exists) != len(sort_labels):
            #     warnings.warn(
            #         "Attention! Not all cuts bins exists!\niv bin labels:{}\nexists bins:{}".format(sort_labels,
            #                                                                                       sort_labels_exists))
            # order cuts
            df = df.loc[sort_labels_exists].reset_index()
        else:
            df.sort_values(by='mean', inplace=True)
            df = df.reset_index()

        df = df.append(df_total, sort=False)

        df = df.reset_index(drop=True)

        df.rename(columns={x: 'Cutpoint', 'count': 'CntRec', 'sum': 'CntBad', 'mean': 'BadRate'}, inplace=True)
        df['CntGood'] = df['CntRec'] - df['CntBad']
        idx_exc_last = df[:-1].index
        df.loc[idx_exc_last, 'CntCumRec'] = df.loc[idx_exc_last, 'CntRec'].cumsum().astype(int)
        df.loc[idx_exc_last, 'CntCumGood'] = df.loc[idx_exc_last, 'CntBad'].cumsum().astype(int)
        df.loc[idx_exc_last, 'CntCumBad'] = df.loc[idx_exc_last, 'CntBad'].cumsum().astype(int)

        df['PctRec'] = df['CntRec'] / df['CntRec'].iloc[-1]
        df['GoodRate'] = 1 - df['BadRate']
        #
        df['Odds'] = df['CntGood'] / df['CntBad']
        df['LnOdds'] = np.log(df['Odds'].clip(lower=0.001))
        df['WoE'] = (df['LnOdds'] - df['LnOdds'].iloc[-1]).round(4)  # ln(GA/BA) – ln(Gtotal/Btotal)
        df['IV'] = (df['CntGood'] / df['CntGood'].iloc[-1] - df['CntBad'] / df['CntBad'].iloc[-1]) * df['WoE']
        df.loc[df.index[-1], 'IV'] = df['IV'].sum()
        df = df[
            ['Cutpoint', 'CntRec', 'CntGood', 'CntBad', 'CntCumRec', 'CntCumGood', 'CntCumBad', 'PctRec', 'GoodRate',
             'BadRate', 'Odds', 'LnOdds', 'WoE', 'IV']]

        df['IV'] = df['IV'].round(4)
        df['Odds'] = df['Odds'].round(4)

        self.iv_table = df
        self.woe_map = dict(zip(df['Cutpoint'][:-1], df['WoE'][:-1]))
        # display(df)
        # print("{:<64}:{:.4f}".format(self.x, df.iloc[-1, -1]))
        return df.copy(deep=True)

    def cal_factor_cuts(self, df, x, y, max_cats=6, min_pct=0.01):
        min_pct = min_pct / len(df) if min_pct > 1 else min_pct

        df_t = df[x].fillna('Missing').value_counts().reset_index(name='count')
        df_t['ratio'] = df_t['count'] / df_t['count'].sum()
        mask_other_bins = (df_t['ratio'] < min_pct) | (df_t.index >= max_cats)
        cuts = [df_t[mask_other_bins]['index'].tolist()] + df_t[~mask_other_bins]['index'].tolist()

        return cuts

    def cal_numeric_cuts(self, df, x, y, max_bins, **kwargs):
        cuts = None
        return cuts

    def get_iv_table(self):
        if self.iv_table is None:
            return "no iv table!"
        else:
            return self.iv_table.copy()

    def plot(self):
        pass

    def get_iv(self):
        if self.iv_table is not None:
            return self.iv_table.iloc[-1, -1]
        else:
            return -1

    def quantile_stat(self, df, x, y):

        df = df.sort_values(by=x)
        q = self.q
        if self.unique_cnt <= self.max_bins:
            init_cuts = df[x].unique()
            init_cuts = [x for x in init_cuts if not np.isnan(x)]
            init_cuts.sort()
        else:
            # init_cuts = df[x].quantile([(i + 1) / q for i in range(q - 1)], interpolation='higher').values.tolist()
            step = max(round(len(df) / q), 1)
            init_cuts = df[x][::step].values.tolist()
            init_cuts = list(set([x for x in init_cuts if not np.isnan(x)]))
            init_cuts.sort()

        min_v = self.min_v - 1
        if min_v < init_cuts[0]:
            init_cuts.insert(0, min_v)
        if self.max_v > init_cuts[-1]:  # inf+1 = inf
            init_cuts.append(self.max_v)

        # print("init_cuts:", init_cuts)

        df_bins = df.groupby([pd.cut(df[x], bins=init_cuts)])[y].agg(
            ['count', 'sum', 'mean']).rename(columns={'sum': 'bad'})

        df_bins = df_bins[df_bins['count'] > 0]
        df_bins['good'] = df_bins['count'] - df_bins['bad']
        df_bins.sort_index(inplace=True)

        return df_bins

    def visual_woe(self, figsize=(12, 6), save_file=None, title=None, rot=None):
        if self.iv_table is None:
            raise Exception("{} have no iv table!".format(self.x))
        if rot is None:
            if self.iv_table['Cutpoint'].map(lambda x: len(str(x))).max() > 12:
                rot = 45
            else:
                rot = 0

        # df_plot = self.iv_table[['Cutpoint', 'PctRec', 'BadRate']].iloc[:-1, :]
        # ax = df_plot.plot(kind='bar'
        #                   , x='Cutpoint'
        #                   , figsize=figsize
        #                   , fontsize=15
        #                   , secondary_y='BadRate'
        #                   , rot=rot
        #                   , legend=True
        #                   , title=title
        #                   , zorder=10
        #                   )
        df_plot = self.iv_table[['Cutpoint', 'BadRate', 'PctRec']].iloc[:-1, :]
        ax = df_plot.plot(kind='bar'
                          , x='Cutpoint'
                          , figsize=figsize
                          , fontsize=15
                          , secondary_y='PctRec'
                          , rot=rot
                          , legend=True
                          , title=title
                          , zorder=10
                          )
        ax.set_xlabel(self.x, fontsize=16)
        ax.margins(y=0.1)
        # plt.title(self.x, fontsize=16)

        # refer :https://matplotlib.org/examples/api/barchart_demo.html

        fontsize = 13 if len(df_plot) >= 6 else 14
        axs = plt.gcf().get_axes()
        for c, ax in enumerate(axs):
            color = 'black' if c == 0 else 'gray'
            for i in ax.patches:
                ax.text(i.get_x() + i.get_width() / 2, i.get_height(),
                        "{:.2%}".format(i.get_height(), 2), fontsize=fontsize, color=color,
                        rotation=0, ha='center', va='bottom', zorder=100)


class ChiBinning(Binning):
    """similar to pandas.qcut"""

    # https://blog.csdn.net/bitcarmanlee/article/details/52279907
    chi_confidence_table_2_class_d = {0.95: 3.84, 0.9: 2.71, 0.8: 1.64, 0.7: 1.07, 0.5: 0.46}

    def __init__(self, auto_missing=True, q=20, confidence=0.8, **kwargs):
        super().__init__(**kwargs)
        self.auto_missing = auto_missing
        self.min_chi_thr = self.chi_confidence_table_2_class_d[confidence]
        self.q = q

    def cal_numeric_cuts(self, df, x, y, max_bins, **kwargs):
        """
        Note:
            if Missing
        :param df:
        :param x:
        :param y:
        :param max_bins:
        :param kwargs:
        :return:
        """
        # 20  equal count bins
        df = df[[x, y]].copy()

        # df = df_all[df_all[y] >= 0][[x, y]].copy()
        df_bins = self.quantile_stat(df=df, x=x, y=y)
        # merge too small bins
        pass
        # 我还没写呢,其实，如果样本量实在太少，卡方也会特别小，就会自动合并，所以呢，不着急

        # init chi
        df_total = df_bins + df_bins.shift(-1)  # all
        df_expect = df_total[['good', 'bad']].mul((df_bins['count'] / df_total['count']), axis=0)
        df_bins['chi'] = ((df_bins[['good', 'bad']] - df_expect) ** 2 / df_expect.fillna(0.5)).sum(axis=1)

        if len(df_bins) > 1:
            min_chi_bin_idx = df_bins['chi'].iloc[:-1].idxmin()
            min_chi_bin_loc = df_bins.index.get_loc(min_chi_bin_idx)
            min_chi = df_bins.loc[min_chi_bin_idx, 'chi']
        else:
            min_chi = 0

        # display(df_bins)
        while len(df_bins) >= 3 and (len(df_bins) > max_bins or min_chi < self.min_chi_thr):

            df_bins.iloc[min_chi_bin_loc] = df_total.iloc[min_chi_bin_loc]
            merge_idx = pd.Interval(df_bins.index[min_chi_bin_loc].left, df_bins.index[min_chi_bin_loc + 1].right)

            df_bins = df_bins.rename(index={min_chi_bin_idx: merge_idx})
            # drop merged bin
            df_bins = df_bins[df_bins.index != df_bins.index[min_chi_bin_loc + 1]].copy()

            # update
            df_bins_up = df_bins[max(min_chi_bin_loc - 1, 0):min(min_chi_bin_loc + 2, len(df_bins))].copy()

            df_total_up = df_bins_up + df_bins_up.shift(-1)  # all
            df_expect_up = df_total_up[['good', 'bad']].mul((df_bins_up['count'] / df_total_up['count']), axis=0)
            df_bins_up['chi'] = ((df_bins_up[['good', 'bad']] - df_expect_up) ** 2 / df_expect_up.fillna(0.5)).sum(
                axis=1)
            # ugly code ...
            for idx in df_bins_up.index[:-1]:
                df_bins.loc[idx] = df_bins_up.loc[idx]

            df_bins['mean'] = df_bins['bad'] / df_bins['count']

            # find min chi bin to merge

            min_chi_bin_idx = df_bins['chi'].iloc[:-1].idxmin()
            min_chi_bin_loc = df_bins.index.get_loc(min_chi_bin_idx)
            min_chi = df_bins.loc[min_chi_bin_idx, 'chi']
        #             print("min_chi:", min_chi)
        #             display(df_bins)

        if min_chi < self.min_chi_thr:
            if self.missing_ratio >= self.min_pct:
                cuts = [df[x].max()]

                return cuts
        else:
            cuts = [x.right for x in df_bins.index[:-1]]

        # auto-merge  Missing
        if self.auto_missing:

            na_bin = df[df[x].isnull()][y].agg(['count', 'sum', 'mean'])
            # na_bin = pd.Series([20, 10, 0.92], ['count', 'sum', 'mean'])
            na_bin.index = ['count', 'bad', 'mean']
            na_bin['good'] = na_bin['count'] - na_bin['bad']
            # display(na_bin)

            min_cnt = min(len(df) * self.min_pct, 2000)

            if 0 < na_bin['count'] < min_cnt:
                bad_ratio_diff = (df_bins['mean'] - na_bin['mean']).abs()
                # merge to first or last bin
                merge_bin_loc = 0 if bad_ratio_diff.iloc[0] < bad_ratio_diff.iloc[-1] else -1

                # merge
                df_bins.iloc[merge_bin_loc] = df_bins.iloc[merge_bin_loc] + na_bin

                if 0 == merge_bin_loc:
                    cuts = [[self.MISSING, cuts[0]]] + cuts[1:]
                else:
                    cuts = cuts[:-1] + [[cuts[-1], self.MISSING]]

        return cuts


class QuantBinning(Binning):
    """similar to pandas.qcut"""

    def __init__(self, q, **kwargs):
        super().__init__(**kwargs)
        self.q = q  # Number of quantiles
        self.max_bins = q

    def cal_numeric_cuts(self, df, x, y, max_bins, **kwargs):
        if kwargs and 'q' in kwargs:
            self.q = kwargs['q']
            self.max_bins = self.q

        cuts = df[x].quantile([(i + 1) / self.q for i in range(self.q - 1)]).values.tolist()
        cuts = list(set(cuts))
        cuts.sort()
        return cuts


class VarBinning(Binning):
    """similar to pandas.qcut"""

    def __init__(self, auto_missing=True, q=150, count_score_base=2500, **kwargs):
        super().__init__(**kwargs)
        self.auto_missing = auto_missing
        self.q = q
        self.count_score_base = count_score_base

    def var_split(self, df_bins, max_bins):

        df_bins['cum_count'] = df_bins['count'].cumsum(axis=0)
        df_bins['cum_y2'] = df_bins['bad'].cumsum(axis=0)

        df_bins['r_cum_y2'] = df_bins['cum_y2'].iloc[-1] - df_bins['cum_y2']
        df_bins['r_cum_count'] = df_bins['cum_count'].iloc[-1] - df_bins['cum_count']

        # display(df_bins)
        var = df_bins['cum_y2'].iloc[-1] - df_bins['cum_y2'].iloc[-1] / df_bins['cum_count'].iloc[-1] * \
              df_bins['cum_y2'].iloc[-1]

        df_left, df_right, var_decrease = self.var_split_step(df_bins)
        cuts_df = [(df_left, df_right, var_decrease)]

        # print("var", var)
        min_dec = var_decrease / 10

        # print("init var_decrease:{:.4f},min_dec:{:.1f}".format(var_decrease, min_dec))

        # greedy split by max variance decrease
        cuts = []
        while len(cuts_df) < max_bins and var_decrease > min_dec:
            df_left, df_right, var_decrease = cuts_df.pop(0)

            if isinstance(df_left.index.dtype, pd.CategoricalDtype):
                cuts.append(df_left.index[-1].right)
            else:
                cuts.append(df_left.index[-1])
            # print(cuts)
            # print("var_decrease:{:.1f}".format(var_decrease))

            ll, lr, ld = self.var_split_step(df_left)
            cuts_df.append([ll, lr, ld])

            rl, rr, rd = self.var_split_step(df_right)
            cuts_df.append([rl, rr, rd])

            cuts_df.sort(key=lambda x: x[-1], reverse=True)
            var_decrease = cuts_df[0][-1]

            min_dec = max(min_dec / 2.38, var / 10000)

            # print("var_decrease next:{:.1f},min_dec:{:.4f}".format(var_decrease,min_dec))
        cuts.sort()

        return cuts, cuts_df, var, var_decrease

    def cal_factor_cuts(self, df, x, y, max_cats=6, min_pct=0.01):

        df_bins = df.groupby(df[x])[y].agg(['count', 'sum', 'mean']).rename(columns={'sum': 'bad'})
        df_bins['good'] = df_bins['count'] - df_bins['bad']

        df_lf = df_bins[df_bins['count'] < self.min_bin_cnt]
        df_hf = df_bins[~(df_bins['count'] < self.min_bin_cnt)]
        df_hf = df_hf.sort_values(by='mean').reset_index()

        cuts = []
        if len(df_hf) > 0:
            cuts_, cuts_df, var, var_decrease = self.var_split(df_hf.drop(columns=[x]), max_bins=max_cats)
            if len(cuts_) > 0:
                last_idx = cuts_[0] + 1

                cuts.append(df_hf[x][:last_idx].tolist())
                for v in cuts_[1:]:
                    cuts.append(df_hf[x][last_idx:v + 1].tolist())
                    last_idx = v + 1

                cuts.append(df_hf[x][last_idx:].tolist())

        #  no high frequency value no meaning
        if len(df_hf) > 0 and len(df_lf) > 0:
            # try to split by frequency (low and high )
            if len(cuts) == 0:
                cuts.append(df_hf[x].tolist())
                cuts.append(self.OTHERS)
            else:
                # merge low frequency values
                cuts.append(self.OTHERS)

        if self.missing_ratio > 0:
            if (self.min_pct <= self.missing_ratio <= (1 - self.min_pct)) or self.auto_missing is False:
                cuts.append(self.MISSING)
            else:
                # 最接近的bin
                pass

        # print("cuts:", cuts)

        return cuts
        # mask_other_bins = (df_t['ratio'] < self.min_pct) | (df_t.index >= max_cats)
        # cuts = [df_t[mask_other_bins]['index'].tolist()] + df_t[~mask_other_bins]['index'].tolist()

    def cal_numeric_cuts(self, df, x, y, max_bins, **kwargs):
        """
        Note:
            if Missing
        :param df:
        :param x:
        :param y:
        :param max_bins:
        :param kwargs:
        :return:
        """
        # get class,attribute table
        # df = df[[x, y]].copy()

        df_bins = self.quantile_stat(df=df, x=x, y=y)

        cuts, cuts_df, var, var_decrease = self.var_split(df_bins, max_bins)

        if len(cuts) == 0:  # No significant cuts!
            return cuts

        # auto-merge  Missing
        if self.auto_missing and self.missing_ratio < self.min_pct:
            na_bin = df[df[x].isnull()][y].agg(['count', 'sum', 'mean'])
            min_cnt = self.min_bin_cnt

            if 0 < na_bin['count'] < min_cnt:
                cuts_df.sort(key=lambda x: x[1].index[-1].right)
                head = cuts_df[0]
                tail = cuts_df[-1]

                # 后面再改成 cum_bad吧。。。
                head_bad_ratio = head[0].iloc[-1] + head[1].iloc[-1]
                head_bad_ratio = head_bad_ratio['cum_y2'] / head_bad_ratio['cum_count']

                tail_bad_ratio = tail[0].iloc[-1] + tail[1].iloc[-1]

                tail_bad_ratio = tail_bad_ratio['cum_y2'] / tail_bad_ratio['cum_count']

                if abs(head_bad_ratio - na_bin['mean']) < abs(tail_bad_ratio - na_bin['mean']):
                    cuts = [[self.MISSING, cuts[0]]] + cuts[1:]
                else:
                    cuts = cuts[:-1] + [[cuts[-1], self.MISSING]]

        # print("cal_numeric_cuts cuts:", cuts)

        return cuts

    def var_split_step(self, df_bins):

        if len(df_bins) <= 1:
            return df_bins, df_bins, 0

            # calculate variance descrese
        # if cum count is two big ,then cum*2 is easy overflow, so we div cum_count first then multi it again !
        # df_bins['var_s'] = df_bins['cum_y_2']/ df_bins['cum_count'] + df_bins['r_cum_y_2']/ df_bins['r_cum_count']

        var = df_bins['cum_y2'].iloc[-1] / df_bins['cum_count'].iloc[-1] * df_bins['cum_y2'].iloc[-1]

        df_bins['var_desc'] = (df_bins['cum_y2'] / df_bins['cum_count'] * df_bins['cum_y2'] + df_bins['r_cum_y2'] /
                               df_bins['r_cum_count'] * df_bins['r_cum_y2']) - var

        # latest bin size
        mask = (df_bins['cum_count'] > self.min_bin_cnt) & (df_bins['r_cum_count'] > self.min_bin_cnt)

        if mask.sum() == 0:
            return df_bins, df_bins, 0

        if self.count_score_base and self.count_score_base > 0:
            df_bins['count_score'] = df_bins[['cum_count', 'r_cum_count']].min(axis=1).apply(
                lambda x: 1 / (1 + np.exp(-x / self.count_score_base)))
        else:
            df_bins['count_score'] = 1

        # normalize
        df_bins['var_score'] = df_bins['var_desc'] * df_bins['count_score']

        # display(df_bins)

        var_decrease = df_bins[mask]['var_desc'].max()

        min_var_bin_idx = df_bins[mask]['var_score'].idxmax()
        min_var_bin_loc = df_bins.index.get_loc(min_var_bin_idx) + 1

        df_left = df_bins[:min_var_bin_loc].copy()

        df_right = df_bins[min_var_bin_loc:].copy()
        df_right = df_right - df_left.iloc[-1, :]

        df_left['r_cum_y2'] = df_left['cum_y2'].iloc[-1] - df_left['cum_y2']
        #     df_left['r_cum_y_2'] = df_left['cum_y_2'][-1] - df_left['cum_y_2']
        df_left['r_cum_count'] = df_left['cum_count'].iloc[-1] - df_left['cum_count']

        df_right['r_cum_y2'] = df_right['cum_y2'].iloc[-1] - df_right['cum_y2']
        #     df_right['r_cum_y_2'] = df_right['cum_y_2'][-1] - df_right['cum_y_2']
        df_right['r_cum_count'] = df_right['cum_count'].iloc[-1] - df_right['cum_count']

        return df_left, df_right, var_decrease


class BinningManager():

    def __init__(self, min_iv=0.01):
        self.min_iv = min_iv
        self.cols_se = []
        self.binning_d = dict()

        self.x_conf = list()

        self.method = None

        self.y = None
        self.max_cats = None
        self.min_pct = None
        self.postfix = None
        self.max_bins = None
        self.b_woe = None

        self.default_kwargs = dict()

        self.ignore_x_d = dict()
        self.update_l = list()

    def create_binning(self, x, x_conf):

        method = self.method
        kwargs = dcopy(self.default_kwargs)

        if x_conf:
            if "method" in x_conf:
                method = x_conf["method"]
                x_conf.pop("method")

            # cover default params
            for k, v in x_conf.items():
                kwargs[k] = v
        else:
            kwargs['x'] = x

        if 'var' == method:
            bin = VarBinning(**kwargs)
        elif 'chi' == method:
            bin = ChiBinning(**kwargs)
        elif 'quant' == method:
            bin = QuantBinning(**kwargs)

        return bin

    def fit(self, df, y, x_conf, method='chi', max_cats=6, min_pct=0.01, postfix='woe', max_bins=8, cache_file=None,
            **kwargs):

        self.method = method
        self.x_conf = x_conf

        # feature_engineering params
        self.y = y
        self.max_cats = max_cats
        self.min_pct = min_pct
        self.postfix = postfix
        self.max_bins = max_bins
        self.cache_file = cache_file

        # self.b_woe = b_woe

        self.default_kwargs = {
            'y': y,
            'max_cats': max_cats,
            'min_pct': min_pct,
            'postfix': postfix,
            'max_bins': max_bins,
        }
        self.default_kwargs.update(kwargs)

        # filter illegal records
        print("df size :{} ".format(len(df)))

        assert is_numeric_dtype(df[self.y]), "{} must numeric!".format(self.y)
        assert 2 == (~np.isnan(df[self.y].unique())).sum(), "{} must be binary!".format(self.y)

        print("df size after filtered by y:{} ".format(len(df)))

        # clear all
        self.binning_d = dict()
        self.ignore_x_d = dict()

        # init
        for conf in self.x_conf:
            x, conf = (conf, None) if isinstance(conf, str) else (conf['x'], conf)
            self.binning_d[x] = self.create_binning(x=x, x_conf=conf)

        # feature_engineering
        for x, bin in self.binning_d.items():
            bin.fit(df, min_iv=self.min_iv)

            if bin.ignore_reason:
                print(x, ":", bin.ignore_reason)
                self.ignore_x_d[x] = bin.ignore_reason
                continue

            print("IV:{:.4f}".format(bin.get_iv()))
            if len(self.binning_d) < 20:
                display(bin.get_iv_table())

        self.cache()

    def cache(self):
        if self.cache_file:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self, f)

    def transform(self, df, inc_cols=[], exc_cols=[]):
        for x, bin in self.binning_d.items():
            if inc_cols and x not in inc_cols:
                continue
            if x in exc_cols:
                continue
            if bin.ignore_reason:
                print(x, ":", bin.ignore_reason)
                self.ignore_x_d[x] = bin.ignore_reason
                continue
            if self.b_woe:
                bin.transform(df)
        return df

    def fit_update(self, df, x_conf):

        binning_d_old = self.binning_d
        binning_d_new = dict()
        x_conf_old = self.x_conf

        print("X cnt:{}".format(len(x_conf)))

        update_l = []
        for it in x_conf:
            x, conf = (it, None) if isinstance(it, str) else (it['x'], it)
            try:
                x_conf_old.index(it)
                # no change -> copy
                binning_d_new[x] = binning_d_old[x]
            except ValueError:
                # change
                update_l.append(x)
                binning_d_new[x] = self.create_binning(x=x, x_conf=conf)
        self.update_l = update_l

        print("\n----- 【Update】(add/change) X cnt:{} -----\n{}".format(len(update_l), ",".join(update_l)))
        add_l = set(binning_d_new.keys()) - set(binning_d_old)
        print("\n----- 【Add】 X cnt:{} -----\n{}".format(len(add_l), ",".join(add_l)))
        del_l = set(binning_d_old.keys()) - set(binning_d_new.keys())
        print("\n----- 【Delete】 X cnt:{} -----\n{}".format(len(del_l), ",".join(del_l)))

        self.x_conf = x_conf
        self.binning_d = binning_d_new

        # feature_engineering

        # if df._is_copy:
        #     df = df.copy()

        for x in update_l:
            bin = self.binning_d[x]
            bin.fit(df, min_iv=self.min_iv)

            if x in self.ignore_x_d.keys():
                self.ignore_x_d.pop(x)

            if bin.ignore_reason:
                print(x, ":", bin.ignore_reason)
                self.ignore_x_d[x] = bin.ignore_reason
                continue

            # if self.b_woe:
            #     bin.transform(df)

            print("IV:{:.4f}".format(bin.get_iv()))

        self.cache()

    def transform_update(self, df, df_out):
        if not self.update_l:
            return df
        # if df._is_copy:
        #     df = df.copy()
        #     warnings.warn("df is copy of a slice from a DataFrame!!! "
        #                   "You'd best pass a copy df after slice to accelerate transform!")
        # bin_cols = []
        for x in self.update_l:
            bin = self.binning_d[x]
            if x in self.ignore_x_d:
                continue
            bin.transform(df, df_out, b_woe=self.b_woe)
            # bin_cols.append(bin.bin_name)

        # df_out[bin_cols] = df[bin_cols]
        return df_out

    def transform(self, df, b_woe=True, keep_cols=[], inc_cols=[], exc_cols=[]):

        self.b_woe = b_woe

        cols_inc_not_find = set(keep_cols) - set(df.columns)
        assert len(cols_inc_not_find) == 0, "columns {} not find in df".format(",".join(cols_inc_not_find))

        df_out = df[keep_cols].copy()

        # is_copy = df._is_copy
        # if is_copy:
        #     init_cols = list(set(self.binning_d.keys()) | set(x_inc))
        #     df = df[init_cols].copy()
        #     warnings.warn("df is copy of a slice from a DataFrame!!! "
        #                   "You'd best pass a copy df after slice to accelerate transform!")
        # feature_engineering
        for x, bin in self.binning_d.items():

            if inc_cols and x not in inc_cols:
                continue
            if x in exc_cols:
                continue

            if x in self.ignore_x_d:
                continue
            bin.transform(df, df_out, b_woe=b_woe)
            # x_inc.append(bin.bin_name)
        # if is_copy:
        #     return df[x_inc]
        # else:
        #     return df[x_inc].copy()
        return df_out

    def iv_report(self):
        l = []
        for x, b in self.binning_d.items():
            l.append([x, 'n' if 'numeric' == b.dtype else 'f', b.missing_ratio, b.unique_cnt, b.max_v, b.min_v, b.cuts,
                      b.get_iv(), b.ignore_reason])
        iv_report_df = pd.DataFrame(l,
                                    columns=['x', 'type', 'miss ratio', 'unique cnt', 'max|top1', 'min|top2', 'cuts',
                                             'iv',
                                             'ignore reason'])
        iv_report_df = iv_report_df.sort_values(by='iv', ascending=False).reset_index(drop=True)
        iv_report_df
        return iv_report_df

    def save_woe(self, directory=None, x_se=None, figsize=(9, 6)):

        plt.ioff()

        if directory:
            if directory.endswith("/"):
                pass
            else:
                directory = directory + "/"

        keys = self.binning_d.keys()
        if x_se:
            keys = set(keys) & set(x_se)
        for k in keys:
            v = self.binning_d.get(k)
            if v.iv_table is not None:
                v.visual_woe(figsize=figsize)
                name = r"{}{}({}).png".format(directory, v.x.replace("/", "_o_"), v.dtype)
                plt.savefig(name, bbox_inches='tight', dpi=150)
                plt.close()
        plt.ion()
