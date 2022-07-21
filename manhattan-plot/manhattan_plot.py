import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib as mpl
from scipy.stats import chi2


DARK_CHR_COLOR = '#5841bf'
LIGHT_CHR_COLOR = '#648fff'
NOVEL_HIT_COLOR = '#dc267f'
NOVEL_TABLE_COLOR = '#eb7fb3'
REP_HIT_COLOR = '#ffbb00'
REP_TABLE_COLOR = '#ffdc7a'
FIFTH_COLOR = '#d45c00'
TABLE_HEAD_COLOR = '#9e9e9e'
COLOR_MAP = 'turbo_r'
CHR_POS_ROUND = 5E4

# https://en.wikipedia.org/wiki/Human_genome
CHR_LENGTHS = {1: 248956422,
               2: 242193529,
               3: 198295559,
               4: 190214555,
               5: 181538259,
               6: 170805979,
               7: 159345973,
               8: 145138636,
               9: 138394717,
               10: 133797422,
               11: 135086622,
               12: 133275309,
               13: 114364328,
               14: 107043718,
               15: 101991189,
               16: 90338345,
               17: 83257441,
               18: 80373285,
               19: 58617616,
               20: 64444167,
               21: 46709983,
               22: 50818468,
               23: 156040895}


# noinspection SpellCheckingInspection
class ManhattanPlot:
    # Attributes

    df = None
    thinned = None

    sig = 5E-8
    sig_line = 5E-8
    sug = 1E-5
    annot_thresh = 5E-8

    annotate = True
    signal_color_col = None
    twas_color_col, twas_updown_col = None, None

    ld_block = 4E5
    plot_x_col = 'ROUNDED_Y'
    plot_y_col = 'ROUNDED_X'
    chr_ticks = []
    max_x, max_y = 10, 10

    vertical = True
    invert = False
    merge_genes = False
    max_log_p = None

    fig, base_ax, table_ax, cbar_ax = None, None, None, None
    annot_list = []
    spec_genes = []

    def __init__(self, file_path, test_rows=None, title='Manhattan Plot'):
        """
        Constructor for ManhattanPlot object
        :param file_path: Path to summary statistics
        :type file_path: str
        :param test_rows: Number of rows to load
        :type test_rows: integer or None
        :param title: Title to be used for the plot
        :type title: str
        """
        self.path = file_path
        self.title = title
        self.test_rows = test_rows

    def load_data(self, delim='\s+'):
        """
        Reads data from the summary statistics file
        :param delim: Delimiter of the table file (common are '\t', ' ', ',')
        :type delim: str
        """
        if '.pickle' in self.path:
            self.df = pd.read_pickle(self.path).reset_index()
            if self.test_rows is not None:
                self.df = self.df.sort_values(by=['#CHROM']).iloc[:int(self.test_rows)]
        else:
            self.df = pd.read_table(self.path, sep=delim, nrows=self.test_rows)

        print('Loaded', len(self.df), 'Rows')
        print(self.df.columns)

    def clean_data(self, col_map=None, logp=None):
        """
        Edits/reformats the loaded table to make it compatible with plotting code
        :param col_map: Dictionary mapping existing columns to required columns: #CHROM, POS, ID, and P
        :type col_map: dict
        :param logp: Column with -log10 p-values if there is no P column
        :type logp: str or None
        """
        if col_map is not None:
            if 'P' in [v for k, v in col_map.items()] and 'P' in self.df.columns:
                self.df = self.df.drop(columns='P')

            col_map = {k: v for k, v in col_map.items() if k in self.df.columns}
            self.df = self.df.rename(columns=col_map)

        if logp is not None:
            self.df['P'] = 10 ** - self.df[logp]

        # df = df[df['#CHROM'] != 'X']
        chromosomes = list(range(1, 23))
        chromosomes.extend([str(i) for i in range(1, 23)])
        chromosomes.append('X')
        self.df = self.df[self.df['#CHROM'].isin(chromosomes)]
        self.df['#CHROM'] = self.df['#CHROM'].replace('X', 23)
        self.df['#CHROM'] = self.df['#CHROM'].astype(int)

        self.df['POS'] = self.df['POS'].astype(int)
        self.df = self.df.sort_values(by=['#CHROM', 'POS'])
        self.df['ID'] = self.df['ID'].fillna('')

        self.df['P'] = self.df['P'].replace(0, min(self.df['P']) / 100)

    def check_data(self):
        """
        Prints the beginning and end of the data table (required columns only) for sanity-checking
        """
        print(self.df.head()[['#CHROM', 'POS', 'P', 'ID']])
        print(self.df.tail()[['#CHROM', 'POS', 'P', 'ID']])
        print(len(self.df))

    def add_annotations(self, annot_df: pd.DataFrame, extra_cols=[]):
        annot_cols = ['#CHROM', 'POS', 'ID']
        annot_cols.extend(extra_cols)
        self.df = self.df.drop(columns='ID_y', errors='ignore')
        self.df = self.df.merge(annot_df[annot_cols], on=['#CHROM', 'POS'], how='left')
        self.df['ID_x'].update(self.df['ID_y'])
        self.df = self.df.rename(columns={'ID_x': 'ID'})

    def update_plotting_parameters(self, annotate='', signal_color_col='', twas_color_col='', twas_updown_col='', sig='', sug='', annot_thresh='', ld_block='', vertical='', max_log_p='', invert='', merge_genes=''):
        self.annotate = self.__update_param(self.annotate, annotate)
        self.ld_block = self.__update_param(self.ld_block, ld_block)

        self.signal_color_col = self.__update_param(self.signal_color_col, signal_color_col)
        self.twas_updown_col = self.__update_param(self.twas_updown_col, twas_updown_col)
        self.twas_color_col = self.__update_param(self.twas_color_col, twas_color_col)

        self.sig = self.__update_param(self.sig, sig)
        self.sug = self.__update_param(self.sug, sug)
        self.annot_thresh = self.__update_param(self.annot_thresh, annot_thresh)
        self.max_log_p = self.__update_param(self.max_log_p, max_log_p)

        self.vertical = self.__update_param(self.vertical, vertical)
        self.plot_x_col = 'ROUNDED_Y' if self.vertical else 'ROUNDED_X'
        self.plot_y_col = 'ROUNDED_X' if self.vertical else 'ROUNDED_Y'
        self.invert = self.__update_param(self.invert, invert)
        self.merge_genes = self.__update_param(self.merge_genes, merge_genes)

    def check_plotting_parameters(self):
        params = {'Significance Threshold': self.sig,
                  'Suggestive Threshold': self.sug,
                  'Annotation Threshold': self.annot_thresh,
                  'LD Block Width': self.ld_block,
                  'Annotating?': self.annotate,
                  'Orientation': 'Vertical' if self.vertical else 'Horizontal',
                  'Maximum Neg. Log P-Val': self.max_log_p}

        if self.signal_color_col is not None:
            params['Edge Color Column'] = self.signal_color_col
        print(params)

    def get_thinned_data(self, log_p_round=2):
        if 'ABS_POS' not in self.df.columns:
            self.df['ABS_POS'] = self.__get_absolute_positions(self.df)

        self.thinned = self.df.copy()
        self.thinned['ROUNDED_X'] = self.thinned['ABS_POS'] // CHR_POS_ROUND * CHR_POS_ROUND
        self.thinned['ROUNDED_Y'] = pd.Series(-np.log10(self.thinned['P'])).round(log_p_round)  # round to 2 decimals
        self.thinned = self.thinned.sort_values(by='P').drop_duplicates(subset=['ROUNDED_X', 'ROUNDED_Y'])
        print(len(self.thinned), 'Variants After Thinning')

    def print_hits(self):
        df = self.df.set_index('ID')
        sortedDF = df[df['P'] < self.sug].sort_values(by='P', ascending=True)
        sortedDF = sortedDF[~sortedDF.index.duplicated(keep='first')]

        keepCols = ['#CHROM', 'POS', 'P', 'ABS_POS']
        printCols = ['#CHROM', 'POS', 'P']

        if self.signal_color_col is not None:
            keepCols.append(self.signal_color_col)
            printCols.append(self.signal_color_col)

        if self.twas_color_col is not None:
            keepCols.append(self.twas_color_col)
            printCols.append(self.twas_color_col)

        if self.twas_updown_col is not None:
            keepCols.append(self.twas_updown_col)
            printCols.append(self.twas_updown_col)

        sigDF = sortedDF.loc[sortedDF['P'] <= self.sig, keepCols]

        print('Significant:')
        print('\n'.join(self.__fmt_print_rows(sigDF[printCols])))
        print('')

        print('\nSuggestive:')
        print('\n'.join(self.__fmt_print_rows(sortedDF.loc[sortedDF['P'] > self.sig, printCols])))
        print('')

    def plot_data(self, with_table=True):
        self.__config_axes(with_table=with_table)

        if self.vertical:
            self.base_ax.set_yticks(self.chr_ticks[0])
            self.base_ax.set_yticklabels(self.chr_ticks[1])
            if self.invert:
                self.base_ax.yaxis.set_label_position('right')
                self.base_ax.yaxis.tick_right()
        else:
            self.base_ax.set_xticks(self.chr_ticks[0])
            self.base_ax.set_xticklabels(self.chr_ticks[1])
            if self.invert:
                self.base_ax.xaxis.set_label_position('top')
                self.base_ax.xaxis.tick_top()

        odds_df, evens_df = self.__get_odds_evens()

        if self.signal_color_col is None and self.twas_color_col is None:
            self.base_ax.scatter(odds_df[self.plot_x_col], odds_df[self.plot_y_col], c=LIGHT_CHR_COLOR, s=2)
            self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], c=DARK_CHR_COLOR, s=2)
        else:
            self.base_ax.scatter(odds_df[self.plot_x_col], odds_df[self.plot_y_col], c='silver', s=2)
            self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], c='dimgray', s=2)

        if self.vertical:
            self.base_ax.set_ylim(self.df['ABS_POS'].min(), self.df['ABS_POS'].max())
            self.base_ax.set_xlabel('- Log10 P')
            self.base_ax.set_ylabel('Chromosomal Position')
            self.base_ax.axvline(-np.log10(self.sig_line), c=FIFTH_COLOR)
            self.base_ax.invert_yaxis()
            invisi_spine = 'right' if not self.invert else 'left'
            self.base_ax.spines[invisi_spine].set_visible(False)
        else:
            self.base_ax.set_xlim(self.df['ABS_POS'].min(), self.df['ABS_POS'].max())
            self.base_ax.set_ylabel('- Log10 P')
            self.base_ax.set_xlabel('Chromosomal Position')
            self.base_ax.axhline(-np.log10(self.sig_line), c=FIFTH_COLOR)
            invisi_spine = 'top' if not self.invert else 'bottom'
            self.base_ax.spines[invisi_spine].set_visible(False)

        self.__add_threshold_ticks()

        if self.vertical:
            if not self.invert:
                self.base_ax.set_xlim(left=0)
                if self.max_log_p is not None:
                    self.base_ax.set_xlim(right=self.max_log_p)
                self.max_x = self.base_ax.get_xlim()[1]
            else:
                self.base_ax.set_xlim(right=0)
                if self.max_log_p is not None:
                    self.base_ax.set_xlim(left=self.max_log_p)
                self.max_x = self.base_ax.get_xlim()[0]
        else:
            if not self.invert:
                self.base_ax.set_ylim(bottom=np.floor(-np.log10(self.df['P'].max())))
                if self.max_log_p is not None:
                    self.base_ax.set_ylim(top=self.max_log_p)
                self.max_y = self.base_ax.get_ylim()[1]
            else:
                self.base_ax.set_ylim(top=np.floor(-np.log10(self.df['P'].max())))
                if self.max_log_p is not None:
                    self.base_ax.set_ylim(bottom=self.max_log_p)
                self.max_y = self.base_ax.get_ylim()[0]

        self.fig.patch.set_facecolor('white')

    def plot_specific_signals(self, signal_bed_df):
        odds_df, evens_df = self.__find_signals_specific(signal_bed_df)

        if self.signal_color_col is None:
            self.__plot_signals(odds_df, evens_df)
        else:
            self.__plot_color_signals(odds_df, evens_df)

    def plot_sig_signals(self, rep_genes=[], rep_boost=False):
        odds_df, evens_df = self.__find_signals_sig(rep_genes, rep_boost)

        if self.signal_color_col is None:
            self.__plot_signals(odds_df, evens_df)
        else:
            self.__plot_color_signals(odds_df, evens_df)

    def plot_annotations(self, plot_sig=True, rep_genes=[], rep_boost=False):
        halfLD = self.ld_block / 2
        alreadyPlottedPos = []
        alreadyPlottedGenes = []
        self.annot_list = []

        # Signals and annotations always adhere to annotation threshold
        annot_mask = self.thinned['P'] < self.annot_thresh
        annotDF = self.thinned[annot_mask]

        if plot_sig:
            sig_mask = annotDF['P'] < self.sig
        else:
            sig_mask = False

        if rep_boost:
            rep_mask = annotDF['ID'].isin(rep_genes)
        else:
            rep_mask = False

        sug_mask = annotDF['P'] < self.sug
        spec_mask = annotDF['ID'].isin(self.spec_genes)

        full_mask = sig_mask | (sug_mask & rep_mask) | spec_mask
        annotDF = annotDF[full_mask].set_index('ID')

        for signalID, row in annotDF.iterrows():
            plot = True
            if signalID in alreadyPlottedGenes:
                plot = False
            elif self.merge_genes:
                for x in alreadyPlottedPos:
                    if not plot:
                        break
                    if x - halfLD < row['ROUNDED_X'] < x + halfLD:
                        plot = False
            if row['P'] > self.annot_thresh:
                # genes follow annotation threshold ALWAYS
                plot = False
            if plot:
                # self.base_ax.annotate(signalID, xy=(row[self.plot_x_col], row[self.plot_y_col]), va='center', ha='left')
                signalDF = annotDF.loc[annotDF.index == signalID]
                if self.vertical:
                    if self.max_log_p is not None:
                        pointer_x = signalDF[signalDF[self.plot_x_col] <= self.max_log_p][self.plot_x_col].max()
                    else:
                        pointer_x = signalDF[self.plot_x_col].max()
                    self.base_ax.plot([pointer_x, self.max_x],
                                      [row[self.plot_y_col], row[self.plot_y_col]],
                                      c='silver', linewidth=1.5, alpha=1)
                else:
                    if self.max_log_p is not None:
                        pointer_y = signalDF[signalDF[self.plot_y_col] <= self.max_log_p][self.plot_y_col].max()
                    else:
                        pointer_y = signalDF[self.plot_y_col].max()
                    self.base_ax.plot([row[self.plot_x_col], row[self.plot_x_col]],
                                      [pointer_y, self.max_y],
                                      c='silver', linewidth=1.5, alpha=1)
                alreadyPlottedPos.append(row['ROUNDED_X'])
                alreadyPlottedGenes.append(signalID)
                self.annot_list.append(row)

    def plot_table(self, extra_cols={}, number_cols=[], rep_genes=[], keep_chr_pos=True):
        if self.vertical:
            self.__plot_table_vertical(extra_cols=extra_cols, number_cols=number_cols, rep_genes=rep_genes, keep_chr_pos=keep_chr_pos)
        else:
            self.__plot_table_horizontal(rep_genes=rep_genes)

    def full_plot(self, rep_genes=[], extra_cols={}, number_cols=[], rep_boost=False, save=None, with_table=True, save_res=150, with_title=True, keep_chr_pos=True):
        self.plot_data(with_table=with_table)
        self.plot_sig_signals(rep_genes=rep_genes, rep_boost=rep_boost)
        if with_table:
            self.plot_annotations(rep_genes=rep_genes, rep_boost=rep_boost)
            self.plot_table(extra_cols=extra_cols, number_cols=number_cols, rep_genes=rep_genes, keep_chr_pos=keep_chr_pos)
        if with_title:
            plt.suptitle(self.title)
            plt.tight_layout()
        if save is not None:
            plt.savefig(save, dpi=save_res)
        plt.show()

    def full_plot_with_specific(self, signal_bed_df, plot_sig=True, rep_boost=False, rep_genes=[], extra_cols={}, number_cols=[], verbose=False, save=None, save_res=150):
        if verbose:
            print('Plotting All Data...', flush=True)
        self.plot_data()
        if plot_sig:
            if verbose:
                print('Plotting Significant Signals...', flush=True)
            self.plot_sig_signals()
        if verbose:
            print('Plotting Specific Signals...', flush=True)
        self.plot_specific_signals(signal_bed_df)
        if verbose:
            print('Finding Annotations...', flush=True)
        self.plot_annotations(plot_sig=plot_sig, rep_genes=rep_genes, rep_boost=rep_boost)
        if verbose:
            print('Adding Table...', flush=True)
        self.plot_table(extra_cols=extra_cols, number_cols=number_cols, rep_genes=rep_genes)
        if save is not None:
            plt.savefig(save, dpi=save_res)
        plt.show()

    def qq_plot(self, save=None, save_res=150, with_title=True, steps=30):
        log_series = -np.log10(self.df['P'])
        chi2_series = pd.Series(chi2.ppf(1 - self.df['P'], df=1), index=log_series.index)

        max_log = log_series.max()

        lambda_gc = chi2_series.median() / chi2.ppf(0.5, 1)
        print('Lambda GC:', lambda_gc)

        step_vals = pd.Series(index=np.linspace(0, max_log, steps), name='Count', dtype=float)

        for thresh in step_vals.index:
            step_vals.loc[thresh] = (log_series >= thresh).astype(int).sum()

        step_fractions = step_vals / len(log_series)
        step_logs = -np.log10(step_fractions)

        plt.gcf().set_size_inches(6, 6)
        plt.gcf().set_facecolor('w')
        plt.plot([0, max_log], [0, max_log], c='r', label='Null Model', linestyle='dashed')
        plt.plot(step_logs.index, step_logs.values, c='blue', label='GWAS Results')
        plt.xlabel('Expected Log10 Quantiles')
        plt.ylabel('Observed Log10 Quantiles')
        plt.legend(loc='upper left')
        plt.annotate('Lambda GC: ' + str(round(lambda_gc, 5)), fontsize=12,
                     xy=(0.95, 0.05), xycoords='axes fraction',
                     horizontalalignment='right', verticalalignment='bottom')
        if with_title:
            plt.title('QQ Plot:\n' + self.title)
        plt.xlim(0, max_log)
        plt.ylim(0, max_log)
        plt.tight_layout()
        if save is not None:
            plt.savefig(save, dpi=save_res)
        plt.show()

    def save_thinned_df(self, path, pickle=True):
        if pickle:
            self.thinned.to_pickle(path)
        else:
            self.thinned.to_csv(path)

    # Private Functions Start Here

    def __get_absolute_positions(self, active_df):
        lastPos = self.df[~self.df['#CHROM'].duplicated(keep='last')].copy()
        lastPos['POS'] = lastPos['POS'].astype(np.int64)
        lastPos = lastPos.set_index('#CHROM').sort_index()['POS']
        lastPos.update(CHR_LENGTHS)

        addMap = lastPos.cumsum()
        addMap[0] = 0
        addMap = addMap.sort_index()

        reps = active_df['#CHROM'].value_counts().sort_index()
        reps = reps.reindex(addMap.index.drop(0), fill_value=0)
        addVec = addMap.iloc[:-1].repeat(reps.values)

        abs_pos = active_df['POS'].values + addVec.values

        tickLocs = (lastPos / 2) + addMap.iloc[:-1].values
        self.chr_ticks = [tickLocs.values, addMap.index[1:]]

        return abs_pos

    def __update_param(self, old, new):
        if new != '' and new != old:
            return new
        return old

    def __fmt_print_rows(self, print_df):
        return print_df.apply(lambda x: x.name + '\t' + '\t'.join(x.astype(str)), axis=1)

    def __get_odds_evens(self):
        odds = np.arange(1, 24, 2)
        evens = np.arange(2, 23, 2)

        odds_df = self.thinned[self.thinned['#CHROM'].isin(odds)].copy()
        evens_df = self.thinned[self.thinned['#CHROM'].isin(evens)].copy()

        return odds_df, evens_df

    def __config_axes(self, with_table=True):
        need_cbar = (self.signal_color_col is not None) or (self.twas_color_col is not None)

        if self.vertical and not need_cbar and with_table:
            # Vertical, no color bar, table
            self.fig, axes = plt.subplots(nrows=1, ncols=2)
            self.fig.set_size_inches(12, 12)
            if not self.invert:
                self.base_ax = axes[0]
                self.table_ax = axes[1]
            else:
                self.base_ax = axes[1]
                self.table_ax = axes[0]
        elif not self.vertical and not need_cbar and with_table:
            # Horizontal, no color bar, table
            print('Horizontal, no color bar')
            ratios = [0.4, 1] if not self.invert else [1, 0.4]
            self.fig, axes = plt.subplots(nrows=2, ncols=1,
                                          gridspec_kw={'height_ratios': ratios})
            self.fig.set_size_inches(14.4, 6)
            if not self.invert:
                self.table_ax = axes[0]
                self.base_ax = axes[1]
            else:
                self.table_ax = axes[1]
                self.base_ax = axes[0]

        elif not self.vertical and need_cbar and with_table:
            # Horizontal, color bar, table
            ratios = [0.08, 0.45, 1] if not self.invert else [1, 0.45, 0.08]
            self.fig, axes = plt.subplots(nrows=3, ncols=1,
                                          gridspec_kw={'height_ratios': ratios})
            self.fig.set_size_inches(14.4, 6)
            self.table_ax = axes[1]

            if not self.invert:
                self.cbar_ax = axes[0]
                self.base_ax = axes[2]
            else:
                self.cbar_ax = axes[2]
                self.base_ax = axes[0]

        elif not self.vertical and need_cbar and not with_table:
            # Horizontal, color bar, no table
            ratios = [0.08, 1] if not self.invert else [1, 0.08]
            self.fig, axes = plt.subplots(nrows=2, ncols=1,
                                          gridspec_kw={'height_ratios': ratios})
            self.fig.set_size_inches(14.4, 4)

            if not self.invert:
                self.cbar_ax = axes[0]
                self.base_ax = axes[1]
            else:
                self.cbar_ax = axes[1]
                self.base_ax = axes[0]

        elif not self.vertical and not need_cbar and not with_table:
            # Horizontal, no color bar, no table
            self.fig, self.base_ax = plt.subplots()
            self.fig.set_size_inches(13, 3)

        else:
            print('No support for your configuration...')

        if self.vertical and self.invert:
            self.base_ax.invert_xaxis()
        elif not self.vertical and self.invert:
            self.base_ax.invert_yaxis()

    def __add_threshold_ticks(self):
        if self.annot_thresh <= self.sug:
            ticks = [-np.log10(self.annot_thresh)]
        else:
            ticks = []
        if self.sug < self.annot_thresh:
            ticks.append(-np.log10(self.sug))
        if self.sig < self.annot_thresh:
            ticks.append(-np.log10(self.sig))

        end1 = self.df['ABS_POS'].max()
        end2 = end1 * 0.99

        for t in ticks:
            if self.vertical:
                self.base_ax.plot([t, t], [end1, end2], c=FIFTH_COLOR)
            else:
                self.base_ax.plot([end1, end2], [t, t], c=FIFTH_COLOR)

    def __find_signals_sig(self, rep_genes=[], rep_boost=False):
        odds_df, evens_df = self.__get_odds_evens()

        halfLD = self.ld_block / 2

        odds_df['SIGNAL'] = False
        evens_df['SIGNAL'] = False

        odds_df['Replication'] = False
        evens_df['Replication'] = False

        # Signals are always limited by annotation threshold
        annot_mask = self.thinned['P'] < self.annot_thresh
        test_df = self.thinned[annot_mask]

        if rep_boost:
            # If boosting known genes, consider suggestive
            p_mask = test_df['P'] < self.sug
        else:
            # If not boosting known genes, consider significant
            p_mask = test_df['P'] < self.sig

        test_df = test_df[p_mask]
        test_df = test_df.sort_values(by='P')

        signal_genes = []

        for rowID, row in test_df.iterrows():
            if rep_boost:
                if row['ID'] not in rep_genes and row['P'] > self.sig:
                    # If boosting known genes, and this gene is novel, use significance threshold
                    continue
            chr_df = odds_df.iloc[:] if row['#CHROM'] % 2 == 1 else evens_df.iloc[:]
            if (self.merge_genes or row['ID'] in signal_genes) and chr_df.loc[rowID, 'SIGNAL']:
                # When not merging genes, test for gene name
                # When merging genes, test for position
                continue

            x, gene = row['ROUNDED_X'], row['ID']
            chr_pos_mask = chr_df['ROUNDED_X'].between(x - halfLD, x + halfLD)
            chr_pos_idx = chr_df.index[chr_pos_mask]

            if self.merge_genes and np.any(chr_df.loc[chr_pos_idx, 'ID'].isin(rep_genes)):
                window_genes = chr_df.loc[chr_pos_idx, ['ID', 'P']].set_index('ID')
                window_genes = window_genes[window_genes.index.isin(rep_genes)]
                new_gene = window_genes.idxmin().values[0]
                self.thinned.loc[chr_pos_idx, 'ID'] = new_gene
                gene = new_gene

            currentRep = chr_df.loc[chr_pos_idx, 'Replication']

            if row['#CHROM'] % 2 == 1:
                odds_df.loc[chr_pos_idx, 'Replication'] = np.logical_or(currentRep, gene in rep_genes)
                odds_df.loc[chr_pos_idx, 'SIGNAL'] = True
            else:
                evens_df.loc[chr_pos_idx, 'Replication'] = np.logical_or(currentRep, gene in rep_genes)
                evens_df.loc[chr_pos_idx, 'SIGNAL'] = True

            signal_genes.append(gene)

        odds_df = odds_df[odds_df['SIGNAL']]
        evens_df = evens_df[evens_df['SIGNAL']]

        return odds_df, evens_df

    def __find_signals_specific(self, signal_bed_df):
        odds_df, evens_df = self.__get_odds_evens()
        self.spec_genes = []

        odds_df['SIGNAL'] = False
        evens_df['SIGNAL'] = False

        odds_df['Replication'] = False
        evens_df['Replication'] = False

        signal_bed_df = signal_bed_df[signal_bed_df['#CHROM'].isin(self.df['#CHROM'])].copy()
        signal_bed_df = signal_bed_df.sort_values(by=['#CHROM', 'POS'])
        signal_bed_df['ABS_POS'] = self.__get_absolute_positions(signal_bed_df)

        signal_bed_df['ROUNDED_X'] = signal_bed_df['ABS_POS'] // CHR_POS_ROUND * CHR_POS_ROUND

        test_df = self.thinned[self.thinned['P'] < self.annot_thresh].reset_index(drop=False).set_index('ROUNDED_X')

        n = self.ld_block // CHR_POS_ROUND // 2 + 1
        keep_locs = []
        for _, row in signal_bed_df.iterrows():
            x = row['ROUNDED_X']
            rounded_locs = x + CHR_POS_ROUND * np.arange(-n, n + 1)
            keep_locs.extend(list(rounded_locs))

        test_df = test_df.loc[test_df.index.intersection(keep_locs)]

        for signal_df in [odds_df, evens_df]:
            for chrom, subDF in signal_df.groupby('#CHROM'):
                if chrom not in list(signal_bed_df['#CHROM']):
                    continue

                print('chr' + str(chrom), end=' ')
                rep_sub_df = signal_bed_df[signal_bed_df['#CHROM'] == chrom]

                for _, row in rep_sub_df.iterrows():
                    x = row['ROUNDED_X']
                    n = self.ld_block // CHR_POS_ROUND // 2 + 1
                    start = row['START']
                    end = row['END']

                    rounded_locs = x + CHR_POS_ROUND * np.arange(-n, n + 1)

                    keep_locs = test_df.index.intersection(rounded_locs)
                    if len(keep_locs) == 0:
                        continue

                    gene_df = test_df.loc[keep_locs].copy().reset_index(drop=False).set_index('index')
                    if 'ID' not in row.index:
                        gene = gene_df['ID'].mode().iloc[0]
                    else:
                        gene = row['ID']

                    self.thinned.loc[gene_df.index, 'ID'] = gene
                    if self.signal_color_col is not None and False in pd.isnull(gene_df[self.signal_color_col]):
                        self.thinned.loc[gene_df.index, self.signal_color_col] = \
                        gene_df[self.signal_color_col].mode().iloc[0]

                    if self.thinned.loc[self.thinned['ID'] == gene, 'P'].min() > self.sug:
                        continue

                    self.spec_genes.append(gene)

                    signal_pos_mask = signal_df['POS'].between(start, end)
                    signal_pos_index = signal_df.index[signal_pos_mask]

                    signal_index = subDF.index.intersection(signal_pos_index)

                    signal_df.loc[signal_index, 'SIGNAL'] = True
                    signal_df.loc[signal_index, 'Replication'] = True

        odds_df = odds_df[odds_df['SIGNAL']]
        evens_df = evens_df[evens_df['SIGNAL']]

        print('')

        return odds_df, evens_df

    def __plot_signals(self, odds_df, evens_df):
        colors = odds_df['Replication'].replace({True: REP_HIT_COLOR, False: NOVEL_HIT_COLOR})
        self.base_ax.scatter(odds_df[self.plot_x_col], odds_df[self.plot_y_col], c=colors, s=10)
        colors = evens_df['Replication'].replace({True: REP_HIT_COLOR, False: NOVEL_HIT_COLOR})
        self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], c=colors, s=10)

    def __plot_color_signals(self, odds_df, evens_df):
        unique_vals = list(odds_df[self.signal_color_col].dropna().unique())
        unique_vals.extend(list(evens_df[self.signal_color_col].dropna().unique()))
        unique_vals = list(set(unique_vals))
        discrete = len(unique_vals) <= 40
        if not discrete:
            self.base_ax.scatter(odds_df[self.plot_x_col], odds_df[self.plot_y_col], c=odds_df[self.signal_color_col],
                                 cmap=plt.cm.get_cmap(COLOR_MAP), s=10,
                                 vmin=odds_df[self.signal_color_col].quantile(0.05),
                                 vmax=odds_df[self.signal_color_col].quantile(0.95))
            scat = self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col],
                                        c=evens_df[self.signal_color_col],
                                        cmap=plt.cm.get_cmap(COLOR_MAP), s=10,
                                        vmin=odds_df[self.signal_color_col].quantile(0.05),
                                        vmax=odds_df[self.signal_color_col].quantile(0.95))
            self.fig.colorbar(scat, cax=self.cbar_ax, orientation='horizontal')

        else:
            categories = sorted(unique_vals)
            cat_to_num = dict(zip(categories, np.arange(len(categories))))
            odds_df['Cat_Num'] = odds_df[self.signal_color_col].replace(cat_to_num)
            evens_df['Cat_Num'] = evens_df[self.signal_color_col].replace(cat_to_num)

            self.base_ax.scatter(odds_df[self.plot_x_col], odds_df[self.plot_y_col], c=odds_df['Cat_Num'],
                                 cmap=plt.cm.get_cmap(COLOR_MAP, len(categories)), s=10)
            scat = self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], c=evens_df['Cat_Num'],
                                        cmap=plt.cm.get_cmap(COLOR_MAP, len(categories)), s=10)
            print(categories, cat_to_num, unique_vals)
            self.__add_color_bar(scat, categories)

    def __add_color_bar(self, mappable, categories):
        print(mappable)
        print(categories)
        cbar = self.fig.colorbar(mappable, cax=self.cbar_ax, orientation='horizontal')
        xmin, xmax = self.cbar_ax.get_xlim()
        factor = (xmax - xmin) / len(categories)

        categories = [c if len(c) < 20 else c[:17] + '...' for c in categories]

        cbar.set_ticks((np.arange(len(categories)) + 0.5) * factor + xmin)
        if not self.invert:
            cbar.ax.set_xticklabels(categories, rotation=30, ha='left')
            self.cbar_ax.xaxis.tick_top()
        else:
            cbar.ax.set_xticklabels(categories, rotation=30, ha='right')
        self.fig.tight_layout()

    def __plot_table_vertical(self, extra_cols={}, number_cols=[], rep_genes=[], keep_chr_pos=True):
        if keep_chr_pos:
            columns = ['ID', 'CHR', 'POS', 'P']
        else:
            columns = ['ID', 'P']

        columns.extend(extra_cols.values())

        if len(self.annot_list) == 0:
            self.table_ax.set_visible(False)
            return

        annot_table = pd.concat(self.annot_list, axis=1).transpose()
        annot_table = annot_table.sort_values(by=['#CHROM', 'POS'])
        annot_table = annot_table.reset_index()
        annot_table = annot_table.rename(columns={'#CHROM': 'CHR',
                                                  'index': 'ID'})
        annot_table = annot_table.rename(columns=extra_cols)
        annot_table['P'] = annot_table['P'].apply(lambda x: '{:.3e}'.format(x))
        annot_table[number_cols] = annot_table[number_cols].applymap(lambda x: '{:.3}'.format(x))

        location = 'center left' if not self.invert else 'center right'
        table = mpl.table.table(ax=self.table_ax,
                                cellText=annot_table[columns].fillna('').values,
                                colLabels=columns, loc=location,
                                colColours=[TABLE_HEAD_COLOR for c in columns])
        table.AXESPAD = 0

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.auto_set_column_width(col=list(range(len(annot_table.columns))))
        self.fig.tight_layout()

        self.table_ax.set_axis_off()
        self.table_ax.invert_yaxis()
        if self.invert:
            self.table_ax.invert_xaxis()

        cell_height = table[(0, 0)].get_height()

        for i in range(len(annot_table)):
            connection_row = annot_table.iloc[i]
            cell = table[(i+1, 0)]
            cell_text = cell.get_text().get_text()
            if cell_text in rep_genes:
                cell.set_facecolor(REP_TABLE_COLOR)
            else:
                cell.set_facecolor(NOVEL_TABLE_COLOR)
            connect_x = 0
            cp = ConnectionPatch(xyA=(self.max_x, connection_row[self.plot_y_col]),
                                 axesA=self.base_ax, coordsA='data',
                                 xyB=(connect_x, (1 - cell.get_y()) - (0.5*cell_height)),
                                 axesB=self.table_ax, coordsB='data',
                                 arrowstyle='-', color='silver')
            self.fig.add_artist(cp)

    def __plot_table_horizontal(self, rep_genes=[]):
        annotTable = pd.concat(self.annot_list, axis=1).transpose()
        annotTable = annotTable.sort_values(by=['#CHROM', 'POS'])
        genes = [list(annotTable.index)]
        num_cols = len(annotTable)

        table = self.table_ax.table(cellText=genes,
                                    loc='lower center',
                                    colWidths=[1 / (num_cols + 2) for g in genes[0]],
                                    cellLoc='center')
        table.AXESPAD = 0

        for cell in table._cells:
            table._cells[cell].get_text().set_rotation(90)
            table._cells[cell].set_height(1)

        self.table_ax.set_axis_off()
        self.fig.tight_layout()

        if self.twas_color_col is not None:
            cmap = plt.cm.get_cmap(COLOR_MAP)
            unique_vals = sorted(annotTable[self.twas_color_col].unique())
            fractions = np.arange(len(unique_vals)) / len(unique_vals)
            colors = [cmap(f) for f in fractions]
            color_map = dict(zip(unique_vals, colors))

            fractions = list(fractions)
            fractions.append(1.0)
            new_norm = mpl.colors.BoundaryNorm(boundaries=fractions, ncolors=len(fractions) - 1)
            new_mappable = plt.cm.ScalarMappable(norm=new_norm, cmap=plt.cm.get_cmap(COLOR_MAP, len(fractions) - 1))
            self.__add_color_bar(new_mappable, color_map.keys())

        cell_width = table[(0, 0)].get_width()
        cell_height = table[(0, 0)].get_height()
        for i in range(num_cols):
            connection_row = annotTable.iloc[i]
            cell_text = table[(0, i)].get_text().get_text()
            if cell_text in rep_genes:
                table[(0, i)].set_facecolor(REP_TABLE_COLOR)
                # table[(0, i)].set_facecolor('silver')
            else:
                table[(0, i)].set_facecolor(NOVEL_TABLE_COLOR)
                # table[(0, i)].set_facecolor('silver')
            connect_y = 0 if not self.invert else cell_height
            cp = ConnectionPatch(xyA=(connection_row[self.plot_x_col], self.max_y),
                                 axesA=self.base_ax, coordsA='data',
                                 xyB=(table[(0, i)].get_x() + (0.5*cell_width), connect_y),
                                 axesB=self.table_ax, coordsB='axes fraction',
                                 arrowstyle='-', color='silver')
            if self.twas_updown_col is not None:
                shape = 'v' if connection_row[self.twas_updown_col] < 0 else '^'
                if self.twas_color_col is None:
                    color = REP_HIT_COLOR if cell_text in rep_genes else NOVEL_HIT_COLOR
                else:
                    color = color_map[connection_row[self.twas_color_col]]

                self.base_ax.scatter(connection_row[self.plot_x_col],
                                     connection_row[self.plot_y_col],
                                     color=color, marker=shape, s=60)

            self.fig.add_artist(cp)
