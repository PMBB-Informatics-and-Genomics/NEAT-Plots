import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches
import matplotlib as mpl
from scipy.stats import chi2
import json


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
class BoroughsPlot:
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

    invert = False
    merge_genes = False
    max_log_p = None
    signal_rep_map = {}

    fig, base_ax, table_ax, cbar_ax = None, None, None, None
    lower_base_ax, upper_base_ax = None, None
    annot_list = []
    spec_genes = []

    facets = []
    facet_count = None

    DARK_CHR_COLOR = '#5841bf'
    LIGHT_CHR_COLOR = '#648fff'
    NOVEL_HIT_COLOR = '#dc267f'
    NOVEL_TABLE_COLOR = '#eb7fb3'
    REP_HIT_COLOR = '#ffbb00'
    REP_TABLE_COLOR = '#ffdc7a'
    FIFTH_COLOR = '#d45c00'
    TABLE_HEAD_COLOR = '#9e9e9e'
    # COLOR_MAP = 'turbo_r'
    COLOR_MAP = plt.cm.get_cmap('turbo_r')
    # COLOR_MAP = 'Paired'

    CHR_POS_ROUND = 5E4
    MIN_PT_SZ = 5
    MAX_PT_SZ = 200

    MIN_TRI_SZ = 5
    MAX_TRI_SZ = 200

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

    def config_colors(self, color_file_json):
        """
        Configures the manhattan plot's color palette as specified by user.
        chromosome color keys: DARK_CHR_COLOR, LIGHT_CHR_COLOR
        novel hit color keys: NOVEL_HIT_COLOR, NOVEL_HIT_TABLE_COLOR
        replicated hit color keys: REP_HIT_COLOR, REP_TABLE_COLOR
        additional color in the palette: FIFTH_COLOR (used for threshold line)
        table color keys: TABLE_HEAD_COLOR
        set color map from matplotlib for colors bars: COLOR_MAP
        :param color_file_json: str
        """
        color_config_dict = json.load(open(color_file_json))

        for k, v in color_config_dict:
            self.__setattr__(k, v)

    def reset_colors(self):
        """
        Resets the color palette options to default
        """
        color_default_dict = {'DARK_CHR_COLOR': '#5841bf',
                              'LIGHT_CHR_COLOR': '#648fff',
                              'NOVEL_HIT_COLOR': '#dc267f',
                              'NOVEL_TABLE_COLOR': '#eb7fb3',
                              'REP_HIT_COLOR': '#ffbb00',
                              'REP_TABLE_COLOR': '#ffdc7a',
                              'FIFTH_COLOR': '#d45c00',
                              'TABLE_HEAD_COLOR': '#9e9e9e',
                              'COLOR_MAP': 'turbo_r'}

        for k, v in color_default_dict:
            self.__setattr__(k, v)

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
            self.df = pd.read_table(self.path, sep=delim, nrows=self.test_rows, low_memory=False)

        self.df.index = np.arange(len(self.df))
        print('Loaded', len(self.df), 'Rows')
        print(self.df.columns)

    def clean_data(self, col_map=None, logp=None, has_chr_prefix=False):
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

        if has_chr_prefix:
            self.df['#CHROM'] = self.df['#CHROM'].str[3:]
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

        self.df['P'] = pd.to_numeric(self.df['P'], errors='coerce')
        self.df['P'] = self.df['P'].replace(0, self.df['P'].min() / 100)

        if 'WRAP' not in self.df.columns:
            raise ValueError('WRAP column must be indicated for Boroughs plot')

    def check_data(self):
        """
        Prints the beginning and end of the data table (required columns only) for sanity-checking
        """
        print(self.df.head()[['#CHROM', 'POS', 'P', 'ID']])
        print(self.df.tail()[['#CHROM', 'POS', 'P', 'ID']])
        print(len(self.df))

    def add_annotations(self, annot_df: pd.DataFrame, extra_cols=[]):
        """

        :param annot_df:
        :type annot_df: pd.DataFrame
        :param extra_cols:
        :type extra_cols: list
        """
        annot_cols = ['#CHROM', 'POS', 'ID']
        annot_cols.extend(extra_cols)
        self.df = self.df.drop(columns='ID_y', errors='ignore')
        annot_df['#CHROM'] = annot_df['#CHROM'].replace('X', 23).astype(int)
        self.df = self.df.merge(annot_df[annot_cols], on=['#CHROM', 'POS'], how='left')
        self.df['ID_x'].update(self.df['ID_y'])
        self.df = self.df.rename(columns={'ID_x': 'ID'})

    def update_plotting_parameters(self, annotate='', signal_color_col='', twas_color_col='', twas_updown_col='', sig='', sug='', annot_thresh='', ld_block='', max_log_p='', invert='', merge_genes='', title=''):
        self.annotate = self.__update_param(self.annotate, annotate)
        self.ld_block = self.__update_param(self.ld_block, ld_block)
        self.title = self.__update_param(self.title, title)

        self.signal_color_col = self.__update_param(self.signal_color_col, signal_color_col)

        self.twas_updown_col = self.__update_param(self.twas_updown_col, twas_updown_col)
        self.twas_color_col = self.__update_param(self.twas_color_col, twas_color_col)

        self.sig = self.__update_param(self.sig, sig)
        self.sug = self.__update_param(self.sug, sug)
        self.annot_thresh = self.__update_param(self.annot_thresh, annot_thresh)
        self.max_log_p = self.__update_param(self.max_log_p, max_log_p)

        self.plot_x_col = 'ROUNDED_X'
        self.plot_y_col = 'ROUNDED_Y'
        self.invert = self.__update_param(self.invert, invert)
        self.merge_genes = self.__update_param(self.merge_genes, merge_genes)

    def check_plotting_parameters(self):
        params = {'Significance Threshold': self.sig,
                  'Suggestive Threshold': self.sug,
                  'Annotation Threshold': self.annot_thresh,
                  'LD Block Width': self.ld_block,
                  'Annotating?': self.annotate,
                  'Maximum Neg. Log P-Val': self.max_log_p}

        if self.signal_color_col is not None:
            params['Edge Color Column'] = self.signal_color_col
        print(params)

    def get_thinned_data(self, log_p_round=2, additional_cols=[]):
        if 'ABS_POS' not in self.df.columns:
            self.df['ABS_POS'] = self.__get_absolute_positions(self.df)

        self.thinned = self.df.copy()
        self.thinned['ROUNDED_X'] = self.thinned['ABS_POS'] // self.CHR_POS_ROUND * self.CHR_POS_ROUND
        self.thinned['ROUNDED_Y'] = pd.Series(-np.log10(self.thinned['P'])).round(log_p_round)  # round to 2 decimals
        subset_cols = ['ROUNDED_X', 'ROUNDED_Y', 'WRAP']
        subset_cols.extend(additional_cols)
        self.thinned = self.thinned.sort_values(by='P').drop_duplicates(subset=subset_cols)
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

    def plot_data(self, with_table=True, legend_loc=None):
        self.__config_axes(with_table=with_table, legend_loc=legend_loc)
        odds_df, evens_df = self.__get_odds_evens()

        for i, b in enumerate(self.base_ax):
            odds = odds_df[i]
            evens = evens_df[i]

            b.set_xticks(self.chr_ticks[0])
            b.set_xticklabels(self.chr_ticks[1])
            if self.invert:
                b.xaxis.set_label_position('top')
                b.xaxis.tick_top()

            if self.signal_color_col is None and self.twas_color_col is None:
                b.scatter(odds[self.plot_x_col], odds[self.plot_y_col], c=self.LIGHT_CHR_COLOR, s=2)
                b.scatter(evens[self.plot_x_col], evens[self.plot_y_col], c=self.DARK_CHR_COLOR, s=2)
            else:
                b.scatter(odds[self.plot_x_col], odds[self.plot_y_col], edgecolors='silver', s=2)
                b.scatter(evens[self.plot_x_col], evens[self.plot_y_col], edgecolors='dimgray', s=2)

        self.__add_threshold_ticks()
        self.__cosmetic_axis_edits()

    def plot_specific_signals(self, signal_bed_df):
        odds_df, evens_df = self.__find_signals_specific(signal_bed_df)

        if self.signal_color_col is None:
            self.__plot_signals(odds_df, evens_df)
        else:
            self.__plot_color_signals(odds_df, evens_df)

    def plot_sig_signals(self, rep_genes=[], rep_boost=False, legend_loc=None):
        odds_df, evens_df = self.__find_signals_sig(rep_genes, rep_boost)

        if self.signal_color_col is None:
            self.__plot_signals(odds_df, evens_df)
        else:
            self.__plot_color_signals(odds_df, evens_df, legend_loc=legend_loc)

    def plot_annotations(self, plot_sig=True, rep_genes=[], rep_boost=False):
        halfLD = self.ld_block / 2

        self.annot_list = []

        for i, b in enumerate(self.base_ax):

            alreadyPlottedPos = []
            alreadyPlottedGenes = []
            this_annot_list = []

            # Signals and annotations always adhere to annotation threshold
            annot_mask = self.thinned['P'] < self.annot_thresh
            annotDF = self.thinned[annot_mask]
            annotDF = annotDF[annotDF['WRAP'] == self.facets[i]]

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
                signal_gene = signalID
                if rep_boost and signalID in self.signal_rep_map.keys():
                    new_gene = self.signal_rep_map[signalID]
                    signalID = new_gene
                    row.name = new_gene
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
                    signalDF = annotDF.loc[annotDF.index == signal_gene]
                    if self.max_log_p is not None:
                        pointer_y = signalDF[signalDF[self.plot_y_col] <= self.max_log_p][self.plot_y_col].max()
                    else:
                        pointer_y = signalDF[self.plot_y_col].max()

                    max_ax_y = b.get_ylim()[1]
                    b.plot([row[self.plot_x_col], row[self.plot_x_col]],
                           [pointer_y, max_ax_y],
                           c='silver', linewidth=1.5, alpha=1)
                    alreadyPlottedPos.append(row['ROUNDED_X'])
                    alreadyPlottedGenes.append(signalID)
                    this_annot_list.append(row)

            self.annot_list.append(this_annot_list)

    def plot_table(self, extra_cols={}, number_cols=[], rep_genes=[], keep_chr_pos=True, with_table_bg=True, with_table_grid=True, text_rep_colors=False):
        self.__plot_table_horizontal(rep_genes=rep_genes, with_table_bg=with_table_bg, with_table_grid=with_table_grid, text_rep_colors=text_rep_colors)

    def full_plot(self, rep_genes=[], extra_cols={}, number_cols=[], rep_boost=False, save=None, with_table=True,
                  save_res=150, with_title=True, keep_chr_pos=True, with_table_bg=True, with_table_grid=True,
                  legend_loc=None, text_rep_colors=False):
        self.facets = sorted(self.thinned['WRAP'].unique())
        self.plot_data(with_table=with_table, legend_loc=legend_loc)
        self.plot_sig_signals(rep_genes=rep_genes, rep_boost=rep_boost, legend_loc=legend_loc)
        if with_table:
            self.plot_annotations(rep_genes=rep_genes, rep_boost=rep_boost)
            self.plot_table(extra_cols=extra_cols, number_cols=number_cols, rep_genes=rep_genes, keep_chr_pos=keep_chr_pos, with_table_bg=with_table_bg, with_table_grid=with_table_grid, text_rep_colors=text_rep_colors)
        if with_title:
            plt.suptitle(self.title)
            plt.tight_layout()
        if save is not None:
            plt.savefig(save, dpi=save_res)
        # plt.show()
        # plt.clf()

    def signal_plot(self, rep_genes=[], extra_cols={}, number_cols=[], rep_boost=False, save=None, with_table=True, save_res=150, with_title=True, keep_chr_pos=True):
        self.__config_axes(with_table=with_table)

        odds_df, evens_df = self.__find_signals_sig(rep_genes, rep_boost)
        signal_df = pd.concat([odds_df, evens_df]).sort_values(by=['#CHROM', 'POS'])
        signal_order = signal_df['ID'].unique()
        signal_min = signal_df.groupby('ID')['POS'].min().loc[signal_order]
        signal_max = signal_df.groupby('ID')['POS'].max().loc[signal_order]
        signal_size = signal_max - signal_min
        start_vals = signal_size.cumsum().values[:-1]
        signal_start = pd.Series(data=start_vals, index=signal_size.index[1:])
        signal_start.loc[signal_size.index[0]] = 0
        signal_start = signal_start.loc[signal_size.index]
        signal_mid = signal_start + (signal_size / 2)
        self.base_ax.set_xticks(signal_mid.values)
        self.base_ax.set_xticklabels(signal_mid.index, rotation=30, ha='right')
        odd_signals = signal_size.index[::2]
        even_signals = signal_size.index[1::2]
        pos_adjust = - signal_min.loc[signal_df['ID']] + signal_start.loc[signal_df['ID']]
        signal_df['SIGNAL_X'] = signal_df['POS'] + pos_adjust.values
        signal_df['SIGNAL_TEST'] = signal_df['POS'] - signal_min.loc[signal_df['ID']].values
        self.df['SIGNAL_POS'] = signal_df['SIGNAL_X']
        self.plot_x_col = 'SIGNAL_X'
        self.plot_y_col = self.plot_y_col

        odds_df = signal_df[signal_df['ID'].isin(odd_signals)]
        evens_df = signal_df[signal_df['ID'].isin(even_signals)]

        if self.signal_color_col is None:
            self.base_ax.scatter(odds_df[self.plot_x_col], odds_df[self.plot_y_col], c=self.LIGHT_CHR_COLOR, s=25)
            self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], c=self.DARK_CHR_COLOR, s=25)
        else:
            self.base_ax.scatter(odds_df[self.plot_x_col], odds_df[self.plot_y_col], c='silver', s=25)
            self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], c='dimgrey', s=25)

            filtered_odds_df = odds_df[odds_df['P'] < 1E-3]
            filtered_evens_df = evens_df[evens_df['P'] < 1E-3]

            color_min = min(odds_df[self.signal_color_col].quantile(0.05), evens_df[self.signal_color_col].quantile(0.05))
            color_max = max(odds_df[self.signal_color_col].quantile(0.95), evens_df[self.signal_color_col].quantile(0.95))
            print(color_min, color_max)

            self.base_ax.scatter(filtered_odds_df[self.plot_x_col], filtered_odds_df[self.plot_y_col], s=25,
                                 c=filtered_odds_df[self.signal_color_col], cmap=self.COLOR_MAP, vmin=color_min, vmax=color_max)
            scat = self.base_ax.scatter(filtered_evens_df[self.plot_x_col], filtered_evens_df[self.plot_y_col], s=25,
                                 c=filtered_evens_df[self.signal_color_col], cmap=self.COLOR_MAP, vmin=color_min, vmax=color_max)

            self.fig.colorbar(scat, cax=self.cbar_ax, orientation='horizontal')

        peak_idx = signal_df.groupby('ID')['ROUNDED_Y'].idxmax()
        signal_df = signal_df.rename(columns=extra_cols)
        annot_df = signal_df.loc[peak_idx.values].set_index('ID')
        self.annot_list = [r for _, r in annot_df.iterrows()]

        self.__cosmetic_axis_edits(signals_only=True)
        self.base_ax.set_xlabel('Signal Label')

        if with_table:
            for _, row in annot_df.iterrows():
                self.base_ax.plot([row[self.plot_x_col], row[self.plot_x_col]],
                                  [row[self.plot_y_col], self.max_y],
                                  c='silver', linewidth=1.5, alpha=1)

            self.plot_table(extra_cols=extra_cols, number_cols=number_cols, rep_genes=rep_genes, keep_chr_pos=keep_chr_pos)

        if with_title:
            plt.suptitle('Signals Only:\n' + self.title)
            plt.tight_layout()
        if save is not None:
            plt.savefig(save, dpi=save_res)
        # plt.show()

    def qq_plot(self, save=None, save_res=150, with_title=True, steps=30):
        use_p = self.df['P'].dropna().copy()
        M = len(use_p)
        print(len(self.df) - M, 'SNPs Dropped for Missing P')

        chi2_med = chi2.ppf((1 - use_p).median(), df=1)
        lambda_gc = chi2_med / chi2.ppf(0.5, 1)
        print('Lambda GC:', lambda_gc)

        max_log = -np.log10(1 / M)
        fraction_seq = (np.arange(M) + 1) / M
        p_vec = use_p.sort_values()

        plot_df = p_vec.to_frame()
        plot_df['Exp P'] = fraction_seq
        plot_df[['Log P', 'Log Exp P']] = -np.log10(plot_df[['P', 'Exp P']]).round(3)
        plot_df = plot_df.drop_duplicates(subset=['Log P', 'Log Exp P'])

        plt.gcf().set_size_inches(6, 6)
        plt.gcf().set_facecolor('w')

        plt.plot([0, np.ceil(max_log)], [0, np.ceil(max_log)], c='r', label='Null Model', linestyle='dashed')
        plt.scatter(plot_df['Log Exp P'], plot_df['Log P'], c='blue', label='GWAS Results', s=16)

        plt.xlabel('Expected Log10 P Quantiles')
        plt.ylabel('Observed Log10 P Quantiles')
        plt.legend(loc='upper left')
        plt.annotate('Lambda GC: ' + str(round(lambda_gc, 5)), fontsize=12,
                     xy=(0.95, 0.05), xycoords='axes fraction',
                     horizontalalignment='right', verticalalignment='bottom')
        if with_title:
            plt.title('QQ Plot:\n' + self.title)

        plt.xlim(0, np.ceil(max_log))
        plt.ylim(0, np.ceil(max(plot_df['Log P'])))
        plt.tight_layout()

        if save is not None:
            plt.savefig(save, dpi=save_res)
            plot_df.to_csv(save.replace('.png', '.csv'))

        # plt.show()
        # plt.clf()

    def save_thinned_df(self, path, pickle=True):
        if pickle:
            self.thinned.to_pickle(path)
        else:
            self.thinned.to_csv(path)

    def abacus_phewas_plot(self, save=None, save_res=150, with_title=True):
        self.signal_color_col = 'TRAIT'

        self.__config_axes(with_table=False)

        self.plot_phewas_signals()

        if with_title:
            plt.suptitle(self.title)
            plt.tight_layout()
        if save is not None:
            plt.savefig(save, dpi=save_res)
        # plt.show()
        # plt.clf()

    def plot_phewas_signals(self):
        self.df = self.df[self.df['P'] < 5E-8]
        unique_snps = self.df['ID'].unique()
        x_map = pd.Series(index=unique_snps, data=np.arange(len(unique_snps)) + 1)

        for x in x_map.values:
            if self.log_p_axis_midpoint is None:
                self.base_ax.axvline(x, c='silver', zorder=0)
            else:
                self.lower_base_ax.axvline(x, c='silver', zorder=0)
                self.upper_base_ax.axvline(x, c='silver', zorder=0)

        unique_traits = list(self.df['TRAIT'].dropna().unique())
        categories = sorted(unique_traits)
        cat_to_num = dict(zip(categories, np.arange(len(categories))))
        cat_num_list = [cat_to_num[t] for t in self.df['TRAIT'].dropna()]

        self.fig.set_facecolor('w')

        if self.log_p_axis_midpoint is None:
            scat = self.base_ax.scatter(x=x_map.loc[self.df.dropna(subset='TRAIT')['ID']],
                                        y=-np.log10(self.df.dropna(subset='TRAIT')['P']),
                                        c=cat_num_list,
                                        cmap=plt.cm.get_cmap(self.COLOR_MAP, len(categories)),
                                        s=60, zorder=10)
            self.base_ax.set_xticks(x_map.values)
            self.base_ax.set_xticklabels(x_map.index, rotation=30, ha='right')
            self.base_ax.set_xlabel('Search Identifiers')
            self.base_ax.set_ylabel('-Log10 P Value (Reported)')
            self.base_ax.set_ylim(0, self.max_log_p)
        else:
            self.upper_base_ax.scatter(x=x_map.loc[self.df.dropna(subset='TRAIT')['ID']],
                                       y=-np.log10(self.df.dropna(subset='TRAIT')['P']),
                                       c=cat_num_list,
                                       cmap=plt.cm.get_cmap(self.COLOR_MAP, len(categories)),
                                       s=60, zorder=10)

            scat = self.lower_base_ax.scatter(x=x_map.loc[self.df.dropna(subset='TRAIT')['ID']],
                                              y=-np.log10(self.df.dropna(subset='TRAIT')['P']),
                                              c=cat_num_list,
                                              cmap=plt.cm.get_cmap(self.COLOR_MAP, len(categories)),
                                              s=60, zorder=10)

            self.lower_base_ax.set_xticks(x_map.values)
            self.lower_base_ax.set_xticklabels(x_map.index, rotation=30, ha='right')
            self.lower_base_ax.set_xlabel('Search Identifiers')
            self.lower_base_ax.set_ylabel('-Log10 P Value (Reported)')

            self.lower_base_ax.set_ylim(0, self.log_p_axis_midpoint)
            self.upper_base_ax.set_ylim(self.log_p_axis_midpoint, self.max_log_p)

            if not self.invert:
                self.lower_base_ax.spines['top'].set_visible(False)
                self.upper_base_ax.spines['bottom'].set_visible(False)
            else:
                self.upper_base_ax.spines['top'].set_visible(False)
                self.lower_base_ax.spines['bottom'].set_visible(False)


        print(categories, cat_to_num, unique_traits)
        self.__add_color_bar(scat, categories)

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
        odds_df = [odds_df[odds_df['WRAP'] == i] for i in self.facets]

        evens_df = self.thinned[self.thinned['#CHROM'].isin(evens)].copy()
        evens_df = [evens_df[evens_df['WRAP'] == i] for i in self.facets]

        return odds_df, evens_df

    def __config_axes(self, with_table=True, legend_loc=None):
        need_cbar = (self.signal_color_col is not None) or (self.twas_color_col is not None)

        facet_count = len(self.facets)
        print('Configuring axes for', facet_count, 'facets/boroughs', flush=True)

        if not need_cbar and with_table:
            return
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

        elif need_cbar and with_table and legend_loc is None:
            return
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

        elif need_cbar and with_table and legend_loc == 'top':
            # Horizontal, table, top legend instead of color bar

            print('Horizontal with table and top legend')

            ratios = [0.15, 0.45, 1] if not self.invert else [1, 0.45, 0.15]
            ratios = np.tile(ratios, facet_count)

            self.fig, axes = plt.subplots(nrows=3 * facet_count, ncols=1,
                                          gridspec_kw={'height_ratios': ratios})
            self.fig.set_size_inches(14.4, 5*facet_count)
            self.table_ax = axes[[1, 4, 7]]

            if not self.invert:
                self.cbar_ax = axes[[0, 3, 6]]
                self.base_ax = axes[[2, 5, 8]]
            else:
                self.cbar_ax = axes[[2, 5, 8]]
                self.base_ax = axes[[0, 3, 6]]

        elif need_cbar and with_table and legend_loc == 'side':
            return
            # Horizontal, side legend instead of color bar, table

            print('Horizontal, table, side legend')

            ratios = [0.4, 1] if not self.invert else [1, 0.4]
            self.fig = plt.figure()

            gs0 = self.fig.add_gridspec(1, 2, width_ratios=[1, 0.2])

            gs1 = gs0[0].subgridspec(2, 1, height_ratios=ratios)

            if not self.invert:
                self.table_ax = self.fig.add_subplot(gs1[0])
                self.base_ax = self.fig.add_subplot(gs1[1])
            else:
                self.table_ax = self.fig.add_subplot(gs1[1])
                self.base_ax = self.fig.add_subplot(gs1[0])

            self.cbar_ax = self.fig.add_subplot(gs0[1])

            self.fig.set_size_inches(14.4, 6)

        elif need_cbar and not with_table:
            # Horizontal, color bar, no table
            return
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

        elif not need_cbar and not with_table:
            # Horizontal, no color bar, no table
            return
            self.fig, self.base_ax = plt.subplots()
            self.fig.set_size_inches(13, 3)

        else:
            print('No support for your configuration...')

        if self.invert:
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
            for b in self.base_ax:
                b.plot([end1, end2], [t, t], c=self.FIFTH_COLOR)

    def __cosmetic_axis_edits(self, signals_only=False):

        for i, b in enumerate(self.base_ax):

            pos_col = 'ABS_POS' if not signals_only else 'SIGNAL_POS'

            b.set_xlim(self.df[pos_col].min(), self.df[pos_col].max())


            ax2 = b.twinx()
            ax2.set_ylabel(self.facets[i])
            ax2.tick_params(axis='y', right=False, left=False, labelright=False, labelleft=False)

            b.set_ylabel('- Log10 P')
            b.set_xlabel('Chromosomal Position')
            b.axhline(-np.log10(self.sig_line), c=self.FIFTH_COLOR)
            invisi_spine = 'top' if not self.invert else 'bottom'
            b.spines[invisi_spine].set_visible(False)
            ax2.spines[invisi_spine].set_visible(False)

            if not self.invert:
                b.set_ylim(bottom=np.floor(-np.log10(self.df['P'].max())))
                if self.max_log_p is not None:
                    b.set_ylim(top=self.max_log_p)
                self.max_y = b.get_ylim()[1]
            else:
                b.set_ylim(top=np.floor(-np.log10(self.df['P'].max())))
                if self.max_log_p is not None:
                    b.set_ylim(bottom=self.max_log_p)
                self.max_y = b.get_ylim()[0]

        self.fig.patch.set_facecolor('white')

    def __find_signals_sig(self, rep_genes=[], rep_boost=False):
        odds_df, evens_df = self.__get_odds_evens()
        facet_count = len(self.facets)

        new_odds_df = []
        new_evens_df = []

        for i in range(facet_count):

            odds = odds_df[i]
            evens = evens_df[i]

            halfLD = self.ld_block / 2

            odds['SIGNAL'] = False
            evens['SIGNAL'] = False

            odds['Replication'] = False
            evens['Replication'] = False

            # Signals are always limited by annotation threshold
            annot_mask = self.thinned['P'] < self.annot_thresh
            test_df = self.thinned[annot_mask]
            test_df = test_df[test_df['WRAP'] == self.facets[i]]

            if rep_boost:
                # If boosting known genes, consider suggestive
                p_mask = test_df['P'] < self.sug
            else:
                # If not boosting known genes, consider significant
                p_mask = test_df['P'] < self.sig

            test_df = test_df[p_mask]
            test_df = test_df.sort_values(by='P')

            signal_genes = []
            self.signal_rep_map = {}

            for rowID, row in test_df.iterrows():
                if rep_boost:
                    if row['ID'] not in rep_genes and row['P'] > self.sig:
                        # If boosting known genes, and this gene is novel, use significance threshold
                        continue
                chr_df = odds.iloc[:] if row['#CHROM'] % 2 == 1 else evens.iloc[:]
                if (self.merge_genes or row['ID'] in signal_genes) and chr_df.loc[rowID, 'SIGNAL']:
                    # When not merging genes, test for gene name
                    # When merging genes, test for position
                    continue

                x, gene = row['ROUNDED_X'], row['ID']
                chr_pos_mask = chr_df['ROUNDED_X'].between(x - halfLD, x + halfLD)
                chr_pos_idx = chr_df.index[chr_pos_mask]

                if rep_boost and self.merge_genes and np.any(chr_df.loc[chr_pos_idx, 'ID'].isin(rep_genes)):
                    window_genes = chr_df.loc[chr_pos_idx, ['ID', 'P']].set_index('ID')
                    window_genes = window_genes[window_genes.index.isin(rep_genes)]
                    new_gene = window_genes.idxmin().values[0]
                    if row['#CHROM'] % 2 == 1:
                        odds.loc[chr_pos_idx, 'ID'] = new_gene
                    else:
                        evens.loc[chr_pos_idx, 'ID'] = new_gene
                    self.signal_rep_map[gene] = new_gene
                    gene = new_gene

                currentRep = chr_df.loc[chr_pos_idx, 'Replication']

                if row['#CHROM'] % 2 == 1:
                    odds.loc[chr_pos_idx, 'Replication'] = np.logical_or(currentRep, gene in rep_genes)
                    odds.loc[chr_pos_idx, 'SIGNAL'] = True
                    odds.loc[chr_pos_idx, 'ID'] = gene
                else:
                    evens.loc[chr_pos_idx, 'Replication'] = np.logical_or(currentRep, gene in rep_genes)
                    evens.loc[chr_pos_idx, 'SIGNAL'] = True
                    evens.loc[chr_pos_idx, 'ID'] = gene

                signal_genes.append(gene)

            odds = odds[odds['SIGNAL']]
            new_odds_df.append(odds)
            evens = evens[evens['SIGNAL']]
            new_evens_df.append(evens)

        print('Due to signal merging and replication prioritization, the following genes were renamed:')
        print('\n'.join([k + ': ' + v for k, v in self.signal_rep_map.items()]))

        if len(odds_df) == 0 and len(evens_df) == 0:
            raise ValueError("No signals to annotate. Try making P-value thresholds less stringent")

        return new_odds_df, new_evens_df

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

        signal_bed_df['ROUNDED_X'] = signal_bed_df['ABS_POS'] // self.CHR_POS_ROUND * self.CHR_POS_ROUND

        test_df = self.thinned[self.thinned['P'] < self.annot_thresh].reset_index(drop=False).set_index('ROUNDED_X')

        n = self.ld_block // self.CHR_POS_ROUND // 2 + 1
        keep_locs = []
        for _, row in signal_bed_df.iterrows():
            x = row['ROUNDED_X']
            rounded_locs = x + self.CHR_POS_ROUND * np.arange(-n, n + 1)
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
                    n = self.ld_block // self.CHR_POS_ROUND // 2 + 1
                    start = row['START']
                    end = row['END']

                    rounded_locs = x + self.CHR_POS_ROUND * np.arange(-n, n + 1)

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
        colors = odds_df['Replication'].replace({True: self.REP_HIT_COLOR, False: self.NOVEL_HIT_COLOR})
        self.base_ax.scatter(odds_df[self.plot_x_col], odds_df[self.plot_y_col], c=colors, s=10)
        colors = evens_df['Replication'].replace({True: self.REP_HIT_COLOR, False: self.NOVEL_HIT_COLOR})
        self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], c=colors, s=10)

    def __convert_linear_scale(self, new_min, new_max, data):
        aRange = data.max() - data.min()
        aMin = data.min()
        new_range = new_max - new_min

        new_data = (((data - aMin) / aRange) * new_range) + new_min

        return new_data

    def __plot_color_signals(self, odds_df, evens_df, legend_loc=None):

        unique_vals = list(self.thinned[self.signal_color_col].dropna().unique())
        unique_vals = list(set(unique_vals))

        discrete = ~pd.api.types.is_numeric_dtype(self.thinned[self.signal_color_col])

        for i, b in enumerate(self.base_ax):
            if not discrete:
                color_min = min(odds_df[i][self.signal_color_col].quantile(0.05),
                                evens_df[i][self.signal_color_col].quantile(0.05))
                color_max = max(odds_df[i][self.signal_color_col].quantile(0.95),
                                evens_df[i][self.signal_color_col].quantile(0.95))

                b.scatter(odds_df[i][self.plot_x_col], odds_df[i][self.plot_y_col], c=odds_df[i][self.signal_color_col],
                                     cmap=plt.cm.get_cmap(self.COLOR_MAP), s=10,
                                     vmin=color_min, vmax=color_max)
                scat = b.scatter(evens_df[i][self.plot_x_col], evens_df[i][self.plot_y_col],
                                            c=evens_df[i][self.signal_color_col],
                                            cmap=plt.cm.get_cmap(self.COLOR_MAP), s=10,
                                            vmin=color_min, vmax=color_max)
                self.fig.colorbar(scat, cax=self.cbar_ax, orientation='horizontal')

            else:
                categories = sorted(unique_vals)
                cat_to_num = dict(zip(categories, np.arange(len(categories))))
                odds_df[i]['Cat_Num'] = odds_df[i][self.signal_color_col].replace(cat_to_num)
                evens_df[i]['Cat_Num'] = evens_df[i][self.signal_color_col].replace(cat_to_num)

                odds_df[i]['pt_sz'] = 10
                evens_df[i]['pt_sz'] = 10

                b.scatter(odds_df[i][self.plot_x_col], odds_df[i][self.plot_y_col], c=odds_df[i]['Cat_Num'],
                                     cmap=plt.cm.get_cmap(self.COLOR_MAP, len(categories)), s=odds_df[i]['pt_sz'])
                scat = b.scatter(evens_df[i][self.plot_x_col], evens_df[i][self.plot_y_col], c=evens_df[i]['Cat_Num'],
                                            cmap=plt.cm.get_cmap(self.COLOR_MAP, len(categories)), s=evens_df[i]['pt_sz'])

                self.__add_color_bar(scat, categories, legend_loc=legend_loc)

    def __add_color_bar(self, mappable, categories, legend_loc=None):
        for i, cb in enumerate(self.cbar_ax):
            if legend_loc is None:
                cbar = self.fig.colorbar(mappable, cax=cb, orientation='horizontal')
                xmin, xmax = self.cbar_ax.get_xlim()
                factor = (xmax - xmin) / len(categories)

                categories = [c if len(c) < 20 else c[:17] + '...' for c in categories]

                cbar.set_ticks((np.arange(len(categories)) + 0.5) * factor + xmin)
                if not self.invert:
                    cbar.ax.set_xticklabels(categories, rotation=30, ha='left')
                    cb.xaxis.tick_top()
                else:
                    cbar.ax.set_xticklabels(categories, rotation=30, ha='right')
                self.fig.tight_layout()
            else:
                handles = []
                mappable_cmap = plt.cm.get_cmap(self.COLOR_MAP, len(categories))
                for i in range(len(categories)):
                    handles.append(mpatches.Patch(color=mappable_cmap(mappable.norm(i)), label=categories[i]))

                if legend_loc == 'side':
                    nrows = 14
                    cb.legend(handles=handles, loc='lower left', ncols=max(len(categories) // nrows, 1))
                elif legend_loc == 'top':
                    ncols = 6
                    cb.legend(handles=handles, loc='lower center', ncols=ncols)

                cb.xaxis.set_visible(False)
                cb.yaxis.set_visible(False)
                cb.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
                self.fig.tight_layout()

    def __plot_table_horizontal(self, rep_genes=[], with_table_bg=True, with_table_grid=True, text_rep_colors=False):
        if len(self.annot_list) == 0:
            raise ValueError("No signals to annotate. Try making P-value thresholds less stringent")

        for axi, ta in enumerate(self.table_ax):

            try:
                annotTable = pd.concat(self.annot_list[axi], axis=1).transpose()
            except ValueError:
                ta.set_visible(False)
                continue

            annotTable = annotTable.sort_values(by=['#CHROM', 'POS'])
            genes = [list(annotTable.index)]
            num_cols = len(annotTable)

            table = ta.table(cellText=genes,
                                        loc='lower center',
                                        colWidths=[1 / (num_cols + 2) for g in genes[0]],
                                        cellLoc='center')
            table.AXESPAD = 0

            ta.set_axis_off()
            self.fig.tight_layout()

            if self.twas_color_col is not None:
                unique_vals = sorted(annotTable[self.twas_color_col].unique())
                cmap = plt.cm.get_cmap(self.COLOR_MAP, len(unique_vals))
                fractions = (np.arange(len(unique_vals)) / len(unique_vals)) + (0.5 / len(unique_vals))
                colors = [cmap(f) for f in fractions]
                color_map = dict(zip(unique_vals, colors))

                fractions = list(fractions)
                fractions.append(1.0)
                new_norm = mpl.colors.BoundaryNorm(boundaries=np.arange(len(unique_vals)+1), ncolors=len(unique_vals))
                new_mappable = plt.cm.ScalarMappable(norm=new_norm, cmap=plt.cm.get_cmap(self.COLOR_MAP, len(unique_vals)))
                self.__add_color_bar(new_mappable, color_map.keys())

            cell_width = table[(0, 0)].get_width()
            cell_height = table[(0, 0)].get_height()

            for index, cell in table.get_celld().items():
                if with_table_grid:
                    cell.get_text().set_rotation(90)
                    cell.PAD = 0
                    cell.set_height(1)
                elif not with_table_grid:
                    cell.set_linewidth(0)
                    cell.get_text().set_visible(False)
                    cell.set_height(1)

                cell.get_text().set_fontsize(cell.get_text().get_fontsize() + 5)

            for i in range(num_cols):
                connection_row = annotTable.iloc[i]
                cell_text = table[(0, i)].get_text().get_text()

                if ((cell_text in rep_genes)) and with_table_bg:
                    table[(0, i)].set_facecolor(self.REP_TABLE_COLOR)
                elif with_table_bg:
                    table[(0, i)].set_facecolor(self.NOVEL_TABLE_COLOR)

                if ((cell_text in rep_genes)) and text_rep_colors:
                    table[(0, i)].get_text().set_color(self.DARK_CHR_COLOR)
                elif text_rep_colors:
                    table[(0, i)].get_text().set_color(self.NOVEL_TABLE_COLOR)

                connect_y = 0 if not self.invert else 1
                connect_x = table[(0, i)].get_x() + (0.5*cell_width) if with_table_grid else table[(0, i)].get_x()
                max_ax_y = self.base_ax[axi].get_ylim()[1]

                cp = ConnectionPatch(xyA=(connection_row[self.plot_x_col], max_ax_y),
                                     axesA=self.base_ax[axi], coordsA='data',
                                     xyB=(connect_x, connect_y),
                                     axesB=ta, coordsB='axes fraction',
                                     arrowstyle='-', color='silver')

                if not with_table_grid:
                    if (cell_text in rep_genes) and text_rep_colors:
                        # row_text_color = self.DARK_CHR_COLOR
                        # row_text_color = 'k'
                        row_text_color = 'dimgrey'
                    elif text_rep_colors:
                        row_text_color = self.NOVEL_HIT_COLOR
                    else:
                        row_text_color = 'k'

                    if not self.invert:
                        ta.text(connect_x - 0.005, connect_y, cell_text,
                                horizontalalignment='left',
                                verticalalignment='bottom',
                                rotation=45, transform=ta.transAxes,
                                color=row_text_color)
                    else:
                        ta.text(connect_x + 0.005, connect_y, cell_text,
                                horizontalalignment='right',
                                verticalalignment='top',
                                rotation=45, transform=ta.transAxes,
                                color=row_text_color)

                if self.twas_updown_col is not None:
                    shape = 'v' if connection_row[self.twas_updown_col] < 0 else '^'
                    if self.twas_color_col is None:
                        color = self.REP_HIT_COLOR if cell_text in rep_genes else self.NOVEL_HIT_COLOR
                    else:
                        color = color_map[connection_row[self.twas_color_col]]

                    self.base_ax[axi].scatter(connection_row[self.plot_x_col],
                                         connection_row[self.plot_y_col],
                                         color=color, marker=shape, s=60)

                self.fig.add_artist(cp)
