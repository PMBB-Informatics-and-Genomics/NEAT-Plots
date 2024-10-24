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
    phewas_updown_col, phewas_rep_color_col = None, None
    phewas_size_col, phewas_annotate_col, phewas_fill_col = None, None, None
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
    signal_rep_map = {}

    fig, base_ax, table_ax, cbar_ax = None, None, None, None
    lower_base_ax, upper_base_ax = None, None
    annot_list = []
    spec_genes = []

    log_p_axis_midpoint = None

    DARK_CHR_COLOR = '#5841bf'
    LIGHT_CHR_COLOR = '#648fff'
    NOVEL_HIT_COLOR = '#dc267f'
    NOVEL_TABLE_COLOR = '#eb7fb3'
    REP_HIT_COLOR = '#ffbb00'
    REP_TABLE_COLOR = '#ffdc7a'
    FIFTH_COLOR = '#d45c00'
    TABLE_HEAD_COLOR = '#9e9e9e'
    COLOR_MAP = 'turbo_r'
    # COLOR_MAP = 'Paired'

    CHR_POS_ROUND = 5E4
    MIN_PT_SZ = 5
    MAX_PT_SZ = 200

    MIN_TRI_SZ = 5
    MAX_TRI_SZ = 200

    TOP_LEGEND_COLS = 6

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
        mpl.rcParams.update({'font.size': 14})

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

        # In case the #CHROM columns has a chr prefix, remove it
        self.df['#CHROM'] = self.df['#CHROM'].astype(str).str.replace('chr', '')

        # df = df[df['#CHROM'] != 'X']
        chromosomes = list(range(1, 23))
        chromosomes.extend([str(i) for i in range(1, 23)])
        chromosomes.append('X')
        self.df = self.df[self.df['#CHROM'].isin(chromosomes)]
        self.df['#CHROM'] = self.df['#CHROM'].replace('X', 23)
        self.df['#CHROM'] = self.df['#CHROM'].astype(int)

        self.df['POS'] = self.df['POS'].astype(float).astype(int)
        self.df = self.df.sort_values(by=['#CHROM', 'POS'])
        self.df['ID'] = self.df['ID'].fillna('')

        self.df['P'] = pd.to_numeric(self.df['P'], errors='coerce')
        self.df['P'] = self.df['P'].replace(0, self.df['P'].min() / 100)

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

    def update_plotting_parameters(self, log_p_axis_midpoint='', annotate='', signal_color_col='', phewas_rep_color_col='', phewas_updown_col='', phewas_size_col='', phewas_annotate_col='', phewas_fill_col='', twas_color_col='', twas_updown_col='', sig='', sug='', annot_thresh='', ld_block='', vertical='', max_log_p='', invert='', merge_genes='', title=''):
        self.annotate = self.__update_param(self.annotate, annotate)
        self.ld_block = self.__update_param(self.ld_block, ld_block)
        self.title = self.__update_param(self.title, title)
        self.log_p_axis_midpoint = self.__update_param(self.log_p_axis_midpoint, log_p_axis_midpoint)

        self.signal_color_col = self.__update_param(self.signal_color_col, signal_color_col)

        self.phewas_rep_color_col = self.__update_param(self.phewas_rep_color_col, phewas_rep_color_col)
        self.phewas_updown_col = self.__update_param(self.phewas_updown_col, phewas_updown_col)
        self.phewas_size_col = self.__update_param(self.phewas_size_col, phewas_size_col)
        self.phewas_annotate_col = self.__update_param(self.phewas_annotate_col, phewas_annotate_col)
        self.phewas_fill_col = self.__update_param(self.phewas_fill_col, phewas_fill_col)

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

    def get_thinned_data(self, log_p_round=2, additional_cols=[]):
        if 'ABS_POS' not in self.df.columns:
            self.df['ABS_POS'] = self.__get_absolute_positions(self.df)

        self.thinned = self.df.copy()
        self.thinned['ROUNDED_X'] = self.thinned['ABS_POS'] // self.CHR_POS_ROUND * self.CHR_POS_ROUND
        self.thinned['ROUNDED_Y'] = pd.Series(-np.log10(self.thinned['P'])).round(log_p_round)  # round to 2 decimals
        subset_cols = ['ROUNDED_X', 'ROUNDED_Y']
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
        if self.base_ax is None:
            self.__config_axes(with_table=with_table, legend_loc=legend_loc)

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
            self.base_ax.scatter(odds_df[self.plot_x_col], odds_df[self.plot_y_col], c=self.LIGHT_CHR_COLOR, s=2)
            self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], c=self.DARK_CHR_COLOR, s=2)
        else:
            if self.phewas_size_col is None:
                odds_df['pt_sz'] = 2
                evens_df['pt_sz'] = 2
            else:
                min_size, max_size = self.MIN_PT_SZ, self.MAX_PT_SZ
                odds_df['pt_sz'] = self.__convert_linear_scale(data=odds_df[self.phewas_size_col].abs(), new_min=min_size, new_max=max_size)
                evens_df['pt_sz'] = self.__convert_linear_scale(data=evens_df[self.phewas_size_col].abs(), new_min=min_size, new_max=max_size)

            if self.phewas_updown_col is None:
                self.base_ax.scatter(odds_df[self.plot_x_col], odds_df[self.plot_y_col], c='silver', s=odds_df['pt_sz'])
                self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], c='dimgray', s=evens_df['pt_sz'])
            else:
                self.base_ax.scatter(odds_df[self.plot_x_col], odds_df[self.plot_y_col], edgecolors='silver', facecolors='none', s=odds_df['pt_sz'], alpha=1, linewidth=0.2)
                self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], edgecolors='dimgray', facecolors='none', s=evens_df['pt_sz'], alpha=1, linewidth=0.2)

        self.__add_threshold_ticks()
        self.__cosmetic_axis_edits()

    def plot_specific_signals(self, signal_bed_df, rep_genes=[], legend_loc=None):
        odds_df, evens_df = self.__find_signals_specific(signal_bed_df, rep_genes=rep_genes)

        if self.signal_color_col is None:
            self.__plot_signals(odds_df, evens_df)
        else:
            self.__plot_color_signals(odds_df, evens_df, legend_loc=legend_loc)

    def plot_sig_signals(self, rep_genes=[], rep_boost=False, legend_loc=None):
        odds_df, evens_df = self.__find_signals_sig(rep_genes, rep_boost)

        if self.signal_color_col is None:
            self.__plot_signals(odds_df, evens_df)
        else:
            self.__plot_color_signals(odds_df, evens_df, legend_loc=legend_loc)

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

    def plot_table(self, extra_cols={}, number_cols=[], rep_genes=[], keep_chr_pos=True, with_table_bg=True, with_table_grid=True, text_rep_colors=False, table_fontsize=12):
        if self.vertical:
            self.__plot_table_vertical(extra_cols=extra_cols, number_cols=number_cols, rep_genes=rep_genes, keep_chr_pos=keep_chr_pos, table_fontsize=table_fontsize)
        else:
            self.__plot_table_horizontal(rep_genes=rep_genes, with_table_bg=with_table_bg, with_table_grid=with_table_grid, text_rep_colors=text_rep_colors)

    def full_plot(self, rep_genes=[], extra_cols={}, number_cols=[], rep_boost=False, save=None, with_table=True,
                  save_res=150, with_title=True, keep_chr_pos=True, with_table_bg=True, with_table_grid=True,
                  legend_loc=None, text_rep_colors=False, table_fontsize=12):
        self.plot_data(with_table=with_table, legend_loc=legend_loc)
        self.plot_sig_signals(rep_genes=rep_genes, rep_boost=rep_boost, legend_loc=legend_loc)
        if with_table:
            if self.phewas_annotate_col is None:
                self.plot_annotations(rep_genes=rep_genes, rep_boost=rep_boost)
            else:
                self.__plot_pointers_only()
            self.plot_table(extra_cols=extra_cols, number_cols=number_cols, rep_genes=rep_genes, keep_chr_pos=keep_chr_pos,
                            with_table_bg=with_table_bg, with_table_grid=with_table_grid, text_rep_colors=text_rep_colors, table_fontsize=table_fontsize)
        if with_title:
            plt.suptitle(self.title)
            plt.tight_layout()
        if save is not None:
            plt.savefig(save, dpi=save_res, bbox_inches='tight')
        # plt.show()
        # plt.clf()

    def signal_plot(self, rep_genes=[], extra_cols={}, number_cols=[], rep_boost=False, save=None, with_table=True,
                    save_res=150, with_title=True, keep_chr_pos=True, table_fontsize=12):
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
        if self.vertical:
            self.base_ax.set_yticks(signal_mid.values)
            self.base_ax.set_yticklabels(signal_mid.index)
        else:
            self.base_ax.set_xticks(signal_mid.values)
            self.base_ax.set_xticklabels(signal_mid.index, rotation=30, ha='right')
        odd_signals = signal_size.index[::2]
        even_signals = signal_size.index[1::2]
        pos_adjust = - signal_min.loc[signal_df['ID']] + signal_start.loc[signal_df['ID']]
        signal_df['SIGNAL_X'] = signal_df['POS'] + pos_adjust.values
        signal_df['SIGNAL_TEST'] = signal_df['POS'] - signal_min.loc[signal_df['ID']].values
        self.df['SIGNAL_POS'] = signal_df['SIGNAL_X']
        self.plot_x_col = 'SIGNAL_X' if not self.vertical else self.plot_x_col
        self.plot_y_col = 'SIGNAL_X' if self.vertical else self.plot_y_col

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
        if self.vertical:
            self.base_ax.set_ylabel('Signal Label')
            self.base_ax.grid(visible=False, which='both', axis='x')
        else:
            self.base_ax.set_xlabel('Signal Label')
            self.base_ax.grid(visible=False, which='both', axis='y')

        if with_table:
            for _, row in annot_df.iterrows():
                if self.vertical:
                    self.base_ax.plot([row[self.plot_x_col], self.max_x],
                                      [row[self.plot_y_col], row[self.plot_y_col]],
                                      c='silver', linewidth=1.5, alpha=1)
                else:
                    self.base_ax.plot([row[self.plot_x_col], row[self.plot_x_col]],
                                      [row[self.plot_y_col], self.max_y],
                                      c='silver', linewidth=1.5, alpha=1)

            self.plot_table(extra_cols=extra_cols, number_cols=number_cols, rep_genes=rep_genes, keep_chr_pos=keep_chr_pos, table_fontsize=table_fontsize)

        if with_title:
            plt.suptitle('Signals Only:\n' + self.title)
            plt.tight_layout()
        if save is not None:
            plt.savefig(save, dpi=save_res, bbox_inches='tight')
        # plt.show()

    def full_plot_with_specific(self, signal_bed_df, plot_sig=True, rep_boost=False, rep_genes=[], extra_cols={},
                                number_cols=[], verbose=False, save=None, save_res=150, keep_chr_pos=True, with_table_bg=True,
                                with_table_grid=True, legend_loc=None, with_table=True, table_fontsize=12):
        if verbose:
            print('Plotting All Data...', flush=True)
        self.plot_data(with_table=with_table)
        if plot_sig:
            if verbose:
                print('Plotting Significant Signals...', flush=True)
            self.plot_sig_signals()
        if verbose:
            print('Plotting Specific Signals...', flush=True)
        self.plot_specific_signals(signal_bed_df, rep_genes=rep_genes, legend_loc=legend_loc)
        if with_table:
            if verbose:
                print('Finding Annotations...', flush=True)
            self.plot_annotations(plot_sig=plot_sig, rep_genes=rep_genes, rep_boost=rep_boost)
            if verbose:
                print('Adding Table...', flush=True)
            self.plot_table(extra_cols=extra_cols, number_cols=number_cols, rep_genes=rep_genes, keep_chr_pos=keep_chr_pos,
                            with_table_grid=with_table_grid, with_table_bg=with_table_bg, table_fontsize=table_fontsize)
        if save is not None:
            if verbose:
                print('Writing Figure to File...', flush=True)
            plt.savefig(save, dpi=save_res, bbox_inches='tight')
        # plt.show()
        # plt.clf()

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
            plt.savefig(save, dpi=save_res, bbox_inches='tight')
            plot_df.to_csv(save.replace('.png', '.csv'))

        # plt.show()
        # plt.clf()

    def save_thinned_df(self, path, pickle=True):
        if pickle:
            self.thinned.to_pickle(path)
        else:
            self.thinned.to_csv(path)

    def abacus_phewas_plot(self, save=None, save_res=150, with_title=True):
        self.vertical = False
        self.signal_color_col = 'TRAIT'

        self.__config_axes(with_table=False)

        self.plot_phewas_signals()

        if with_title:
            plt.suptitle(self.title)
            plt.tight_layout()
        if save is not None:
            plt.savefig(save, dpi=save_res, bbox_inches='tight')
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
        lastPos['POS'] = lastPos['POS'].astype(float).astype(np.int64)
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

    def __config_axes(self, with_table=True, legend_loc=None):
        need_cbar = (self.signal_color_col is not None) or (self.twas_color_col is not None)

        if self.log_p_axis_midpoint is None:
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

            elif not self.vertical and need_cbar and with_table and legend_loc is None:
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

            elif not self.vertical and need_cbar and with_table and legend_loc == 'top':
                # Horizontal, table, top legend instead of color bar
                ratios = [0.15, 0.45, 1] if not self.invert else [1, 0.45, 0.15]
                self.fig, axes = plt.subplots(nrows=3, ncols=1, gridspec_kw={'height_ratios': ratios})
                self.fig.set_size_inches(14.4, 6)
                self.table_ax = axes[1]

                if not self.invert:
                    self.cbar_ax = axes[0]
                    self.base_ax = axes[2]
                else:
                    self.cbar_ax = axes[2]
                    self.base_ax = axes[0]

            elif not self.vertical and need_cbar and with_table and legend_loc == 'side':
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
                raise ValueError('No support for your configuration...')

        elif self.log_p_axis_midpoint is not None:

            if not self.vertical and need_cbar and with_table:
                # Horizontal, color bar, table
                ratios = [0.08, 0.45, 0.5, 0.5] if not self.invert else [0.5, 0.5, 0.45, 0.08]

                self.fig, axes = plt.subplots(nrows=4, ncols=1,
                                              gridspec_kw={'height_ratios': ratios, 'hspace': 0.05})

                self.fig.set_size_inches(14.4, 6)

                if not self.invert:
                    self.cbar_ax = axes[0]
                    self.table_ax = axes[1]
                    self.upper_base_ax = axes[2]
                    self.lower_base_ax = axes[3]
                else:
                    self.cbar_ax = axes[3]
                    self.table_ax = axes[2]
                    self.upper_base_ax = axes[1]
                    self.lower_base_ax = axes[0]

            elif not self.vertical and need_cbar and not with_table:
                # Horizontal, color bar, no table
                ratios = [0.08, 0.5, 0.5] if not self.invert else [0.5, 0.5, 0.08]
                self.fig, axes = plt.subplots(nrows=3, ncols=1,
                                              gridspec_kw={'height_ratios': ratios, 'hspace': 0})
                self.fig.set_size_inches(14.4, 4)

                if not self.invert:
                    self.cbar_ax = axes[0]
                    self.upper_base_ax = axes[1]
                    self.lower_base_ax = axes[2]
                else:
                    self.cbar_ax = axes[2]
                    self.upper_base_ax = axes[1]
                    self.lower_base_ax = axes[0]

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
                self.base_ax.plot([t, t], [end1, end2], c=self.FIFTH_COLOR)
            else:
                self.base_ax.plot([end1, end2], [t, t], c=self.FIFTH_COLOR)

    def __cosmetic_axis_edits(self, signals_only=False):
        pos_col = 'ABS_POS' if not signals_only else 'SIGNAL_POS'
        if self.vertical:
            self.base_ax.set_ylim(self.df[pos_col].min(), self.df[pos_col].max())
            self.base_ax.set_xlabel('- Log10 P')
            self.base_ax.set_ylabel('Chromosomal Position')
            self.base_ax.axvline(-np.log10(self.sig_line), c=self.FIFTH_COLOR)
            self.base_ax.invert_yaxis()
            invisi_spine = 'right' if not self.invert else 'left'
            self.base_ax.spines[invisi_spine].set_visible(False)

            print(self.thinned[self.plot_x_col].min())
            print(self.plot_x_col)

            if not self.invert:
                self.base_ax.set_xlim(left=self.thinned[self.plot_x_col].min())
                if self.max_log_p is not None:
                    self.base_ax.set_xlim(right=self.max_log_p)
                self.max_x = self.base_ax.get_xlim()[1]
            else:
                self.base_ax.set_xlim(right=self.thinned[self.plot_x_col].min())
                if self.max_log_p is not None:
                    self.base_ax.set_xlim(left=self.max_log_p)
                self.max_x = self.base_ax.get_xlim()[0]
        else:
            self.base_ax.set_xlim(self.df[pos_col].min(), self.df[pos_col].max())
            self.base_ax.set_ylabel('- Log10 P')
            self.base_ax.set_xlabel('Chromosomal Position')
            self.base_ax.axhline(-np.log10(self.sig_line), c=self.FIFTH_COLOR)
            invisi_spine = 'top' if not self.invert else 'bottom'
            self.base_ax.spines[invisi_spine].set_visible(False)

            if not self.invert:
                self.base_ax.set_ylim(bottom=np.floor(-np.log10(self.df['P'].max())))
                if self.max_log_p is not None:
                    self.base_ax.set_ylim(top=self.max_log_p)
                self.max_y = self.base_ax.get_ylim()[1]
            else:
                self.base_ax.set_ylim(top=np.floor(-np.log10(self.df['P'].max())))
                if self.max_log_p is not None:
                    self.base_ax.set_ylim(cottom=self.max_log_p)
                self.max_y = self.base_ax.get_ylim()[0]

        self.fig.patch.set_facecolor('white')

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
        self.signal_rep_map = {}

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

            if rep_boost and self.merge_genes and np.any(chr_df.loc[chr_pos_idx, 'ID'].isin(rep_genes)):
                window_genes = chr_df.loc[chr_pos_idx, ['ID', 'P']].set_index('ID')
                window_genes = window_genes[window_genes.index.isin(rep_genes)]
                new_gene = window_genes.idxmin().values[0]
                if row['#CHROM'] % 2 == 1:
                    odds_df.loc[chr_pos_idx, 'ID'] = new_gene
                else:
                    evens_df.loc[chr_pos_idx, 'ID'] = new_gene
                self.signal_rep_map[gene] = new_gene
                gene = new_gene

            currentRep = chr_df.loc[chr_pos_idx, 'Replication']

            if row['#CHROM'] % 2 == 1:
                odds_df.loc[chr_pos_idx, 'Replication'] = np.logical_or(currentRep, gene in rep_genes)
                odds_df.loc[chr_pos_idx, 'SIGNAL'] = True
                odds_df.loc[chr_pos_idx, 'ID'] = gene
            else:
                evens_df.loc[chr_pos_idx, 'Replication'] = np.logical_or(currentRep, gene in rep_genes)
                evens_df.loc[chr_pos_idx, 'SIGNAL'] = True
                evens_df.loc[chr_pos_idx, 'ID'] = gene

            signal_genes.append(gene)

        odds_df = odds_df[odds_df['SIGNAL']]
        evens_df = evens_df[evens_df['SIGNAL']]

        print('Due to signal merging and replication prioritization, the following genes were renamed:')
        print('\n'.join([k + ': ' + v for k, v in self.signal_rep_map.items()]))

        if len(odds_df) == 0 and len(evens_df) == 0:
            raise ValueError("No signals to annotate. Try making P-value thresholds less stringent")

        return odds_df, evens_df

    def __find_signals_specific(self, signal_bed_df, rep_genes=[]):
        odds_df, evens_df = self.__get_odds_evens()
        self.spec_genes = []

        odds_df['SIGNAL'] = False
        evens_df['SIGNAL'] = False

        odds_df['Replication'] = False
        evens_df['Replication'] = False

        signal_bed_df = signal_bed_df[signal_bed_df['#CHROM'].isin(self.df['#CHROM'])].copy()
        signal_bed_df = signal_bed_df.sort_values(by=['#CHROM', 'POS'])

        for data_df in [odds_df, evens_df]:
            search_n = len(signal_bed_df)
            data_m = len(data_df)
            shape_2D = (search_n, data_m)
            shape_2D_T = (data_m, search_n)

            search_starts = np.broadcast_to(signal_bed_df['START'], shape_2D_T).T
            search_stops = np.broadcast_to(signal_bed_df['END'], shape_2D_T).T
            search_chr = np.broadcast_to(signal_bed_df['#CHROM'], shape_2D_T).T

            data_pos = np.broadcast_to(data_df['POS'], shape_2D)
            data_chr = np.broadcast_to(data_df['#CHROM'], shape_2D)

            chr_match = search_chr == data_chr
            pos_match = (search_starts < data_pos) & (data_pos < search_stops)
            overlap = chr_match & pos_match

            for i in range(len(signal_bed_df)):
                row = signal_bed_df.iloc[i]

                overlap_mask = overlap[i]
                keep_locs = data_df.index[overlap_mask]
                if len(keep_locs) == 0:
                    continue

                gene_df = data_df.loc[keep_locs].copy().reset_index(drop=False).set_index('index')
                if 'ID' not in row.index:
                    gene = gene_df.sort_values(by='P')['ID'].iloc[0]
                else:
                    gene = row['ID']

                self.thinned.loc[gene_df.index, 'ID'] = gene
                if self.signal_color_col is not None and False in pd.isnull(gene_df[self.signal_color_col]):
                    self.thinned.loc[gene_df.index, self.signal_color_col] = \
                    gene_df[self.signal_color_col].mode().iloc[0]

                if self.thinned.loc[self.thinned['ID'] == gene, 'P'].min() > self.sug:
                    continue

                self.spec_genes.append(gene)

                signal_pos_index = data_df.index[overlap_mask]
                signal_index = data_df.index.intersection(signal_pos_index)

                data_df.loc[signal_index, 'SIGNAL'] = True
                data_df.loc[signal_index, 'Replication'] = gene in rep_genes

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
        if self.phewas_rep_color_col is not None:
            # Filter points that get signal colors by Known == True
            odds_df = odds_df[~odds_df[self.phewas_rep_color_col]].copy()
            evens_df = evens_df[~evens_df[self.phewas_rep_color_col]].copy()
            # self.base_ax.set_yticks(np.arange(0, 350, 20))
            self.fig.set_size_inches(14.4, 8)

        unique_vals = list(odds_df[self.signal_color_col].dropna().unique())
        unique_vals.extend(list(evens_df[self.signal_color_col].dropna().unique()))
        unique_vals = list(set(unique_vals))

        discrete = ~pd.api.types.is_numeric_dtype(odds_df[self.signal_color_col])

        if not discrete:
            color_min = min(odds_df[self.signal_color_col].quantile(0.05),
                            evens_df[self.signal_color_col].quantile(0.05))
            color_max = max(odds_df[self.signal_color_col].quantile(0.95),
                            evens_df[self.signal_color_col].quantile(0.95))

            self.base_ax.scatter(odds_df[self.plot_x_col], odds_df[self.plot_y_col], c=odds_df[self.signal_color_col],
                                 cmap=plt.cm.get_cmap(self.COLOR_MAP), s=10,
                                 vmin=color_min, vmax=color_max)
            scat = self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col],
                                        c=evens_df[self.signal_color_col],
                                        cmap=plt.cm.get_cmap(self.COLOR_MAP), s=10,
                                        vmin=color_min, vmax=color_max)
            self.fig.colorbar(scat, cax=self.cbar_ax, orientation='horizontal')

        else:
            categories = sorted(unique_vals)
            cat_to_num = dict(zip(categories, np.arange(len(categories))))
            odds_df['Cat_Num'] = odds_df[self.signal_color_col].replace(cat_to_num)
            evens_df['Cat_Num'] = evens_df[self.signal_color_col].replace(cat_to_num)

            if self.phewas_updown_col is None:
                if self.phewas_size_col is None:
                    odds_df['pt_sz'] = 10
                    evens_df['pt_sz'] = 10
                else:
                    min_size, max_size = self.MIN_PT_SZ, self.MAX_PT_SZ
                    odds_df['pt_sz'] = self.__convert_linear_scale(data=odds_df[self.phewas_size_col].abs(), new_min=min_size, new_max=max_size)
                    evens_df['pt_sz'] = self.__convert_linear_scale(data=evens_df[self.phewas_size_col].abs(), new_min=min_size, new_max=max_size)

                self.base_ax.scatter(odds_df[self.plot_x_col], odds_df[self.plot_y_col], c=odds_df['Cat_Num'],
                                     cmap=plt.cm.get_cmap(self.COLOR_MAP, len(categories)), s=odds_df['pt_sz'])
                scat = self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], c=evens_df['Cat_Num'],
                                            cmap=plt.cm.get_cmap(self.COLOR_MAP, len(categories)), s=evens_df['pt_sz'])

            elif self.phewas_updown_col is not None:
                if self.phewas_size_col is None:
                    odds_df['pt_sz'] = 60
                    evens_df['pt_sz'] = 60
                else:
                    min_size, max_size = self.MIN_TRI_SZ, self.MAX_TRI_SZ
                    odds_df['pt_sz'] = self.__convert_linear_scale(data=odds_df[self.phewas_size_col].abs(), new_min=min_size, new_max=max_size)
                    evens_df['pt_sz'] = self.__convert_linear_scale(data=evens_df[self.phewas_size_col].abs(), new_min=min_size, new_max=max_size)

                odds_df['up'] = (odds_df[self.phewas_updown_col] > 0)
                evens_df['up'] = (evens_df[self.phewas_updown_col] > 0)

                for updown, subDF in odds_df.groupby('up'):
                    shape = '^' if updown else 'v'
                    if self.phewas_fill_col is None:
                        self.base_ax.scatter(subDF[self.plot_x_col], subDF[self.plot_y_col], c=subDF['Cat_Num'],
                                             cmap=plt.cm.get_cmap(self.COLOR_MAP, len(categories)),
                                             s=subDF['pt_sz'], marker=shape, edgecolors='k', linewidth=0.3)
                    elif self.phewas_fill_col is not None:
                        print('Updown and Fill', flush=True)
                        color_map = plt.cm.get_cmap(self.COLOR_MAP, len(categories))
                        edge_color_list = subDF['Cat_Num'].apply(lambda x: color_map(x))
                        face_color_list = subDF[['Cat_Num', self.phewas_fill_col]].apply(lambda x: color_map(x['Cat_Num']) if x[self.phewas_fill_col] else 'none', axis=1)
                        print(face_color_list.value_counts())
                        self.base_ax.scatter(subDF[self.plot_x_col], subDF[self.plot_y_col],
                                             s=subDF['pt_sz'], marker=shape,
                                             edgecolors=edge_color_list.values, linewidth=1,
                                             facecolors=face_color_list.values)

                        unique_vals = sorted(categories)
                        cmap = plt.cm.get_cmap(self.COLOR_MAP, len(unique_vals))
                        fractions = (np.arange(len(unique_vals)) / len(unique_vals)) + (0.5 / len(unique_vals))
                        colors = [cmap(f) for f in fractions]
                        color_map = dict(zip(unique_vals, colors))

                        fractions = list(fractions)
                        fractions.append(1.0)
                        new_norm = mpl.colors.BoundaryNorm(boundaries=np.arange(len(unique_vals) + 1),
                                                           ncolors=len(unique_vals))
                        scat = plt.cm.ScalarMappable(norm=new_norm,
                                                     cmap=plt.cm.get_cmap(self.COLOR_MAP, len(unique_vals)))

                for updown, subDF in evens_df.groupby('up'):
                    shape = '^' if updown else 'v'
                    if self.phewas_fill_col is None:
                        scat = self.base_ax.scatter(subDF[self.plot_x_col], subDF[self.plot_y_col],
                                                    c=subDF['Cat_Num'], cmap=plt.cm.get_cmap(self.COLOR_MAP, len(categories)),
                                                    s=subDF['pt_sz'], marker=shape, edgecolors='k', linewidth=0.3)
                    elif self.phewas_fill_col is not None:
                        print('Updown and Fill', flush=True)
                        color_map = plt.cm.get_cmap(self.COLOR_MAP, len(categories))
                        edge_color_list = subDF['Cat_Num'].apply(lambda x: color_map(x))
                        face_color_list = subDF[['Cat_Num', self.phewas_fill_col]].apply(lambda x: color_map(x['Cat_Num']) if x[self.phewas_fill_col] else 'none', axis=1)
                        self.base_ax.scatter(subDF[self.plot_x_col], subDF[self.plot_y_col],
                                             s=subDF['pt_sz'], marker=shape,
                                             edgecolors=edge_color_list.values, linewidth=1,
                                             facecolors=face_color_list.values)

                        unique_vals = sorted(categories)
                        cmap = plt.cm.get_cmap(self.COLOR_MAP, len(unique_vals))
                        fractions = (np.arange(len(unique_vals)) / len(unique_vals)) + (0.5 / len(unique_vals))
                        colors = [cmap(f) for f in fractions]
                        color_map = dict(zip(unique_vals, colors))

                        fractions = list(fractions)
                        fractions.append(1.0)
                        new_norm = mpl.colors.BoundaryNorm(boundaries=np.arange(len(unique_vals) + 1),
                                                           ncolors=len(unique_vals))
                        scat = plt.cm.ScalarMappable(norm=new_norm, cmap=plt.cm.get_cmap(self.COLOR_MAP, len(unique_vals)))

            self.__add_color_bar(scat, categories, legend_loc=legend_loc)

    def __plot_pointers_only(self):

        signalDF = self.thinned[self.thinned[self.phewas_annotate_col]]

        for _, row in signalDF.iterrows():
            if self.vertical:
                if self.max_log_p is not None:
                    pointer_x = signalDF[signalDF[self.plot_x_col] <= self.max_log_p][self.plot_x_col].max()
                else:
                    pointer_x = signalDF[self.plot_x_col].max()
                self.base_ax.plot([pointer_x, self.max_x],
                                  [row[self.plot_y_col], row[self.plot_y_col]],
                                  c='silver', linewidth=1.5, alpha=1)
            else:
                pointer_y = row[self.plot_y_col]
                self.base_ax.plot([row[self.plot_x_col], row[self.plot_x_col]],
                                  [pointer_y, self.max_y],
                                  c='silver', linewidth=1.5, alpha=1)

        self.annot_list.append(row)

    def __add_color_bar(self, mappable, categories, legend_loc=None):
        if legend_loc is None:
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
        else:
            plt.rc('legend', fontsize=12)
            handles = []
            mappable_cmap = mappable.get_cmap()
            for i in range(len(categories)):
                handles.append(mpatches.Patch(color=mappable_cmap(i), label=categories[i]))

            if legend_loc == 'side':
                nrows = 14
                self.cbar_ax.legend(handles=handles, loc='lower left', ncols=max(len(categories) // nrows, 1))
            elif legend_loc == 'top':
                ncols = self.TOP_LEGEND_COLS
                self.cbar_ax.legend(handles=handles, loc='lower center', ncols=ncols)

            self.cbar_ax.xaxis.set_visible(False)
            self.cbar_ax.yaxis.set_visible(False)
            self.cbar_ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
            self.fig.tight_layout()

    def __plot_table_vertical(self, extra_cols={}, number_cols=[], rep_genes=[], keep_chr_pos=True, table_fontsize=12):
        if len(self.annot_list) == 0:
            raise ValueError("No signals to annotate. Try making P-value thresholds less stringent")

        if keep_chr_pos:
            columns = ['ID', 'CHR', 'POS', 'P']
        else:
            columns = ['ID', 'P']

        columns.extend(extra_cols.values())

        annot_table = pd.concat(self.annot_list, axis=1).transpose()
        annot_table = annot_table.sort_values(by=['#CHROM', 'POS'])
        annot_table = annot_table.reset_index()
        annot_table = annot_table.rename(columns={'#CHROM': 'CHR',
                                                  'index': 'ID'})
        annot_table = annot_table.rename(columns=extra_cols)
        annot_table['P'] = annot_table['P'].apply(lambda x: '{:.2e}'.format(x))
        annot_table['ID'] = annot_table['ID'].apply(lambda x: '$\it{' + x + '}$')
        annot_table[number_cols] = annot_table[number_cols].map(lambda x: '{:.3}'.format(x))

        location = 'center left' if not self.invert else 'center right'

        table = mpl.table.table(ax=self.table_ax,
                                cellText=annot_table[columns].fillna('').values,
                                colLabels=columns, loc=location,
                                colColours=[self.TABLE_HEAD_COLOR for c in columns])
        table.AXESPAD = 0

        table.auto_set_font_size(False)
        table.set_fontsize(table_fontsize)
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
            cell_text = cell.get_text().get_text().replace('$\it{', '').replace('}$', '')
            if cell_text in rep_genes:
                cell.set_facecolor(self.REP_TABLE_COLOR)
            else:
                cell.set_facecolor(self.NOVEL_TABLE_COLOR)
            connect_x = 0
            cp = ConnectionPatch(xyA=(self.max_x, connection_row[self.plot_y_col]),
                                 axesA=self.base_ax, coordsA='data',
                                 xyB=(connect_x, (1 - cell.get_y()) - (0.5*cell_height)),
                                 axesB=self.table_ax, coordsB='data',
                                 arrowstyle='-', color='silver')
            self.fig.add_artist(cp)

    def __plot_table_horizontal(self, rep_genes=[], with_table_bg=True, with_table_grid=True, text_rep_colors=False):
        if len(self.annot_list) == 0 and self.phewas_annotate_col is None:
            raise ValueError("No signals to annotate. Try making P-value thresholds less stringent")

        if self.phewas_annotate_col is None:
            annotTable = pd.concat(self.annot_list, axis=1).transpose()
        else:
            annotTable = self.thinned[self.thinned[self.phewas_annotate_col]].set_index('ID')

        annotTable = annotTable.sort_values(by=['#CHROM', 'POS'])
        annotTable.index = ['$\it{' + i + '}$' for i in annotTable.index]
        genes = [list(annotTable.index)]
        num_cols = len(annotTable)

        table = self.table_ax.table(cellText=genes,
                                    loc='lower center',
                                    colWidths=[1 / (num_cols + 2) for g in genes[0]],
                                    cellLoc='center')
        table.AXESPAD = 0

        self.table_ax.set_axis_off()
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
            cell_text = table[(0, i)].get_text().get_text()[5:-2]

            if ((cell_text in rep_genes) or (self.phewas_rep_color_col is not None and connection_row[self.phewas_rep_color_col])) and with_table_bg:
                table[(0, i)].set_facecolor(self.REP_TABLE_COLOR)
            elif with_table_bg:
                table[(0, i)].set_facecolor(self.NOVEL_TABLE_COLOR)

            if ((cell_text in rep_genes) or (self.phewas_rep_color_col is not None and connection_row[self.phewas_rep_color_col])) and text_rep_colors:
                table[(0, i)].get_text().set_color(self.DARK_CHR_COLOR)
            elif text_rep_colors:
                table[(0, i)].get_text().set_color(self.NOVEL_TABLE_COLOR)

            connect_y = 0 if not self.invert else 1
            connect_x = table[(0, i)].get_x() + (0.5*cell_width) if with_table_grid else table[(0, i)].get_x()
            cp = ConnectionPatch(xyA=(connection_row[self.plot_x_col], self.max_y),
                                 axesA=self.base_ax, coordsA='data',
                                 xyB=(connect_x, connect_y),
                                 axesB=self.table_ax, coordsB='axes fraction',
                                 arrowstyle='-', color='silver')

            if not with_table_grid:
                if (cell_text in rep_genes) or (self.phewas_rep_color_col is not None and connection_row[self.phewas_rep_color_col]) and text_rep_colors:
                    # row_text_color = self.DARK_CHR_COLOR
                    # row_text_color = 'k'
                    row_text_color = 'dimgrey'
                elif text_rep_colors:
                    row_text_color = self.NOVEL_HIT_COLOR
                else:
                    row_text_color = 'k'

                if not self.invert:
                    self.table_ax.text(connect_x - 0.005, connect_y, cell_text,
                                       horizontalalignment='left',
                                       verticalalignment='bottom',
                                       rotation=45, transform=self.table_ax.transAxes,
                                       color=row_text_color)
                else:
                    self.table_ax.text(connect_x + 0.005, connect_y, cell_text,
                                       horizontalalignment='right',
                                       verticalalignment='top',
                                       rotation=45, transform=self.table_ax.transAxes,
                                       color=row_text_color)

            if self.twas_updown_col is not None:
                shape = 'v' if connection_row[self.twas_updown_col] < 0 else '^'
                if self.twas_color_col is None:
                    color = self.REP_HIT_COLOR if cell_text in rep_genes else self.NOVEL_HIT_COLOR
                else:
                    color = color_map[connection_row[self.twas_color_col]]

                self.base_ax.scatter(connection_row[self.plot_x_col],
                                     connection_row[self.plot_y_col],
                                     color=color, marker=shape, s=60)

            self.fig.add_artist(cp)
