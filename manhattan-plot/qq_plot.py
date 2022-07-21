import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from manhattan_plot import ManhattanPlot


class QQPlot(ManhattanPlot):

    def __init__(self, file_path: str, test_rows=None, title='QQ Plot'):
        super().__init__(file_path=file_path, test_rows=test_rows, title=title)

    def full_plot(self, save=None, save_res=150, with_title=True, steps=20):
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
        plt.title(self.title)
        plt.xlim(0, max_log)
        plt.ylim(0, max_log)
        plt.tight_layout()
        if save is not None:
            plt.savefig(save, dpi=save_res)
        plt.show()
