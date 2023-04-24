import pandas as pd
from manhattan_plot import ManhattanPlot

if __name__ == "__main__":
    known_genes = open('POAGG_known_genes.txt').read().splitlines()

    mp = ManhattanPlot(file_path='TWAS_GTEX_MEGA_ALL_forPlot.txt',
                       title='African-Ancestry POAAGG Mega-Analysis TWAS',
                       test_rows=None)
    mp.load_data()
    mp.clean_data(col_map={'chr': '#CHROM',
                           'start_position': 'POS',
                           'pvalue': 'P',
                           'gene_name': 'ID'})
    mp.get_thinned_data()

    # Vertical With Table
    mp.update_plotting_parameters(sug=1E-5, annot_thresh=1E-5, sig=5E-6,
                                  ld_block=1E5, merge_genes=False,
                                  invert=False,
                                  twas_color_col='tissue', twas_updown_col='zscore',
                                  vertical=False)

    mp.full_plot(rep_genes=known_genes, rep_boost=True, save='POAGG_MEGA_TWAS_plot.png', save_res=150,
                 keep_chr_pos=False)
