import pandas as pd
from manhattan_plot import ManhattanPlot

if __name__ == "__main__":
    annotDF = pd.read_csv('meta_suggestive_biofilter_genes_rsids_UPDATED.csv')
    annotDF = annotDF.rename(columns={'Gene': 'ID'})

    known_genes = open('TSH_EUR_genes.txt').read().splitlines()

    mp = ManhattanPlot(file_path='PMBB_AFR_EUR.INV_NORM_TSH_merged.meta',
                       title='PMBB Multi-Ancestry Meta-Analysis for TSH',
                       test_rows=None)
    mp.load_data()
    mp.clean_data(col_map={'CHR': '#CHROM',
                           'BP': 'POS',
                           'P(R)': 'P',
                           'SNP': 'ID'})
    mp.add_annotations(annotDF, extra_cols=['RSID'])
    mp.get_thinned_data()

    # Vertical With Table
    mp.update_plotting_parameters(sug=1E-5, annot_thresh=1E-5, sig=5E-8,
                                  ld_block=4E5, merge_genes=True,
                                  invert=False)

    mp.full_plot(rep_genes=known_genes,
                 extra_cols={'RSID': 'RSID', 'A1': 'Allele'},
                 rep_boost=True,
                 keep_chr_pos=False,
                 save_res=150, save='PMBB_TSH_Meta-Analysis_GWAS_plot_vertical.png')

    # Horizontal Without Table
    mp.update_plotting_parameters(sug=1E-5, annot_thresh=1E-5, sig=5E-8,
                                  ld_block=4E5, merge_genes=True,
                                  invert=False, vertical=False)

    mp.full_plot(rep_genes=known_genes,
                 rep_boost=True,
                 with_table=False,
                 save_res=150, save='PMBB_TSH_Meta-Analysis_GWAS_plot_horizontal.png')
