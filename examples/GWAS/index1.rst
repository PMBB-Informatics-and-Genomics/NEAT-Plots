.. NEAT-Plots documentation master file, created by
   sphinx-quickstart on Thu Nov  9 20:12:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NEAT-Plots's documentation!
======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

To create a Manhattan Plot:

Import the pandas, os, manhattan_plot, ManhattanPlot from manhattan_plot, and matplotlib.pyplot.


For example:

.. doctest::
	import pandas as pd
	import os
	import manhattan_plot
	from manhattan_plot import ManhattanPlot
	import matplotlib.pyplot as plt

Read your file using "annotDF" (pandas) and rename your columns in the DataFrame by using "annotDF.rename". In the following example the column 'Gene' was renamed to 'ID'.

.. doctest::
	annotDF = pd.read_csv('yourfile.csv')
	annotDF = annotDF.rename(columns={'Gene': 'ID'})
	print annotDF

Identify known genes (or what you are searching for) and then create a list of them. By doing so, you can highlight these genes/objects in your plots,

For example:

.. doctest::
	known_genes = ['enter', 'known', 'genes']
	print known_genes


Load data from your specified file. Then clean the data by mapping the column names in the col_map parameter to the corresponding column names in the data. Add annotations from the previous steps and thin data.


For example:

.. doctest::
mp = ManhattanPlot(file_path='Data/filename.tsv',
					   title='Your Title',/
					   test_rows=None)

	mp.load_data()
	mp.clean_data(col_map={'hm_chrom': '#CHROM',
						   'hm_pos': 'POS',
						   'p_value': 'P',
						   'hm_variant_id': 'ID'})
	mp.add_annotations(annotDF, extra_cols=['RSID'])
	mp.get_thinned_data()

**Your data is now ready for plot generation.**


*For a Vertical Manhattan Plot*


State plotting parameters and values for those parameters. Create a full plot with columns for known genes and display plot.
	

For example:
.. doctest::
	mp.update_plotting_parameters(sug=1E-5, annot_thresh=1E-5, sig=5E-8,
								  ld_block=1E6, merge_genes=True,
								  invert=False)

	mp.full_plot(rep_genes=known_genes, extra_cols={'RSID': 'RSID', 'effect_allele': 'Allele'},
				 rep_boost=True, keep_chr_pos=False)
	plt.show()


*For a Horizontal Manhattan Plot*


Update your plotting parameters to fit a horizontal plot. Create a full plot with columns for known genes and display plot.

For example:
.. doctest::
	mp.update_plotting_parameters(sug=1E-5, annot_thresh=1E-5, sig=5E-8,
								  ld_block=1E6, merge_genes=True,
								  invert=False, vertical=False, max_log_p=30)

	mp.full_plot(rep_genes=known_genes, rep_boost=True, with_table_grid=False, with_table_bg=False)
	plt.show()


NEAT-Plots can also generate other plots, such as QQ Plots and Signal Plots. Reference the following examples to do so.


*Generating a Signal Plot*

.. doctest::
mp.update_plotting_parameters(sug=1E-5, annot_thresh=1E-5, sig=5E-8,
                              ld_block=1E6, merge_genes=True,
                              invert=False)

mp.signal_plot(rep_genes=known_genes, extra_cols={'RSID': 'RSID', 'effect_allele': 'Allele'},
             rep_boost=True, keep_chr_pos=False)
plt.show()


*Generating a QQ Plot*

.. doctest::
mp.qq_plot()
plt.show()


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

