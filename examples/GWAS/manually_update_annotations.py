import pandas as pd

annotDF = pd.read_csv('meta_suggestive_biofilter_genes_rsids.csv')

# Manual re-annotation of genes near known genes
change = {'LOC107986195': 'NR3C2',

          'LOC107986598': 'VEGFA',
          'LINC02537': 'VEGFA',
          'LOC105375068': 'VEGFA',

          'LOC105371356': 'MAFTRR',
          'LOC101928278': 'IGFBP5',

          'FGF7': 'FAM227B',
          'GALK2': 'FAM227B',
          'COPS2': 'FAM227B',
          'SECISBP2L': 'FAM227B',
          'DTWD1': 'FAM227B',

          'LOC112268051': 'PTSCS2',
          'LOC110121153': 'PTCSC2',
          'LOC110121154': 'PTSCS2',
          'LOC114827821': 'PTCSC2',
          'FOXE1': 'PTCSC2',
          'TRMO': 'PTCSC2',
          'NANS': 'PTCSC2',
          'TRIM14': 'PTCSC2',

          'LINC01747': 'LNCNEF',
          'LINC01384': 'LNCNEF',

          'LOC112637023': 'ABO',
          'LOC112679202': 'ABO',
          'LCN1P2': 'ABO',

          'MICOS10-NBL1': 'CAPZB',
          'LOC105376819': 'CAPZB',
          'LOC105378614': 'CAPZB'}

annotDF['Gene'] = annotDF['Gene'].replace(change)

annotDF.to_csv('meta_suggestive_biofilter_genes_rsids_UPDATED.csv', index=False)
