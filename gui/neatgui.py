import sys

from PySide6.QtWidgets import (QApplication, QWidget, QMainWindow, QFileDialog, QSizePolicy, 
   QHBoxLayout, QGroupBox,QFormLayout, QLabel, QLineEdit, QComboBox, QVBoxLayout, 
   QPushButton, QTableWidget, QListWidget, QCheckBox, QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import pandas as pd
import manhattan_plot 
from section import Section
import matplotlib.pyplot as plt

from main_window import Ui_MainWindow

class Window(QMainWindow, Ui_MainWindow): 
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.connectSignalsSlots()
        self.addSections()
        self.delims={'whitespace': '\s+', 
            ',' : ',',
            'tab' : '\t',
            '|' : '|'
            }
        self.known_genes = []
        
        self.layoutTabPlot.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.layoutTabProcess.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.layoutTabLoad.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.layoutTabAnnotations.setAlignment(Qt.AlignHCenter | Qt.AlignTop)        
        self.setActiveTab(0)
        self.annotDF = pd.DataFrame()
        self.mp = None

    def connectSignalsSlots(self):
        self.btnDataFn.clicked.connect(self.selectFile)
        self.btnDataFnLoad.clicked.connect(self.loadDataFile)
        self.btnAnnotationsFn.clicked.connect(self.selectAnnotFile)
        self.btnLoadAnnotFile.clicked.connect(self.loadAnnotFile)
        #self.btnAddAnnot.clicked.connect(self.addAnnotation)
        self.btnPlot.clicked.connect(self.plotImage)
        self.btnSave.clicked.connect(self.saveImage)
        self.btnKnownGenesFn.clicked.connect(self.selectKnownGenesFn)
        self.btnLoadKnownGenesFile.clicked.connect(self.loadKnownGenesFile)
        self.btnPrev1.clicked.connect(self.moveTab1)
        self.btnNext1.clicked.connect(self.moveTab2)
        self.btnPrev2.clicked.connect(self.moveTab2)
        self.btnNext2.clicked.connect(self.moveTab3)
        self.btnPrev3.clicked.connect(self.moveTab3)
        self.btnNext3.clicked.connect(self.moveTab4)
        
        
#         self.action_Exit.triggered.connect(self.close)
#         self.action_Find_Replace.triggered.connect(self.findAndReplace)
#         self.action_About.triggered.connect(self.about)
# 
# 
#     def about(self):
#         QMessageBox.about(
#             self,
#             "About Sample Editor",
#             "<p>A sample text editor app built with:</p>"
#             "<p>- PyQt</p>"
#             "<p>- Qt Designer</p>"
#             "<p>- Python</p>",
#         )

    def moveTab1(self):
        self.tabWidget.setCurrentIndex(0)
        self.setActiveTab(0)

    def moveTab2(self):
        self.tabWidget.setCurrentIndex(1)
        self.setActiveTab(1)
    
    def moveTab3(self):
        self.tabWidget.setCurrentIndex(2)
        self.setActiveTab(2)
    
    def moveTab4(self):
        self.tabWidget.setCurrentIndex(3)
        self.setActiveTab(3)
        self.addAnnotation()
        
    def setActiveTab(self, tabIndex):
        num = self.tabWidget.count()
        for i in range(num):
           self.tabWidget.setTabEnabled(i,False) if i != tabIndex else self.tabWidget.setTabEnabled(i,True)

    # add collapsible sections
    def addSections(self):
        self.sect = Section(title="Color Options")
        layout = QFormLayout()
        self.comboBxTWASColor = QComboBox()
        self.comboBxTWASDirection = QComboBox()
        self.comboBxTWASSignal = QComboBox()
        layout.addRow(QLabel("TWAS Color Column"),self.comboBxTWASColor)
        layout.addRow(QLabel("TWAS Direction Column"),self.comboBxTWASDirection)
        layout.addRow(QLabel("Signal Color Column"),self.comboBxTWASSignal)
        self.sect.setContentLayout(layout)
        
        self.sectTWAS = Section(title="Table Options")
        layout = QFormLayout()
        self.tableAddColumns = QTableWidget(0,3)
        self.tableAddColumns.setMinimumWidth(200)
        self.tableAddColumns.setHorizontalHeaderLabels(['Include','Name   ','Rename'])
        horizontalHeader =  self.tableAddColumns.horizontalHeader()
        horizontalHeader.setSectionResizeMode(0,QHeaderView.ResizeToContents)
        horizontalHeader.setSectionResizeMode(1,QHeaderView.ResizeToContents)
        horizontalHeader.setSectionResizeMode(2,QHeaderView.Stretch)
        
        layout.addRow(QLabel("Extra Columns"), self.tableAddColumns)
        self.listNumericCols = QListWidget()
        self.listNumericCols.setMaximumWidth(150)
        layout.addRow(QLabel("Select Numeric"), self.listNumericCols)
        self.chkBoxKeepGenomic = QCheckBox()
        self.chkBoxKeepGenomic.setCheckState(Qt.Unchecked)
        layout.addRow(QLabel("Keep Genomic Pos"),self.chkBoxKeepGenomic)
        self.sectTWAS.setContentLayout(layout)
        
        self.layoutTabPlot.insertWidget(4,self.sect)
        self.layoutTabPlot.insertWidget(5,self.sectTWAS)

    def selectFile(self):
        fileName = QFileDialog.getOpenFileName(self, "Open File", "./")
        self.lineDataFn.setText(fileName[0])
    
    def loadDataFile(self):
        nrows = self.spinTestRows.value() if self.spinTestRows.value() > 0 else None
        
        delimiter=self.getDelim()        
        self.mp = manhattan_plot.ManhattanPlot(file_path=self.lineDataFn.text(),
                       title=self.lineDataTitle.text(),
                       test_rows=nrows)
        self.mp.load_data(delim=delimiter)
        self.setSelectionColumns(self.mp.df.columns)
    
    # select Annotation file from dialog
    def selectAnnotFile(self):
        fileName = QFileDialog.getOpenFileName(self, "Open File", "./")
        self.lineAnnotationsFn.setText(fileName[0])
    
    def loadAnnotFile(self):
        self.annotDF = pd.read_csv(self.lineAnnotationsFn.text())
        self.setAnnotColumns(self.annotDF.columns)
        
        
    def selectKnownGenesFn(self):
        fileName = QFileDialog.getOpenFileName(self, "Open File", "./")
        self.lineKnownGenesFn.setText(fileName[0])
        
    def loadKnownGenesFile(self):
        self.known_genes = open(self.lineKnownGenesFn.text()).read().splitlines()
    
    def saveImage(self):
        saveFn = QFileDialog.getSaveFileName(self, "Save plot image", "neat.png", "PNG files (*.png)")
        self.canvasPlot.fig.savefig(saveFn[0], dpi=150)
        
    # return delmiiter for file set in combobox
    def getDelim(self):
        return self.delims.setdefault(self.bxDelimiter.currentText(), self.bxDelimiter.currentText())
    
    # set columns for comboboxes for user selection
    def setSelectionColumns(self, cols):
        self.bxChromCol.clear()
        self.bxChromCol.addItems(cols)
        self.bxPosCol.clear()
        self.bxPosCol.addItems(cols)
        self.bxIDCol.clear()
        self.bxIDCol.addItems(cols)
        self.bxPvalueCol.clear()
        self.bxPvalueCol.addItems(cols)

    # set columns for comboboxes for user selection for annotation file
    def setAnnotColumns(self, cols):
        self.bxChromAnn.clear()
        self.bxChromAnn.addItems(cols)
        self.bxPosAnn.clear()
        self.bxPosAnn.addItems(cols)
        self.bxIDAnn.clear()
        self.bxIDAnn.addItems(cols)
        self.listOtherAnn.clear()
        self.listOtherAnn.addItems(cols)
        
    def addAnnotation(self):
        self.mp.clean_data(col_map={ self.bxChromCol.currentText() : '#CHROM',
                           self.bxPosCol.currentText() : 'POS' ,
                           self.bxPvalueCol.currentText() : 'P',
                           self.bxIDCol.currentText() : 'ID' })
        if not self.annotDF.empty: 
            self.annotDF = self.annotDF.rename(columns={self.bxIDAnn.currentText(): 'ID',
              self.bxChromAnn.currentText(): '#CHROM', self.bxPosAnn.currentText(): 'POS'})
            addCols = [item.text() for item in self.listOtherAnn.selectedItems()]
            self.mp.add_annotations(self.annotDF, extra_cols=addCols)
        self.mp.get_thinned_data()
        
        self.fillExtraColsTable()
        self.fillNumericList()
        self.fillTWASboxes()

    # draw image on canvas
    def plotImage(self):
        # close any open plots to prevent memory leak in matplotlib
        plt.close('all') 
        suggest = float(self.lineSuggestThresh.text())
        significant = float(self.lineSigThresh.text())
        ldblock = float(self.lineLDLBlockWidth.text())
        merge = self.convertTF(self.bxMergeGenes.currentText())
        maxLogP = float(self.lineMaxLogP.text()) if self.lineMaxLogP.text() else ''

        twas_color_col=self.comboBxTWASColor.currentText()
        twas_updown_col=self.comboBxTWASDirection.currentText()
        signal_color_col=self.comboBxTWASSignal.currentText()
        
        self.mp.update_plotting_parameters(sug=suggest, annot_thresh=1E-5, sig=significant,
                                      ld_block=ldblock, merge_genes=merge,
                                      invert=False, twas_color_col=twas_color_col,
                                      signal_color_col=signal_color_col, twas_updown_col=twas_updown_col)
                                      
        # below taken from manhattan_plot full_plot method
        with_table=True
        rep_genes=self.known_genes
        rep_boost=True
        
        extra_cols=self.selectedExtraCols()
        addCols = [item.text() for item in self.listOtherAnn.selectedItems()]
        [extra_cols.update({x:x}) for x in addCols]
        
        number_cols = [item.text() for item in self.listNumericCols.selectedItems()]
        if self.chkBoxKeepGenomic.checkState() == Qt.Checked:
            keep_chr_pos = True
        else:
            keep_chr_pos = False
        with_title=True
        
        self.mp.full_plot(extra_cols=extra_cols, number_cols=number_cols, rep_genes=rep_genes, keep_chr_pos=keep_chr_pos,
          with_table=with_table, rep_boost=rep_boost, with_title=with_title)
        
        self.horizontalLayoutCentral.itemAt(1).widget().deleteLater()
        self.canvasPlot = PlotCanvas(self,self.mp.fig)
        self.horizontalLayoutCentral.addWidget(self.canvasPlot)
        self.canvasPlot.plot()

    def convertTF(self, text):
        tf = True if text == 'True' else False
        return tf
    
    def getExtraCols(self):
        columns = set(list(self.mp.df.columns))
        if not self.annotDF.empty:
            columns.update(list(self.annotDF.columns))
        # get all selected columns
        # remove those from the columns set
        # add those columns to the table
        selectedCols = set(item.text() for item in self.listOtherAnn.selectedItems())
        selectedCols.add(self.bxChromCol.currentText())
        selectedCols.add(self.bxPosCol.currentText())
        selectedCols.add(self.bxIDCol.currentText())
        selectedCols.add(self.bxPvalueCol.currentText())
        selectedCols.add(self.bxChromAnn.currentText())
        selectedCols.add(self.bxPosAnn.currentText())
        selectedCols.add(self.bxIDAnn.currentText())
        return list(columns - selectedCols)    

    # fill extra columns table with any columns not already selected
    def fillExtraColsTable(self):
        self.tableAddColumns.setRowCount(0)
        extracols = self.getExtraCols()
        extracols.sort()
        self.tableAddColumns.clearContents()
        for column in extracols:
            self.addRowExtraColTable(column)
        
    def addRowExtraColTable(self, colname):
        self.tableAddColumns.insertRow(self.tableAddColumns.rowCount())
        index = self.tableAddColumns.rowCount()-1
        chkbox = QTableWidgetItem()
        chkbox.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        chkbox.setCheckState(Qt.Unchecked)
        self.tableAddColumns.setItem(index,0,chkbox)
        self.tableAddColumns.setItem(index,1,QTableWidgetItem(colname))
        
    # list extra columns for specifying which are numeric
    def fillNumericList(self):
        self.listNumericCols.clear()
        extracols = self.getExtraCols()
        extracols.sort()
        self.listNumericCols.addItems(extracols)
    
    # return dictionary of extra columns with altered names for plotting
    def selectedExtraCols(self):
        extra = {}
        for r in range(0,self.tableAddColumns.rowCount()):
            if self.tableAddColumns.item(r, 0).checkState() == Qt.Checked:
                if self.tableAddColumns.item(r, 2) is not None and self.tableAddColumns.item(r, 2).text():
                   extra[self.tableAddColumns.item(r,1).text()] = self.tableAddColumns.item(r,2).text()
                else:
                   extra[self.tableAddColumns.item(r,1).text()] = self.tableAddColumns.item(r,1).text()
        return extra

    def fillTWASboxes(self):
       cols = [''] + self.getExtraCols()
       self.comboBxTWASColor.clear()
       self.comboBxTWASColor.addItems(cols)
       self.comboBxTWASDirection.clear()
       self.comboBxTWASDirection.addItems(cols)
       self.comboBxTWASSignal.clear()
       self.comboBxTWASSignal.addItems(cols)


# For display of plots    
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, fig=None):
        self.fig = fig
        FigureCanvas.__init__(self,fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self,
            QSizePolicy.Expanding,
            QSizePolicy.Expanding)
    
    def plot(self):
        self.draw()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())
