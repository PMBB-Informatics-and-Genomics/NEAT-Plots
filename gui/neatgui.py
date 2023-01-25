import sys
import os.path
import traceback

from PySide6.QtWidgets import (QApplication, QWidget, QMainWindow, QFileDialog, QSizePolicy, 
   QHBoxLayout, QGroupBox,QFormLayout, QLabel, QLineEdit, QComboBox, QVBoxLayout, 
   QPushButton, QTableWidget, QListWidget, QCheckBox, QTableWidgetItem, QHeaderView, 
   QDialog, QApplication, QMessageBox)
from PySide6.QtCore import Qt, QObject, QThread, Signal
from PySide6.QtGui import QMovie, QDoubleValidator

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

syspath = os.path.join('..', 'manhattan-plot')
sys.path.insert(1,syspath)

import pandas as pd
import re
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
            ', (comma)' : ',',
            'tab' : '\t',
            '| (pipe)' : '|'
            }
        self.known_genes = []
        self.canvasPlot = None
        self.prevDataFn = ""
        
        self.layoutTabPlot.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.layoutTabProcess.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.layoutTabLoad.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.layoutTabAnnotations.setAlignment(Qt.AlignHCenter | Qt.AlignTop) 
        self.verticalLayoutScrollAreaPlot.setAlignment(Qt.AlignHCenter | Qt.AlignTop)    
        self.setActiveTab(0)
        self.annotDF = pd.DataFrame()
        self.mp = None
        self.chkBoxWithTable.setEnabled(False)
        self.progressBarBlue.setVisible(False)
        self.loadedDataFile = False
        self.dataFileLoadingRequired = False
        self.annotChangeMade = False
        self.lastDir = "."
        self.errorMessage=""
        self.dataCopy = FrameHolder()
        self.firstTimeStep3 = True
        
        # numeric validators
        self.threshMin = 0.0
        self.threshMax = 1.0
        self.maxDigits = 10
        self.upperLimits = 1e50
        self.lineSuggestThresh.setValidator(QDoubleValidator(self.threshMin ,self.threshMax,self.maxDigits ))
        self.lineSigThresh.setValidator(QDoubleValidator(self.threshMin ,self.threshMax,self.maxDigits ))
        self.lineMaxLogP.setValidator(QDoubleValidator(self.threshMin ,self.upperLimits,self.maxDigits ))
        self.lineLDLBlockWidth.setValidator(QDoubleValidator(self.threshMin ,self.upperLimits,self.maxDigits ))

        self.centerCombo(self.bxDelimiter)
        self.centerCombo(self.bxAnnotDelimiter)
        
        
        
    def connectSignalsSlots(self):
        self.lineDataFn.textChanged.connect(self.dataFileChange)
        self.btnDataFn.clicked.connect(self.selectFile)
        self.btnDataFnLoad.clicked.connect(self.loadHeaders)
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
        self.btnNext2.clicked.connect(self.moveTab3Forward)
        self.btnPrev3.clicked.connect(self.moveTab3)
        self.btnNext3.clicked.connect(self.moveTab4)
        self.chkBoxVertical.stateChanged.connect(self.restrictTableOption)
        self.actionExit.triggered.connect(self.close)
        self.actionInformation.triggered.connect(self.about)
        self.bxChromAnn.activated.connect(self.annotChanged)
        self.bxChromAnn.currentIndexChanged.connect(self.updateAnnotList)
        self.bxPosAnn.activated.connect(self.annotChanged)
        self.bxPosAnn.currentIndexChanged.connect(self.updateAnnotList)
        self.bxIDAnn.activated.connect(self.annotChanged)
        self.bxIDAnn.currentIndexChanged.connect(self.updateAnnotList)
        self.listOtherAnn.itemSelectionChanged.connect(self.annotChanged)
        
 
    def centerCombo(self, box):
        line_edit = box.lineEdit()
        line_edit.setAlignment(Qt.AlignCenter)
        line_edit.setReadOnly(True)
 
    def about(self):
        QMessageBox.about(
            self,
            "NEAT",
            "<p>A graphical user iterface for Lindsay Guare's Python plotting ibrary</p>",
        )

    def moveTab1(self):
        self.tabWidget.setCurrentIndex(0)
        self.setActiveTab(0)

    def moveTab2(self):
        if self.loadedDataFile != True:
            QMessageBox.warning(self,"Error", "<p>Please load data input file before proceeding</p>")
        else:
            self.tabWidget.setCurrentIndex(1)
            self.setActiveTab(1)
    
    def moveTab3(self):
        self.tabWidget.setCurrentIndex(2)
        self.setActiveTab(2)
    
    def moveTab3Forward(self):
        cols = set([self.bxChromCol.currentText(), self.bxPosCol.currentText(), self.bxIDCol.currentText(),
            self.bxPvalueCol.currentText()])
        if len(cols) != 4:   
            QMessageBox.warning(self,"Error", "<p>Please set column names in data input file</p>")
        elif self.dataFileLoadingRequired == True:
            self.loadDataFile()
            
        if self.errorMessage != "":
            QMessageBox.warning(self,"Error", f"<p>{self.errorMessage}</p>")
            self.errorMessage=""
        else:
            self.tabWidget.setCurrentIndex(2)
            self.setActiveTab(2)
            self.dataFileLoadingRequired = False
        
    
    def moveTab4(self):
        if not self.annotDF.empty: 
            cols = set([self.bxChromAnn.currentText(), self.bxPosAnn.currentText(), self.bxIDAnn.currentText()])
            if len(cols) != 3:   
                QMessageBox.warning(self,"Error", "<p>Please set column names for annotation file</p>")
                return
        self.addAnnotation()
        self.tabWidget.setCurrentIndex(3)
        self.setActiveTab(3)
        
    def setActiveTab(self, tabIndex):
        num = self.tabWidget.count()
        for i in range(num):
           self.tabWidget.setTabEnabled(i,False) if i != tabIndex else self.tabWidget.setTabEnabled(i,True)

    def dataFileChange(self):
        if self.lineDataFn.text() != self.prevDataFn:
            self.prevDataFn = self.lineDataFn.text()
            self.loadedDataFile = False
            
    def annotChanged(self):
        self.annotChangeMade = True

    # hide items in annotation list for selecting other columns
    # so that the renamed columns will not be used in the final plot causing
    # an error  
    def updateAnnotList(self):
        selected = set([self.bxChromAnn.currentText(), self.bxPosAnn.currentText(), 
            self.bxIDAnn.currentText()])
        
        for idx in range(self.listOtherAnn.count()):
            if self.listOtherAnn.item(idx).text() in selected:
                self.listOtherAnn.item(idx).setHidden(True)
            else:
                self.listOtherAnn.item(idx).setHidden(False)
                self.listOtherAnn.item(idx).setSelected(False)


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
        
        self.verticalLayoutScrollAreaPlot.insertWidget(5,self.sect)
        self.verticalLayoutScrollAreaPlot.insertWidget(6 ,self.sectTWAS)
        

    def selectFile(self):
        fileName = QFileDialog.getOpenFileName(self, "Open File", self.lastDir)
        self.lastDir = os.path.dirname(os.path.abspath(fileName[0]))
        self.lineDataFn.setText(fileName[0])
    
    def loadHeaders(self):
        if self.lineDataFn.text() != "":
            delimiter=self.getDelim(self.bxDelimiter.currentText())
            try: 
                with open(self.lineDataFn.text()) as f:
                    self.headers = re.split(delimiter,f.readline())
            except Exception as ex:
                traceback_str = ''.join(traceback.format_tb(ex.__traceback__))
                QMessageBox.warning(self,"Error", "<p>Loading file failed: " + str(ex) + "</p><p>Details: " +
                    traceback_str + "</p>")
            else:
                self.setSelectionColumns(self.headers)
                self.loadedDataFile = True
                self.dataFileLoadingRequired = True
    
    def loadDataFile(self):
        nrows = self.spinTestRows.value() if self.spinTestRows.value() > 0 else None
        
        delimiter=self.getDelim(self.bxDelimiter.currentText())        
        self.mp = manhattan_plot.ManhattanPlot(file_path=self.lineDataFn.text(),
                       title=self.lineDataTitle.text(),
                       test_rows=nrows)
                       
        #use thread here
        self.thread = QThread()
        self.worker = FileWorker()
        self.worker.mp = self.mp
        self.worker.copyHolder = self.dataCopy
        self.worker.delim = delimiter
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.loadDataFile)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.exceptCaught.connect(self.thread.quit)
        self.worker.exceptCaught.connect(self.workerError)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.dlg = ProcessingDialog(self)
        self.thread.finished.connect(self.dlg.accept)
        self.thread.finished.connect(self.dlg.deleteLater)
        self.thread.start()
        self.dlg.exec()
                
        self.firstTimeStep3 = True
        
        
    
    def workerError(self):
        self.errorMessage=self.worker.errorStr
        self.worker.deleteLater
            
    # select Annotation file from dialog
    def selectAnnotFile(self):
        fileName = QFileDialog.getOpenFileName(self, "Open File", self.lastDir)
        self.lastDir = os.path.dirname(os.path.abspath(fileName[0]))
        self.lineAnnotationsFn.setText(fileName[0])
    
    def loadAnnotFile(self):
        if self.lineAnnotationsFn.text() != "":
            try:
                self.annotDF = pd.read_csv(self.lineAnnotationsFn.text(), delimiter=self.getDelim(self.bxAnnotDelimiter.currentText()))
                self.setAnnotColumns(self.annotDF.columns)
                self.annotChanged()
            except Exception as ex:
                traceback_str = ''.join(traceback.format_tb(ex.__traceback__))
                QMessageBox.warning(self,"Error", "Loading file failed: " + str(ex) + "</p><p>Details: " +
                    traceback_str + "</p>")
        else:
            self.annotDF = pd.DataFrame()
        
        
    def selectKnownGenesFn(self):
        fileName = QFileDialog.getOpenFileName(self, "Open File", self.lastDir)
        self.lastDir = os.path.dirname(os.path.abspath(fileName[0]))
        self.lineKnownGenesFn.setText(fileName[0])
        
    def loadKnownGenesFile(self):
        if self.lineKnownGenesFn.text() != "":
            try: 
                self.known_genes = open(self.lineKnownGenesFn.text()).read().splitlines()
            except Exception as ex:
                traceback_str = ''.join(traceback.format_tb(ex.__traceback__))
                QMessageBox.warning(self,"Error", "Loading file failed: " + str(ex) + "</p><p>Details: " +
                    traceback_str + "</p>")
        else:
            self.known_genes = []
    
    def saveImage(self):
        defaultName = os.path.join(self.lastDir, "neat.png")
        saveFn = QFileDialog.getSaveFileName(self, "Save plot image", defaultName, "PNG files (*.png)")
        if saveFn[0] != "":
            try:
                self.lastDir = os.path.dirname(os.path.abspath(saveFn[0]))
                self.canvasPlot.fig.savefig(saveFn[0], dpi=150)
            except Exception as ex:
                traceback_str = ''.join(traceback.format_tb(ex.__traceback__))
                QMessageBox.warning(self,"Error", "Saving file failed: " + str(ex) + "</p><p>Details: " +
                    traceback_str + "</p>")
        
    # return delmiiter for file set in combobox
    def getDelim(self, text):
        #return self.delims.setdefault(self.bxDelimiter.currentText(), self.bxDelimiter.currentText())
        return self.delims.setdefault(text, text)
    
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
        addCols = []
        columnMap = { self.bxChromCol.currentText() : '#CHROM',
                           self.bxPosCol.currentText() : 'POS' ,
#                           self.bxPvalueCol.currentText() : 'P',
                           self.bxIDCol.currentText() : 'ID' }
        # only set P value when it is not a logp column
        logpcol = None
        if self.chkBxLog10Pval.isChecked():
            logpcol = self.bxPvalueCol.currentText()
        else:
            columnMap[self.bxPvalueCol.currentText()]='P'

        if not self.annotDF.empty or self.annotChangeMade: 
            self.annotDF = self.annotDF.rename(columns={self.bxIDAnn.currentText(): 'ID',
              self.bxChromAnn.currentText(): '#CHROM', self.bxPosAnn.currentText(): 'POS'})
            addCols = [item.text() for item in self.listOtherAnn.selectedItems()]
            
        # if it is the first time through this
        # or the annotation changed then we add annotation and thin data again
        if self.firstTimeStep3 or self.annotChangeMade:
            self.mp.df = self.dataCopy.df.copy()
            self.thread = QThread()
            self.worker = FileWorker()
            self.worker.mp = self.mp
            #self.worker.delim = delimiter
            self.worker.annotDF = self.annotDF
            self.worker.columnMap = columnMap
            self.worker.addCols = addCols
            self.worker.copyHolder = self.dataCopy
            self.worker.logp = logpcol
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.thinData)
            self.worker.finished.connect(self.thread.quit)
            self.worker.exceptCaught.connect(self.thread.quit)
            self.worker.exceptCaught.connect(self.workerError)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
        
            self.dlg = ProcessingDialog(self, message="Processing Data...")
            self.thread.finished.connect(self.dlg.accept)
            self.thread.finished.connect(self.dlg.deleteLater)
            self.thread.finished.connect(self.fillPlotOptions)
            self.thread.start()
            self.dlg.exec()
            
            self.annotChangeMade = False
            self.firstTimeStep3 = False
        
        
    def fillPlotOptions(self):
        self.fillExtraColsTable()
        self.fillNumericList()
        self.fillTWASboxes()

    def updateProgress(self, progValue=0):
        self.progressBarBlue.setValue(progValue)
        QApplication.processEvents() 
    
    def withinRange(self, minimum, maximum, value):
        if value < minimum or value > maximum:
            return False
        else:
            return True
    
    def validatePlotInputs(self):
        eMessage=""
        if self.lineSuggestThresh.text() != "" and not self.withinRange(self.threshMin, self.threshMax, float(self.lineSuggestThresh.text())):
            eMessage += "<p>Suggestive Threshold is outside of expected range (0.0-1.0)</p>"
        if self.lineSigThresh.text() != "" and not self.withinRange(self.threshMin, self.threshMax, float(self.lineSigThresh.text())):
            eMessage += "<p>Significant Threshold is outside of expected range (0.0-1.0)</p>"
        if self.lineMaxLogP.text() != "" and not self.withinRange(self.threshMin, self.upperLimits, float(self.lineMaxLogP.text())):
            eMessage += "<p>Max Log P is outside of expected range (0.0-1e50)</p>"
        if self.lineLDLBlockWidth.text() != "" and not self.withinRange(self.threshMin, self.upperLimits, float(self.lineLDLBlockWidth.text())):
            eMessage += "<p>LD Block Width is outside of expected range (0.0-1e50)</p>"
        
        if eMessage != "":
            QMessageBox.warning(self,"Error", eMessage)
            return False
        else:
            return True
        
        
    # draw image on canvas
    def plotImage(self):
        if not self.validatePlotInputs():
            return
        if self.canvasPlot is not None:
            self.canvasPlot.setVisible(False)
            self.canvasPlot.deleteLater()
        progValue=0
        self.progressBarBlue.setVisible(True)
        self.updateProgress(0)
 
        # close any open plots to prevent memory leak in matplotlib
        plt.close('all') 
        suggest = float(self.lineSuggestThresh.text())
        significant = float(self.lineSigThresh.text())
        ldblock = float(self.lineLDLBlockWidth.text())
        merge = self.convertTF(self.bxMergeGenes.currentText())
        rep_boost = self.convertTF(self.bxBoostKnown.currentText())
        maxLogP = float(self.lineMaxLogP.text()) if self.lineMaxLogP.text() else ''
        inverted = True if self.chkBoxInvert.checkState() == Qt.Checked else False
        include_title = True if self.chkBoxWithTitle.checkState() == Qt.Checked else False
        include_table = True if self.chkBoxWithTable.checkState() == Qt.Checked else False
        signals_only = True if self.chkBoxSignalsOnly.checkState() == Qt.Checked else False
        vertOrientation = True if self.chkBoxVertical.checkState() == Qt.Checked else False
        
        twas_color_col=self.comboBxTWASColor.currentText()
        twas_updown_col=self.comboBxTWASDirection.currentText()
        signal_color_col=self.comboBxTWASSignal.currentText()
        
        self.mp.update_plotting_parameters(sug=suggest, annot_thresh=1E-5, sig=significant,
                                      ld_block=ldblock, merge_genes=merge,
                                      invert=inverted, twas_color_col=twas_color_col,
                                      signal_color_col=signal_color_col, twas_updown_col=twas_updown_col,
                                      title=self.lineDataTitle.text(), vertical=vertOrientation)
        self.updateProgress(25)
        rep_genes=self.known_genes
        
        extra_cols=self.selectedExtraCols()
        addCols = [item.text() for item in self.listOtherAnn.selectedItems()]
        [extra_cols.update({x:x}) for x in addCols]
        
        number_cols = [item.text() for item in self.listNumericCols.selectedItems()]
        if self.chkBoxKeepGenomic.checkState() == Qt.Checked:
            keep_chr_pos = True
        else:
            keep_chr_pos = False
        
        try:
            if self.chkBoxSignalsOnly.checkState() == Qt.Unchecked:
                self.mp.full_plot(extra_cols=extra_cols, number_cols=number_cols, rep_genes=rep_genes, keep_chr_pos=keep_chr_pos,
                  with_table=include_table, rep_boost=rep_boost, with_title=include_title)
            else:
                self.mp.signal_plot(extra_cols=extra_cols, number_cols=number_cols, rep_genes=rep_genes, keep_chr_pos=keep_chr_pos,
                  with_table=include_table, rep_boost=rep_boost, with_title=include_title)
            self.updateProgress(75)
            
            self.mp.fig.dpi=70
            self.canvasPlot = PlotCanvas(self,self.mp.fig)
            self.horizontalLayoutFrame.insertWidget(1,self.canvasPlot)
            self.canvasPlot.plot()
        except Exception as ex:
            traceback_str = ''.join(traceback.format_tb(ex.__traceback__))
            QMessageBox.warning(self,"Error", "<p>Unable to create plot: "  + str(ex) + "</p><p>Details: " +
                    traceback_str + "</p>")
        
        self.progressBarBlue.setVisible(False)

    def convertTF(self, text):
        tf = True if text == 'True' else False
        return tf
    
    def getExtraCols(self):
        columns = set(list(self.mp.df.columns))
        if not self.annotDF.empty:
            columns.update(list(self.annotDF.columns))
        
        columns.discard('#CHROM')
        columns.discard('POS')
        
        # get all selected columns
        # remove those from the columns set
        # add those columns to the table
        #selectedCols = set(item.text() for item in self.listOtherAnn.selectedItems())
        #selectedCols.add(self.bxChromCol.currentText())
        #selectedCols.add(self.bxPosCol.currentText())
        #selectedCols.add(self.bxIDCol.currentText())
        #selectedCols.add(self.bxPvalueCol.currentText())
        #selectedCols.add(self.bxChromAnn.currentText())
        #selectedCols.add(self.bxPosAnn.currentText())
        #selectedCols.add(self.bxIDAnn.currentText())
        #retcols = list(columns - selectedCols)
        retcols = list(columns)
        retcols.insert(0,self.bxPosCol.currentText())
        retcols.insert(0,self.bxChromCol.currentText())
        return retcols
        
        

    # fill extra columns table with any columns not already selected
    def fillExtraColsTable(self):
        self.tableAddColumns.setRowCount(0)
        extracols = self.getExtraCols()
        #extracols.sort()
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
       
    # when vertical option set, the with table checkbox must be set also
    def restrictTableOption(self):
        if self.chkBoxVertical.checkState() == Qt.Checked:
            self.chkBoxWithTable.setCheckState(Qt.Checked)
            self.chkBoxWithTable.setEnabled(False)
        else:
            self.chkBoxWithTable.setEnabled(True)
    

class ProcessingDialog(QDialog):
    def __init__(self, parent=None, message="Loading File..."):
        super().__init__(parent)

        loadingGif = get_path('resources/loading.gif')
        self.setWindowTitle("Please wait")
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignHCenter)
        hlayout = QHBoxLayout()
        hlayout.setAlignment(Qt.AlignHCenter)
        message = QLabel(message)
        hlayout.addWidget(message)
        #self.layout.addWidget(message)
        self.layout.addLayout(hlayout)
        self.movie = QMovie(loadingGif, parent=self)
        movielabel = QLabel(self)
        movielabel.setMovie(self.movie)
        self.layout.addWidget(movielabel)
        self.movie.start()
        
        
        self.setLayout(self.layout)
        self.resize(200,200)


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
        
# For using separate thread on long-running tasks
class FileWorker(QObject):
    finished = Signal()
    exceptCaught = Signal()
    errorStr = ""
    mp = None
    delim = "\s+"
    annotDF = pd.DataFrame()
    columnMap = {}
    plotParams={}
    canvasPlot = None
    copyHolder = None
    logp = None
    
    
    def loadDataFile(self):
        try:
            self.mp.load_data(delim=self.delim) 
            self.copyHolder.df = self.mp.df.copy()
        except Exception as ex:
            traceback_str = ''.join(traceback.format_tb(ex.__traceback__))
            self.errorStr = str(ex) + "<p>Details: " + traceback_str + "</p>"
            self.exceptCaught.emit()
        else:
            self.finished.emit()
    
    def thinData(self):
        try:
            self.mp.df = self.copyHolder.df.copy()
            self.mp.clean_data(col_map=self.columnMap, logp=self.logp)
            if not self.annotDF.empty: 
                self.mp.add_annotations(self.annotDF, extra_cols=self.addCols)
            self.mp.get_thinned_data()
        except Exception as ex:
            traceback_str = ''.join(traceback.format_tb(ex.__traceback__))
            self.errorStr = str(ex) + "<p>Details: " + traceback_str + "</p>"
            self.exceptCaught.emit()
        else:
            self.finished.emit()
            
# For dataframe backup and manipulation
class FrameHolder:
    def __init__(self):
        self.df = pd.DataFrame()

def get_path(filename):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, filename)
    else:
        return filename

if __name__ == "__main__":
    app = QApplication(sys.argv)
    QApplication.setAttribute(Qt.AA_DontUseNativeMenuBar);
    win = Window()
    win.show()
    sys.exit(app.exec())
