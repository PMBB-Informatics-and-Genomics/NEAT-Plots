# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.3.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
    QFormLayout, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMenu, QMenuBar,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QStatusBar, QTabWidget, QVBoxLayout,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1135, 891)
        MainWindow.setMinimumSize(QSize(0, 0))
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName(u"actionExit")
        self.actionInformation = QAction(MainWindow)
        self.actionInformation.setObjectName(u"actionInformation")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_2 = QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setMinimumSize(QSize(400, 0))
        self.tabWidget.setMaximumSize(QSize(400, 16777215))
        font = QFont()
        font.setFamilies([u"Arial"])
        self.tabWidget.setFont(font)
        self.tabWidget.setContextMenuPolicy(Qt.NoContextMenu)
        self.tabWidget.setTabShape(QTabWidget.Rounded)
        self.tabLoad = QWidget()
        self.tabLoad.setObjectName(u"tabLoad")
        self.layoutTabLoad = QVBoxLayout(self.tabLoad)
        self.layoutTabLoad.setObjectName(u"layoutTabLoad")
        self.btnNext1 = QPushButton(self.tabLoad)
        self.btnNext1.setObjectName(u"btnNext1")
        self.btnNext1.setMinimumSize(QSize(100, 0))
        self.btnNext1.setMaximumSize(QSize(150, 16777215))

        self.layoutTabLoad.addWidget(self.btnNext1, 0, Qt.AlignHCenter)

        self.line = QFrame(self.tabLoad)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.layoutTabLoad.addWidget(self.line)

        self.groupBoxLoadData = QGroupBox(self.tabLoad)
        self.groupBoxLoadData.setObjectName(u"groupBoxLoadData")
        self.groupBoxLoadData.setMaximumSize(QSize(16777215, 400))
        self.formLayout_4 = QFormLayout(self.groupBoxLoadData)
        self.formLayout_4.setObjectName(u"formLayout_4")
        self.labelDataFn = QLabel(self.groupBoxLoadData)
        self.labelDataFn.setObjectName(u"labelDataFn")

        self.formLayout_4.setWidget(0, QFormLayout.LabelRole, self.labelDataFn)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.lineDataFn = QLineEdit(self.groupBoxLoadData)
        self.lineDataFn.setObjectName(u"lineDataFn")
        self.lineDataFn.setEnabled(True)
        font1 = QFont()
        font1.setFamilies([u"Arial"])
        font1.setBold(False)
        self.lineDataFn.setFont(font1)
        self.lineDataFn.setFrame(True)
        self.lineDataFn.setClearButtonEnabled(False)

        self.verticalLayout_2.addWidget(self.lineDataFn)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.btnDataFn = QPushButton(self.groupBoxLoadData)
        self.btnDataFn.setObjectName(u"btnDataFn")

        self.horizontalLayout.addWidget(self.btnDataFn)

        self.btnDataFnLoad = QPushButton(self.groupBoxLoadData)
        self.btnDataFnLoad.setObjectName(u"btnDataFnLoad")

        self.horizontalLayout.addWidget(self.btnDataFnLoad)


        self.verticalLayout_2.addLayout(self.horizontalLayout)


        self.formLayout_4.setLayout(0, QFormLayout.FieldRole, self.verticalLayout_2)

        self.labelNumRows = QLabel(self.groupBoxLoadData)
        self.labelNumRows.setObjectName(u"labelNumRows")

        self.formLayout_4.setWidget(1, QFormLayout.LabelRole, self.labelNumRows)

        self.spinTestRows = QSpinBox(self.groupBoxLoadData)
        self.spinTestRows.setObjectName(u"spinTestRows")
        self.spinTestRows.setMaximum(10000000)
        self.spinTestRows.setSingleStep(100)
        self.spinTestRows.setValue(100000)

        self.formLayout_4.setWidget(1, QFormLayout.FieldRole, self.spinTestRows)

        self.labelFileDelm = QLabel(self.groupBoxLoadData)
        self.labelFileDelm.setObjectName(u"labelFileDelm")

        self.formLayout_4.setWidget(2, QFormLayout.LabelRole, self.labelFileDelm)

        self.bxDelimiter = QComboBox(self.groupBoxLoadData)
        self.bxDelimiter.addItem("")
        self.bxDelimiter.addItem("")
        self.bxDelimiter.addItem("")
        self.bxDelimiter.addItem("")
        self.bxDelimiter.setObjectName(u"bxDelimiter")
        self.bxDelimiter.setEditable(True)

        self.formLayout_4.setWidget(2, QFormLayout.FieldRole, self.bxDelimiter)


        self.layoutTabLoad.addWidget(self.groupBoxLoadData)

        self.tabWidget.addTab(self.tabLoad, "")
        self.tabProcess = QWidget()
        self.tabProcess.setObjectName(u"tabProcess")
        self.layoutTabProcess = QVBoxLayout(self.tabProcess)
        self.layoutTabProcess.setObjectName(u"layoutTabProcess")
        self.horizontalLayoutProcessButtons = QHBoxLayout()
        self.horizontalLayoutProcessButtons.setObjectName(u"horizontalLayoutProcessButtons")
        self.btnPrev1 = QPushButton(self.tabProcess)
        self.btnPrev1.setObjectName(u"btnPrev1")
        self.btnPrev1.setMinimumSize(QSize(100, 0))
        self.btnPrev1.setMaximumSize(QSize(150, 16777215))
        self.btnPrev1.setStyleSheet(u"")

        self.horizontalLayoutProcessButtons.addWidget(self.btnPrev1)

        self.btnNext2 = QPushButton(self.tabProcess)
        self.btnNext2.setObjectName(u"btnNext2")
        self.btnNext2.setMinimumSize(QSize(100, 0))
        self.btnNext2.setMaximumSize(QSize(150, 16777215))
        self.btnNext2.setStyleSheet(u"")

        self.horizontalLayoutProcessButtons.addWidget(self.btnNext2)


        self.layoutTabProcess.addLayout(self.horizontalLayoutProcessButtons)

        self.line_2 = QFrame(self.tabProcess)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.layoutTabProcess.addWidget(self.line_2)

        self.groupBoxSelectCols = QGroupBox(self.tabProcess)
        self.groupBoxSelectCols.setObjectName(u"groupBoxSelectCols")
        self.groupBoxSelectCols.setMinimumSize(QSize(0, 0))
        self.groupBoxSelectCols.setMaximumSize(QSize(16777215, 250))
        self.formLayout_3 = QFormLayout(self.groupBoxSelectCols)
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.lblChromCol = QLabel(self.groupBoxSelectCols)
        self.lblChromCol.setObjectName(u"lblChromCol")

        self.formLayout_3.setWidget(0, QFormLayout.LabelRole, self.lblChromCol)

        self.bxChromCol = QComboBox(self.groupBoxSelectCols)
        self.bxChromCol.setObjectName(u"bxChromCol")

        self.formLayout_3.setWidget(0, QFormLayout.FieldRole, self.bxChromCol)

        self.lblPosCol = QLabel(self.groupBoxSelectCols)
        self.lblPosCol.setObjectName(u"lblPosCol")

        self.formLayout_3.setWidget(1, QFormLayout.LabelRole, self.lblPosCol)

        self.bxPosCol = QComboBox(self.groupBoxSelectCols)
        self.bxPosCol.setObjectName(u"bxPosCol")

        self.formLayout_3.setWidget(1, QFormLayout.FieldRole, self.bxPosCol)

        self.lblIDCol = QLabel(self.groupBoxSelectCols)
        self.lblIDCol.setObjectName(u"lblIDCol")

        self.formLayout_3.setWidget(2, QFormLayout.LabelRole, self.lblIDCol)

        self.bxIDCol = QComboBox(self.groupBoxSelectCols)
        self.bxIDCol.setObjectName(u"bxIDCol")

        self.formLayout_3.setWidget(2, QFormLayout.FieldRole, self.bxIDCol)

        self.lblPvalue = QLabel(self.groupBoxSelectCols)
        self.lblPvalue.setObjectName(u"lblPvalue")

        self.formLayout_3.setWidget(3, QFormLayout.LabelRole, self.lblPvalue)

        self.bxPvalueCol = QComboBox(self.groupBoxSelectCols)
        self.bxPvalueCol.setObjectName(u"bxPvalueCol")

        self.formLayout_3.setWidget(3, QFormLayout.FieldRole, self.bxPvalueCol)


        self.layoutTabProcess.addWidget(self.groupBoxSelectCols)

        self.tabWidget.addTab(self.tabProcess, "")
        self.tabAnnotations = QWidget()
        self.tabAnnotations.setObjectName(u"tabAnnotations")
        self.layoutTabAnnotations = QVBoxLayout(self.tabAnnotations)
        self.layoutTabAnnotations.setObjectName(u"layoutTabAnnotations")
        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.btnPrev2 = QPushButton(self.tabAnnotations)
        self.btnPrev2.setObjectName(u"btnPrev2")
        self.btnPrev2.setMinimumSize(QSize(100, 0))
        self.btnPrev2.setMaximumSize(QSize(150, 16777215))
        self.btnPrev2.setStyleSheet(u"")

        self.horizontalLayout_6.addWidget(self.btnPrev2)

        self.btnNext3 = QPushButton(self.tabAnnotations)
        self.btnNext3.setObjectName(u"btnNext3")
        self.btnNext3.setMinimumSize(QSize(100, 0))
        self.btnNext3.setMaximumSize(QSize(150, 16777215))
        self.btnNext3.setStyleSheet(u"")

        self.horizontalLayout_6.addWidget(self.btnNext3)


        self.layoutTabAnnotations.addLayout(self.horizontalLayout_6)

        self.line_3 = QFrame(self.tabAnnotations)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.layoutTabAnnotations.addWidget(self.line_3)

        self.groupBoxAnnFiles = QGroupBox(self.tabAnnotations)
        self.groupBoxAnnFiles.setObjectName(u"groupBoxAnnFiles")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBoxAnnFiles.sizePolicy().hasHeightForWidth())
        self.groupBoxAnnFiles.setSizePolicy(sizePolicy)
        self.groupBoxAnnFiles.setMaximumSize(QSize(16777215, 200))
        self.gridLayout_4 = QGridLayout(self.groupBoxAnnFiles)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.labelDataFn_3 = QLabel(self.groupBoxAnnFiles)
        self.labelDataFn_3.setObjectName(u"labelDataFn_3")

        self.gridLayout_4.addWidget(self.labelDataFn_3, 0, 0, 1, 1)

        self.lineAnnotationsFn = QLineEdit(self.groupBoxAnnFiles)
        self.lineAnnotationsFn.setObjectName(u"lineAnnotationsFn")
        self.lineAnnotationsFn.setEnabled(True)
        self.lineAnnotationsFn.setFont(font1)
        self.lineAnnotationsFn.setFrame(True)
        self.lineAnnotationsFn.setClearButtonEnabled(False)

        self.gridLayout_4.addWidget(self.lineAnnotationsFn, 0, 1, 1, 3)

        self.bxAnnotDelimiter = QComboBox(self.groupBoxAnnFiles)
        self.bxAnnotDelimiter.addItem("")
        self.bxAnnotDelimiter.addItem("")
        self.bxAnnotDelimiter.addItem("")
        self.bxAnnotDelimiter.addItem("")
        self.bxAnnotDelimiter.setObjectName(u"bxAnnotDelimiter")
        self.bxAnnotDelimiter.setStyleSheet(u"text-align: center;")
        self.bxAnnotDelimiter.setEditable(True)
        self.bxAnnotDelimiter.setFrame(False)

        self.gridLayout_4.addWidget(self.bxAnnotDelimiter, 1, 1, 1, 2)

        self.labelAnnotDelm = QLabel(self.groupBoxAnnFiles)
        self.labelAnnotDelm.setObjectName(u"labelAnnotDelm")

        self.gridLayout_4.addWidget(self.labelAnnotDelm, 1, 3, 1, 1)

        self.btnAnnotationsFn = QPushButton(self.groupBoxAnnFiles)
        self.btnAnnotationsFn.setObjectName(u"btnAnnotationsFn")

        self.gridLayout_4.addWidget(self.btnAnnotationsFn, 2, 1, 1, 1)

        self.btnLoadAnnotFile = QPushButton(self.groupBoxAnnFiles)
        self.btnLoadAnnotFile.setObjectName(u"btnLoadAnnotFile")

        self.gridLayout_4.addWidget(self.btnLoadAnnotFile, 2, 2, 1, 2)

        self.lblKnownGeneFn = QLabel(self.groupBoxAnnFiles)
        self.lblKnownGeneFn.setObjectName(u"lblKnownGeneFn")

        self.gridLayout_4.addWidget(self.lblKnownGeneFn, 3, 0, 1, 1)

        self.lineKnownGenesFn = QLineEdit(self.groupBoxAnnFiles)
        self.lineKnownGenesFn.setObjectName(u"lineKnownGenesFn")
        self.lineKnownGenesFn.setEnabled(True)
        self.lineKnownGenesFn.setFont(font1)
        self.lineKnownGenesFn.setFrame(True)
        self.lineKnownGenesFn.setClearButtonEnabled(False)

        self.gridLayout_4.addWidget(self.lineKnownGenesFn, 3, 1, 1, 3)

        self.btnKnownGenesFn = QPushButton(self.groupBoxAnnFiles)
        self.btnKnownGenesFn.setObjectName(u"btnKnownGenesFn")

        self.gridLayout_4.addWidget(self.btnKnownGenesFn, 4, 1, 1, 1)

        self.btnLoadKnownGenesFile = QPushButton(self.groupBoxAnnFiles)
        self.btnLoadKnownGenesFile.setObjectName(u"btnLoadKnownGenesFile")

        self.gridLayout_4.addWidget(self.btnLoadKnownGenesFile, 4, 2, 1, 2)


        self.layoutTabAnnotations.addWidget(self.groupBoxAnnFiles)

        self.groupBoxAnnCols = QGroupBox(self.tabAnnotations)
        self.groupBoxAnnCols.setObjectName(u"groupBoxAnnCols")
        self.groupBoxAnnCols.setMaximumSize(QSize(16777215, 320))
        self.formLayout_6 = QFormLayout(self.groupBoxAnnCols)
        self.formLayout_6.setObjectName(u"formLayout_6")
        self.lblChromAnn = QLabel(self.groupBoxAnnCols)
        self.lblChromAnn.setObjectName(u"lblChromAnn")

        self.formLayout_6.setWidget(0, QFormLayout.LabelRole, self.lblChromAnn)

        self.bxChromAnn = QComboBox(self.groupBoxAnnCols)
        self.bxChromAnn.setObjectName(u"bxChromAnn")

        self.formLayout_6.setWidget(0, QFormLayout.FieldRole, self.bxChromAnn)

        self.lblPosAnn = QLabel(self.groupBoxAnnCols)
        self.lblPosAnn.setObjectName(u"lblPosAnn")

        self.formLayout_6.setWidget(1, QFormLayout.LabelRole, self.lblPosAnn)

        self.bxPosAnn = QComboBox(self.groupBoxAnnCols)
        self.bxPosAnn.setObjectName(u"bxPosAnn")

        self.formLayout_6.setWidget(1, QFormLayout.FieldRole, self.bxPosAnn)

        self.lblIDAnn = QLabel(self.groupBoxAnnCols)
        self.lblIDAnn.setObjectName(u"lblIDAnn")

        self.formLayout_6.setWidget(2, QFormLayout.LabelRole, self.lblIDAnn)

        self.bxIDAnn = QComboBox(self.groupBoxAnnCols)
        self.bxIDAnn.setObjectName(u"bxIDAnn")

        self.formLayout_6.setWidget(2, QFormLayout.FieldRole, self.bxIDAnn)

        self.lblOtherAnn = QLabel(self.groupBoxAnnCols)
        self.lblOtherAnn.setObjectName(u"lblOtherAnn")

        self.formLayout_6.setWidget(3, QFormLayout.LabelRole, self.lblOtherAnn)

        self.listOtherAnn = QListWidget(self.groupBoxAnnCols)
        self.listOtherAnn.setObjectName(u"listOtherAnn")
        self.listOtherAnn.setMaximumSize(QSize(16777215, 200))
        self.listOtherAnn.setSelectionMode(QAbstractItemView.MultiSelection)

        self.formLayout_6.setWidget(3, QFormLayout.FieldRole, self.listOtherAnn)


        self.layoutTabAnnotations.addWidget(self.groupBoxAnnCols)

        self.tabWidget.addTab(self.tabAnnotations, "")
        self.tabPlot = QWidget()
        self.tabPlot.setObjectName(u"tabPlot")
        self.layoutTabPlot = QVBoxLayout(self.tabPlot)
#ifndef Q_OS_MAC
        self.layoutTabPlot.setSpacing(-1)
#endif
        self.layoutTabPlot.setObjectName(u"layoutTabPlot")
        self.layoutTabPlot.setContentsMargins(-1, -1, 1, -1)
        self.scrollAreaPlot = QScrollArea(self.tabPlot)
        self.scrollAreaPlot.setObjectName(u"scrollAreaPlot")
        self.scrollAreaPlot.setWidgetResizable(True)
        self.scrollAreaPlotContents = QWidget()
        self.scrollAreaPlotContents.setObjectName(u"scrollAreaPlotContents")
        self.scrollAreaPlotContents.setGeometry(QRect(0, 0, 371, 752))
        self.verticalLayoutScrollAreaPlot = QVBoxLayout(self.scrollAreaPlotContents)
        self.verticalLayoutScrollAreaPlot.setObjectName(u"verticalLayoutScrollAreaPlot")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.btnPrev3 = QPushButton(self.scrollAreaPlotContents)
        self.btnPrev3.setObjectName(u"btnPrev3")
        self.btnPrev3.setMaximumSize(QSize(120, 16777215))
        self.btnPrev3.setLayoutDirection(Qt.LeftToRight)
        self.btnPrev3.setStyleSheet(u"")

        self.horizontalLayout_3.addWidget(self.btnPrev3)


        self.verticalLayoutScrollAreaPlot.addLayout(self.horizontalLayout_3)

        self.line_4 = QFrame(self.scrollAreaPlotContents)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setFrameShape(QFrame.HLine)
        self.line_4.setFrameShadow(QFrame.Sunken)

        self.verticalLayoutScrollAreaPlot.addWidget(self.line_4)

        self.groupBoxPval = QGroupBox(self.scrollAreaPlotContents)
        self.groupBoxPval.setObjectName(u"groupBoxPval")
        self.groupBoxPval.setMaximumSize(QSize(16777215, 150))
        self.formLayout = QFormLayout(self.groupBoxPval)
        self.formLayout.setObjectName(u"formLayout")
        self.lblSuggestThresh = QLabel(self.groupBoxPval)
        self.lblSuggestThresh.setObjectName(u"lblSuggestThresh")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.lblSuggestThresh)

        self.lineSuggestThresh = QLineEdit(self.groupBoxPval)
        self.lineSuggestThresh.setObjectName(u"lineSuggestThresh")
        self.lineSuggestThresh.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.lineSuggestThresh)

        self.lblSigThresh = QLabel(self.groupBoxPval)
        self.lblSigThresh.setObjectName(u"lblSigThresh")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.lblSigThresh)

        self.lineSigThresh = QLineEdit(self.groupBoxPval)
        self.lineSigThresh.setObjectName(u"lineSigThresh")
        self.lineSigThresh.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.lineSigThresh)

        self.lblMaxLogP = QLabel(self.groupBoxPval)
        self.lblMaxLogP.setObjectName(u"lblMaxLogP")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.lblMaxLogP)

        self.lineMaxLogP = QLineEdit(self.groupBoxPval)
        self.lineMaxLogP.setObjectName(u"lineMaxLogP")
        self.lineMaxLogP.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.lineMaxLogP)


        self.verticalLayoutScrollAreaPlot.addWidget(self.groupBoxPval)

        self.groupBoxSignals = QGroupBox(self.scrollAreaPlotContents)
        self.groupBoxSignals.setObjectName(u"groupBoxSignals")
        self.groupBoxSignals.setMaximumSize(QSize(16777215, 180))
        self.formLayout_2 = QFormLayout(self.groupBoxSignals)
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.lblLDBlockWidth = QLabel(self.groupBoxSignals)
        self.lblLDBlockWidth.setObjectName(u"lblLDBlockWidth")

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.lblLDBlockWidth)

        self.lineLDLBlockWidth = QLineEdit(self.groupBoxSignals)
        self.lineLDLBlockWidth.setObjectName(u"lineLDLBlockWidth")
        self.lineLDLBlockWidth.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.formLayout_2.setWidget(0, QFormLayout.FieldRole, self.lineLDLBlockWidth)

        self.lblMergeGenes = QLabel(self.groupBoxSignals)
        self.lblMergeGenes.setObjectName(u"lblMergeGenes")
        self.lblMergeGenes.setMidLineWidth(0)

        self.formLayout_2.setWidget(1, QFormLayout.LabelRole, self.lblMergeGenes)

        self.bxMergeGenes = QComboBox(self.groupBoxSignals)
        self.bxMergeGenes.addItem("")
        self.bxMergeGenes.addItem("")
        self.bxMergeGenes.setObjectName(u"bxMergeGenes")

        self.formLayout_2.setWidget(1, QFormLayout.FieldRole, self.bxMergeGenes)

        self.lblBoostKnown = QLabel(self.groupBoxSignals)
        self.lblBoostKnown.setObjectName(u"lblBoostKnown")

        self.formLayout_2.setWidget(2, QFormLayout.LabelRole, self.lblBoostKnown)

        self.bxBoostKnown = QComboBox(self.groupBoxSignals)
        self.bxBoostKnown.addItem("")
        self.bxBoostKnown.addItem("")
        self.bxBoostKnown.setObjectName(u"bxBoostKnown")

        self.formLayout_2.setWidget(2, QFormLayout.FieldRole, self.bxBoostKnown)


        self.verticalLayoutScrollAreaPlot.addWidget(self.groupBoxSignals)

        self.groupBoxPlotOptions = QGroupBox(self.scrollAreaPlotContents)
        self.groupBoxPlotOptions.setObjectName(u"groupBoxPlotOptions")
        self.gridLayout = QGridLayout(self.groupBoxPlotOptions)
        self.gridLayout.setObjectName(u"gridLayout")
        self.lineDataTitle = QLineEdit(self.groupBoxPlotOptions)
        self.lineDataTitle.setObjectName(u"lineDataTitle")
        self.lineDataTitle.setEnabled(True)
        self.lineDataTitle.setFont(font1)
        self.lineDataTitle.setFrame(True)
        self.lineDataTitle.setClearButtonEnabled(False)

        self.gridLayout.addWidget(self.lineDataTitle, 3, 1, 1, 2)

        self.chkBoxWithTitle = QCheckBox(self.groupBoxPlotOptions)
        self.chkBoxWithTitle.setObjectName(u"chkBoxWithTitle")
        self.chkBoxWithTitle.setLayoutDirection(Qt.RightToLeft)

        self.gridLayout.addWidget(self.chkBoxWithTitle, 0, 2, 1, 1)

        self.labelTitle = QLabel(self.groupBoxPlotOptions)
        self.labelTitle.setObjectName(u"labelTitle")
        self.labelTitle.setLayoutDirection(Qt.LeftToRight)
        self.labelTitle.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.labelTitle, 3, 0, 1, 1)

        self.chkBoxSignalsOnly = QCheckBox(self.groupBoxPlotOptions)
        self.chkBoxSignalsOnly.setObjectName(u"chkBoxSignalsOnly")
        self.chkBoxSignalsOnly.setLayoutDirection(Qt.RightToLeft)

        self.gridLayout.addWidget(self.chkBoxSignalsOnly, 0, 1, 1, 1)

        self.chkBoxWithTable = QCheckBox(self.groupBoxPlotOptions)
        self.chkBoxWithTable.setObjectName(u"chkBoxWithTable")
        self.chkBoxWithTable.setLayoutDirection(Qt.RightToLeft)
        self.chkBoxWithTable.setChecked(True)

        self.gridLayout.addWidget(self.chkBoxWithTable, 0, 0, 1, 1)

        self.chkBoxInvert = QCheckBox(self.groupBoxPlotOptions)
        self.chkBoxInvert.setObjectName(u"chkBoxInvert")
        self.chkBoxInvert.setLayoutDirection(Qt.RightToLeft)

        self.gridLayout.addWidget(self.chkBoxInvert, 1, 2, 1, 1)

        self.chkBoxVertical = QCheckBox(self.groupBoxPlotOptions)
        self.chkBoxVertical.setObjectName(u"chkBoxVertical")
        self.chkBoxVertical.setLayoutDirection(Qt.RightToLeft)
        self.chkBoxVertical.setChecked(True)
        self.chkBoxVertical.setTristate(False)

        self.gridLayout.addWidget(self.chkBoxVertical, 1, 1, 1, 1)


        self.verticalLayoutScrollAreaPlot.addWidget(self.groupBoxPlotOptions)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.btnPlot = QPushButton(self.scrollAreaPlotContents)
        self.btnPlot.setObjectName(u"btnPlot")
        self.btnPlot.setMaximumSize(QSize(150, 16777215))
        self.btnPlot.setStyleSheet(u"")

        self.horizontalLayout_2.addWidget(self.btnPlot)

        self.btnSave = QPushButton(self.scrollAreaPlotContents)
        self.btnSave.setObjectName(u"btnSave")
        self.btnSave.setMaximumSize(QSize(150, 16777215))
        self.btnSave.setStyleSheet(u"")

        self.horizontalLayout_2.addWidget(self.btnSave)


        self.verticalLayoutScrollAreaPlot.addLayout(self.horizontalLayout_2)

        self.scrollAreaPlot.setWidget(self.scrollAreaPlotContents)

        self.layoutTabPlot.addWidget(self.scrollAreaPlot)

        self.tabWidget.addTab(self.tabPlot, "")

        self.gridLayout_3.addWidget(self.tabWidget, 0, 0, 1, 1)

        self.framePlot = QFrame(self.centralwidget)
        self.framePlot.setObjectName(u"framePlot")
        self.framePlot.setMaximumSize(QSize(16777215, 16777215))
        self.framePlot.setFont(font)
        self.framePlot.setFrameShape(QFrame.StyledPanel)
        self.framePlot.setFrameShadow(QFrame.Raised)
        self.progressBarBlue = QProgressBar(self.framePlot)
        self.progressBarBlue.setObjectName(u"progressBarBlue")
        self.progressBarBlue.setGeometry(QRect(196, 364, 300, 80))
        self.progressBarBlue.setMinimumSize(QSize(300, 80))
        self.progressBarBlue.setMaximumSize(QSize(300, 80))
        self.progressBarBlue.setStyleSheet(u"#progressBarBlue {\n"
"    border: 2px solid #2196F3;\n"
"    border-radius: 5px;\n"
"    background-color: #E0E0E0;\n"
"}\n"
"#progressBarBlue::chunk {\n"
"    background-color: #2196F3;\n"
"    width: 10px; \n"
"    margin: 0.5px;\n"
"}")
        self.progressBarBlue.setValue(1)

        self.gridLayout_3.addWidget(self.framePlot, 0, 1, 1, 1)


        self.gridLayout_2.addLayout(self.gridLayout_3, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1135, 24))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuAbout = QMenu(self.menubar)
        self.menuAbout.setObjectName(u"menuAbout")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        QWidget.setTabOrder(self.tabWidget, self.btnNext1)
        QWidget.setTabOrder(self.btnNext1, self.lineDataFn)
        QWidget.setTabOrder(self.lineDataFn, self.btnDataFn)
        QWidget.setTabOrder(self.btnDataFn, self.btnDataFnLoad)
        QWidget.setTabOrder(self.btnDataFnLoad, self.spinTestRows)
        QWidget.setTabOrder(self.spinTestRows, self.bxDelimiter)
        QWidget.setTabOrder(self.bxDelimiter, self.btnPrev1)
        QWidget.setTabOrder(self.btnPrev1, self.btnNext2)
        QWidget.setTabOrder(self.btnNext2, self.bxChromCol)
        QWidget.setTabOrder(self.bxChromCol, self.bxPosCol)
        QWidget.setTabOrder(self.bxPosCol, self.bxIDCol)
        QWidget.setTabOrder(self.bxIDCol, self.bxPvalueCol)
        QWidget.setTabOrder(self.bxPvalueCol, self.btnPrev2)
        QWidget.setTabOrder(self.btnPrev2, self.btnNext3)
        QWidget.setTabOrder(self.btnNext3, self.lineAnnotationsFn)
        QWidget.setTabOrder(self.lineAnnotationsFn, self.btnAnnotationsFn)
        QWidget.setTabOrder(self.btnAnnotationsFn, self.btnLoadAnnotFile)
        QWidget.setTabOrder(self.btnLoadAnnotFile, self.lineKnownGenesFn)
        QWidget.setTabOrder(self.lineKnownGenesFn, self.btnKnownGenesFn)
        QWidget.setTabOrder(self.btnKnownGenesFn, self.btnLoadKnownGenesFile)
        QWidget.setTabOrder(self.btnLoadKnownGenesFile, self.bxChromAnn)
        QWidget.setTabOrder(self.bxChromAnn, self.bxPosAnn)
        QWidget.setTabOrder(self.bxPosAnn, self.bxIDAnn)
        QWidget.setTabOrder(self.bxIDAnn, self.listOtherAnn)
        QWidget.setTabOrder(self.listOtherAnn, self.scrollAreaPlot)
        QWidget.setTabOrder(self.scrollAreaPlot, self.btnPrev3)
        QWidget.setTabOrder(self.btnPrev3, self.lineSuggestThresh)
        QWidget.setTabOrder(self.lineSuggestThresh, self.lineSigThresh)
        QWidget.setTabOrder(self.lineSigThresh, self.lineMaxLogP)
        QWidget.setTabOrder(self.lineMaxLogP, self.lineLDLBlockWidth)
        QWidget.setTabOrder(self.lineLDLBlockWidth, self.bxMergeGenes)
        QWidget.setTabOrder(self.bxMergeGenes, self.bxBoostKnown)
        QWidget.setTabOrder(self.bxBoostKnown, self.chkBoxWithTable)
        QWidget.setTabOrder(self.chkBoxWithTable, self.chkBoxSignalsOnly)
        QWidget.setTabOrder(self.chkBoxSignalsOnly, self.chkBoxWithTitle)
        QWidget.setTabOrder(self.chkBoxWithTitle, self.chkBoxVertical)
        QWidget.setTabOrder(self.chkBoxVertical, self.chkBoxInvert)
        QWidget.setTabOrder(self.chkBoxInvert, self.lineDataTitle)
        QWidget.setTabOrder(self.lineDataTitle, self.btnPlot)
        QWidget.setTabOrder(self.btnPlot, self.btnSave)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuAbout.menuAction())
        self.menuFile.addAction(self.actionExit)
        self.menuAbout.addAction(self.actionInformation)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)
        self.bxDelimiter.setCurrentIndex(0)
        self.bxAnnotDelimiter.setCurrentIndex(0)
        self.bxMergeGenes.setCurrentIndex(1)
        self.bxBoostKnown.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"NEAT", None))
        self.actionExit.setText(QCoreApplication.translate("MainWindow", u"&Exit", None))
        self.actionInformation.setText(QCoreApplication.translate("MainWindow", u"Information", None))
        self.btnNext1.setText(QCoreApplication.translate("MainWindow", u"Next", None))
        self.groupBoxLoadData.setTitle(QCoreApplication.translate("MainWindow", u"Input Data", None))
        self.labelDataFn.setText(QCoreApplication.translate("MainWindow", u"Data file", None))
#if QT_CONFIG(tooltip)
        self.lineDataFn.setToolTip(QCoreApplication.translate("MainWindow", u"Data input file", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.btnDataFn.setToolTip(QCoreApplication.translate("MainWindow", u"Select file", None))
#endif // QT_CONFIG(tooltip)
        self.btnDataFn.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.btnDataFnLoad.setText(QCoreApplication.translate("MainWindow", u"Load", None))
#if QT_CONFIG(tooltip)
        self.labelNumRows.setToolTip(QCoreApplication.translate("MainWindow", u"Set to zero to include all rows", None))
#endif // QT_CONFIG(tooltip)
        self.labelNumRows.setText(QCoreApplication.translate("MainWindow", u"Test Rows", None))
#if QT_CONFIG(tooltip)
        self.spinTestRows.setToolTip(QCoreApplication.translate("MainWindow", u"Limit input ", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.labelFileDelm.setToolTip(QCoreApplication.translate("MainWindow", u"Delimiter separating columns in data file", None))
#endif // QT_CONFIG(tooltip)
        self.labelFileDelm.setText(QCoreApplication.translate("MainWindow", u"File delimiter", None))
        self.bxDelimiter.setItemText(0, QCoreApplication.translate("MainWindow", u"whitespace", None))
        self.bxDelimiter.setItemText(1, QCoreApplication.translate("MainWindow", u", (comma)", None))
        self.bxDelimiter.setItemText(2, QCoreApplication.translate("MainWindow", u"tab", None))
        self.bxDelimiter.setItemText(3, QCoreApplication.translate("MainWindow", u"| (pipe)", None))

#if QT_CONFIG(tooltip)
        self.bxDelimiter.setToolTip(QCoreApplication.translate("MainWindow", u"Splits input file into columns", None))
#endif // QT_CONFIG(tooltip)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabLoad), QCoreApplication.translate("MainWindow", u"Load", None))
        self.btnPrev1.setText(QCoreApplication.translate("MainWindow", u"Previous", None))
        self.btnNext2.setText(QCoreApplication.translate("MainWindow", u"Next", None))
        self.groupBoxSelectCols.setTitle(QCoreApplication.translate("MainWindow", u"Select columns ", None))
#if QT_CONFIG(tooltip)
        self.lblChromCol.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.lblChromCol.setText(QCoreApplication.translate("MainWindow", u"Chromosome column", None))
#if QT_CONFIG(tooltip)
        self.bxChromCol.setToolTip(QCoreApplication.translate("MainWindow", u"Column header in input file", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.lblPosCol.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.lblPosCol.setText(QCoreApplication.translate("MainWindow", u"Position column", None))
#if QT_CONFIG(tooltip)
        self.bxPosCol.setToolTip(QCoreApplication.translate("MainWindow", u"Column header in input file", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.lblIDCol.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.lblIDCol.setText(QCoreApplication.translate("MainWindow", u"ID column", None))
#if QT_CONFIG(tooltip)
        self.bxIDCol.setToolTip(QCoreApplication.translate("MainWindow", u"Column header in input file", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.lblPvalue.setToolTip(QCoreApplication.translate("MainWindow", u"Either P value column OR Log-P column name", None))
#endif // QT_CONFIG(tooltip)
        self.lblPvalue.setText(QCoreApplication.translate("MainWindow", u"P-Value column", None))
#if QT_CONFIG(tooltip)
        self.bxPvalueCol.setToolTip(QCoreApplication.translate("MainWindow", u"Column header in input file", None))
#endif // QT_CONFIG(tooltip)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabProcess), QCoreApplication.translate("MainWindow", u"Process", None))
        self.btnPrev2.setText(QCoreApplication.translate("MainWindow", u"Previous", None))
        self.btnNext3.setText(QCoreApplication.translate("MainWindow", u"Next", None))
        self.groupBoxAnnFiles.setTitle(QCoreApplication.translate("MainWindow", u"Load Files", None))
        self.labelDataFn_3.setText(QCoreApplication.translate("MainWindow", u"Annotations", None))
#if QT_CONFIG(tooltip)
        self.lineAnnotationsFn.setToolTip(QCoreApplication.translate("MainWindow", u"Annotations file", None))
#endif // QT_CONFIG(tooltip)
        self.bxAnnotDelimiter.setItemText(0, QCoreApplication.translate("MainWindow", u", (comma)", None))
        self.bxAnnotDelimiter.setItemText(1, QCoreApplication.translate("MainWindow", u"whitespace", None))
        self.bxAnnotDelimiter.setItemText(2, QCoreApplication.translate("MainWindow", u"tab", None))
        self.bxAnnotDelimiter.setItemText(3, QCoreApplication.translate("MainWindow", u"| (pipe)", None))

#if QT_CONFIG(tooltip)
        self.bxAnnotDelimiter.setToolTip(QCoreApplication.translate("MainWindow", u"Splits annotation file into columns", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.labelAnnotDelm.setToolTip(QCoreApplication.translate("MainWindow", u"Delimiter separating columns in data file", None))
#endif // QT_CONFIG(tooltip)
        self.labelAnnotDelm.setText(QCoreApplication.translate("MainWindow", u"Delimiter", None))
#if QT_CONFIG(tooltip)
        self.btnAnnotationsFn.setToolTip(QCoreApplication.translate("MainWindow", u"Select file", None))
#endif // QT_CONFIG(tooltip)
        self.btnAnnotationsFn.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.btnLoadAnnotFile.setText(QCoreApplication.translate("MainWindow", u"Load", None))
        self.lblKnownGeneFn.setText(QCoreApplication.translate("MainWindow", u"Known genes", None))
#if QT_CONFIG(tooltip)
        self.lineKnownGenesFn.setToolTip(QCoreApplication.translate("MainWindow", u"Gene information file", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.btnKnownGenesFn.setToolTip(QCoreApplication.translate("MainWindow", u"Select file", None))
#endif // QT_CONFIG(tooltip)
        self.btnKnownGenesFn.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.btnLoadKnownGenesFile.setText(QCoreApplication.translate("MainWindow", u"Load", None))
        self.groupBoxAnnCols.setTitle(QCoreApplication.translate("MainWindow", u"Annotation file column identification", None))
#if QT_CONFIG(tooltip)
        self.lblChromAnn.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.lblChromAnn.setText(QCoreApplication.translate("MainWindow", u"Chromosome ", None))
#if QT_CONFIG(tooltip)
        self.bxChromAnn.setToolTip(QCoreApplication.translate("MainWindow", u"Column header in annotations file ", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.lblPosAnn.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.lblPosAnn.setText(QCoreApplication.translate("MainWindow", u"Position ", None))
#if QT_CONFIG(tooltip)
        self.bxPosAnn.setToolTip(QCoreApplication.translate("MainWindow", u"Column header in annotations file ", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.lblIDAnn.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.lblIDAnn.setText(QCoreApplication.translate("MainWindow", u"ID ", None))
#if QT_CONFIG(tooltip)
        self.bxIDAnn.setToolTip(QCoreApplication.translate("MainWindow", u"Column header in annotations file ", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.lblOtherAnn.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.lblOtherAnn.setText(QCoreApplication.translate("MainWindow", u"Other columns", None))
#if QT_CONFIG(tooltip)
        self.listOtherAnn.setToolTip(QCoreApplication.translate("MainWindow", u"Click to select additional columns from annotation", None))
#endif // QT_CONFIG(tooltip)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabAnnotations), QCoreApplication.translate("MainWindow", u"Annotations", None))
        self.btnPrev3.setText(QCoreApplication.translate("MainWindow", u"Previous", None))
        self.groupBoxPval.setTitle(QCoreApplication.translate("MainWindow", u"P-Value Options", None))
#if QT_CONFIG(tooltip)
        self.lblSuggestThresh.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.lblSuggestThresh.setText(QCoreApplication.translate("MainWindow", u"Suggestive Threshold", None))
        self.lineSuggestThresh.setText(QCoreApplication.translate("MainWindow", u"1E-5", None))
#if QT_CONFIG(tooltip)
        self.lblSigThresh.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.lblSigThresh.setText(QCoreApplication.translate("MainWindow", u"Significant Threshold", None))
        self.lineSigThresh.setText(QCoreApplication.translate("MainWindow", u"5E-8", None))
#if QT_CONFIG(tooltip)
        self.lblMaxLogP.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.lblMaxLogP.setText(QCoreApplication.translate("MainWindow", u"Max Log P (axis limit)", None))
        self.groupBoxSignals.setTitle(QCoreApplication.translate("MainWindow", u"Signal Choice Options", None))
#if QT_CONFIG(tooltip)
        self.lblLDBlockWidth.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.lblLDBlockWidth.setText(QCoreApplication.translate("MainWindow", u"LD Block Width", None))
        self.lineLDLBlockWidth.setText(QCoreApplication.translate("MainWindow", u"4E5", None))
#if QT_CONFIG(tooltip)
        self.lblMergeGenes.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.lblMergeGenes.setText(QCoreApplication.translate("MainWindow", u"Merge Genes", None))
        self.bxMergeGenes.setItemText(0, QCoreApplication.translate("MainWindow", u"True", None))
        self.bxMergeGenes.setItemText(1, QCoreApplication.translate("MainWindow", u"False", None))

#if QT_CONFIG(tooltip)
        self.lblBoostKnown.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.lblBoostKnown.setText(QCoreApplication.translate("MainWindow", u"Boost Known Signals", None))
        self.bxBoostKnown.setItemText(0, QCoreApplication.translate("MainWindow", u"True", None))
        self.bxBoostKnown.setItemText(1, QCoreApplication.translate("MainWindow", u"False", None))

        self.bxBoostKnown.setCurrentText(QCoreApplication.translate("MainWindow", u"True", None))
        self.groupBoxPlotOptions.setTitle(QCoreApplication.translate("MainWindow", u"Plot Options", None))
#if QT_CONFIG(tooltip)
        self.lineDataTitle.setToolTip(QCoreApplication.translate("MainWindow", u"Plot title", None))
#endif // QT_CONFIG(tooltip)
        self.chkBoxWithTitle.setText(QCoreApplication.translate("MainWindow", u"With Title", None))
        self.labelTitle.setText(QCoreApplication.translate("MainWindow", u"Title", None))
        self.chkBoxSignalsOnly.setText(QCoreApplication.translate("MainWindow", u"Signals Only", None))
        self.chkBoxWithTable.setText(QCoreApplication.translate("MainWindow", u"With Table", None))
        self.chkBoxInvert.setText(QCoreApplication.translate("MainWindow", u"Invert", None))
        self.chkBoxVertical.setText(QCoreApplication.translate("MainWindow", u"Vertical", None))
        self.btnPlot.setText(QCoreApplication.translate("MainWindow", u"GENERATE PLOT", None))
        self.btnSave.setText(QCoreApplication.translate("MainWindow", u"SAVE PLOT", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabPlot), QCoreApplication.translate("MainWindow", u"Plot", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuAbout.setTitle(QCoreApplication.translate("MainWindow", u"About", None))
    # retranslateUi

