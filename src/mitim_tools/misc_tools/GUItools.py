"""
Notebook tabs, originally from F. Sciortino (MIT, 2019) but modified
extensively by PRF.
"""

import sys
from IPython import embed
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.misc_tools.IOtools import printMsg as print

try:
    # ----------- PyQt -----------
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import (
        NavigationToolbar2QT as NavigationToolbar,
    )
    from PyQt6 import QtWidgets, QtCore
    from PyQt6.QtWidgets import QTabWidget, QTabBar

    # -----------------------------
except:
    print(
        " > PyQt6 module or backends could not be loaded by MITIM, notebooks will not work but I let you continue",
        typeMsg="w",
    )

    class QTabWidget:
        pass

    class QTabBar:
        pass


import matplotlib.pyplot as plt
from mitim_tools.misc_tools.CONFIGread import read_dpi

plt.rcParams["figure.max_open_warning"] = False

dpi_notebook = read_dpi()


class FigureNotebook:
    def __init__(
        self, dummy, windowtitle, parent=None, geometry="1800x900", vertical=True
    ):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)
        self.app.setStyle("Fusion")
        self.MainWindow = QtWidgets.QMainWindow()
        self.MainWindow.__init__()
        self.windowtitle = windowtitle
        self.MainWindow.setWindowTitle(self.windowtitle)
        self.canvases = []
        self.figure_handles = []
        self.toolbar_handles = []
        self.tab_handles = []
        self.current_window = -1

        self.tabs = TabWidget(
            vertical=vertical, xextend=int(geometry.split("x")[0]) - 200
        )
        self.MainWindow.setCentralWidget(self.tabs)
        self.MainWindow.resize(int(geometry.split("x")[0]), int(geometry.split("x")[1]))
        self.MainWindow.show()

    def add_figure(self, label=""):
        figure = plt.figure(dpi=dpi_notebook)
        self.addPlot(label, figure)

        return figure

    def subplots(self, ncols=1, nrows=1, sharey=False, sharex=False, label=""):
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, sharey=sharey, sharex=sharex)

        self.addPlot(label, fig)

        return fig, ax

    def addPlot(self, title, figure):
        new_tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        new_tab.setLayout(layout)

        figure.subplots_adjust(wspace=0.2, hspace=0.2)
        new_canvas = FigureCanvas(figure)
        new_toolbar = NavigationToolbar(new_canvas, new_tab)

        layout.addWidget(new_canvas)
        layout.addWidget(new_toolbar)

        # Tabs ~~~~~~~~~
        self.tabs.insertTab(-1, new_tab, title)
        # ~~~~~~~~~~~~~~

        self.toolbar_handles.append(new_toolbar)
        self.canvases.append(new_canvas)
        self.figure_handles.append(figure)
        self.tab_handles.append(new_tab)

    def show(self):
        print(f"\n> MITIM Notebook open, titled: {self.windowtitle}", typeMsg="i")
        print("\t- Close the notebook to continue")
        self.app.exec()

    def save(self, folder):
        print(f"- Saving Notebook to {folder}/")
        IOtools.askNewFolder(folder)

        for i, fig in enumerate(self.figure_handles):
            print(f"\t- Saving figure #{i+1}/{len(self.figure_handles)}")
            GRAPHICStools.output_figure_papers(f"{folder}/figure{i+1}", fig=fig)


class TabWidget(QTabWidget):
    def __init__(self, vertical=False, xextend=1600, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setTabBar(TabBar(self, vertical=vertical, xextend=xextend))


class TabBar(QTabBar):
    def __init__(self, parent=None, vertical=False, xextend=1600):
        super().__init__(parent)

        self.vertical = vertical

        if self.vertical:
            self.setFixedSize(xextend, 150)
        else:
            self.setFixedSize(xextend, 30)

        self.setStyleSheet(
            """
                    QTabBar::tab { 
                        font-size:           9pt;
                        }
                    QTabBar::tab:selected {
                        background:          #90EE90;
                        color:               #191970;
                        font:                bold;
                        }
                    QTabBar::tab:hover {
                        background: #ADD8E6;
                        color: #000000;
                        }
                            """
        )

    def tabSizeHint(self, i):
        if self.vertical:
            tw = int(self.width() / (self.count()))
            return QtCore.QSize(tw, self.height())

        else:
            return super().tabSizeHint(i)

    def paintEvent(self, event):
        if self.vertical:
            painter = QtWidgets.QStylePainter(self)
            opt = QtWidgets.QStyleOptionTab()

            for i in range(self.count()):
                self.initStyleOption(opt, i)
                painter.drawControl(
                    QtWidgets.QStyle.ControlElement.CE_TabBarTabShape, opt
                )
                painter.save()

                s = opt.rect.size()
                s.transpose()
                r = QtCore.QRect(QtCore.QPoint(), s)
                r.moveCenter(opt.rect.center())
                opt.rect = r

                c = self.tabRect(i).center()
                painter.translate(c)
                painter.rotate(-90)
                painter.translate(-c)
                painter.drawControl(
                    QtWidgets.QStyle.ControlElement.CE_TabBarTabLabel, opt
                )
                painter.restore()
        else:
            super().paintEvent(event)
