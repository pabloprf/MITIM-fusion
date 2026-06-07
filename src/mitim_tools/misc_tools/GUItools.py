"""
Notebook tabs, originally from F. Sciortino (MIT, 2019) but modified
extensively by PRF with the help of ChatGPT to add headless support
and figure saving capabilities.
"""

import sys
import re
import os
import matplotlib
from pathlib import Path
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.misc_tools.LOGtools import printMsg as print

# -----------------------------------------------------------------------------
# Matplotlib backend selection
# -----------------------------------------------------------------------------
# On headless Linux nodes, Matplotlib may default to a Qt backend. Creating a
# figure can then trigger Qt initialization and hard-abort with the xcb plugin
# error. Force a non-GUI backend early (before importing pyplot) when headless.
_MITIM_HEADLESS = (
    (sys.platform.startswith("linux"))
    and (os.environ.get("DISPLAY") is None)
    and (os.environ.get("WAYLAND_DISPLAY") is None)
) or (str(os.environ.get("MITIM_HEADLESS", "0")) == "1")

if _MITIM_HEADLESS and (os.environ.get("MPLBACKEND") is None):
    matplotlib.use("Agg")

# If running headless, do not import Qt or Matplotlib Qt backends at module import
# time. Even importing these modules can cause backend selection or Qt loading
# that later crashes when no platform plugin/display is available.
_MITIM_ENABLE_QT = not _MITIM_HEADLESS

if _MITIM_ENABLE_QT:
    try:
        # ----------- PyQt -----------
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
        from PyQt6 import QtWidgets, QtCore, QtGui
        from PyQt6.QtWidgets import QTabWidget, QTabBar
        # -----------------------------
    except ImportError:
        print(" > PyQt6 module or backends could not be loaded by MITIM, plotting notebooks will not work but I let you continue",typeMsg="w")
        _MITIM_ENABLE_QT = False

if not _MITIM_ENABLE_QT:
    class QTabWidget:
        pass
    class QTabBar:
        pass

import matplotlib.pyplot as plt
from mitim_tools.misc_tools.CONFIGread import read_dpi
from IPython import embed

plt.rcParams["figure.max_open_warning"] = False

class FigureNotebook:
    def __init__(
        self,
        windowtitle,
        parent=None,
        geometry="1800x900",
        vertical=True,
        show=True,
        headless="auto",
    ):
        plt.ioff()

        # Headless environments (e.g. HPC nodes) may have PyQt installed but no display.
        # Creating a QApplication there can hard-abort with an xcb plugin error.
        if not show:
            print(" > Running in headless mode because I am not showing figures anyway")
            headless = True
            matplotlib.use("Agg")
        if headless == "auto":
            headless = _MITIM_HEADLESS

        self._headless = bool(headless)
        self.windowtitle = windowtitle
        self.geometry = geometry

        try:
            self._geometry_px = (
                int(str(geometry).split("x")[0]),
                int(str(geometry).split("x")[1]),
            )
        except Exception:
            self._geometry_px = None

        self.canvases = []
        self.figure_handles = []
        self.toolbar_handles = []
        self.tab_handles = []
        self.tab_titles = []
        self.current_window = -1

        # Headless: do not touch Qt at all.
        if self._headless:
            self.app = None
            self.MainWindow = None
            self.tabs = None
            return

        try:
            self.app = QtWidgets.QApplication.instance()
        except NameError:
            raise Exception(
                "[MITIM] MITIM was installed without [pyqt] option, no GUI available"
            )
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)
        self.app.setStyle("Fusion")
        self.MainWindow = QtWidgets.QMainWindow()
        self.MainWindow.__init__()
        self.MainWindow.setWindowTitle(self.windowtitle)

        self.tabs = TabWidget(
            vertical=vertical, xextend=int(geometry.split("x")[0]) - 200
        )
        self.MainWindow.setCentralWidget(self.tabs)
        self.MainWindow.resize(int(geometry.split("x")[0]), int(geometry.split("x")[1]))

        if show:
            self.MainWindow.show()
        else:
            # Keep the window hidden but still allow saving via fig.savefig
            self.MainWindow.hide()

    def _offscreen_show_begin(self):
        """Show the Qt window off-screen so layouts compute real sizes."""

        try:
            if self._headless:
                return False
            if self.app is None:
                return False

            was_visible = self.MainWindow.isVisible()
            if not was_visible:
                try:
                    self.MainWindow.setAttribute(
                        QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True
                    )
                except Exception:
                    pass

                if self._geometry_px is not None:
                    self.MainWindow.resize(*self._geometry_px)

                self.MainWindow.show()
                self.app.processEvents()

            return was_visible
        except Exception:
            return False

    def _offscreen_show_end(self, was_visible: bool):
        """Revert the temporary off-screen show state."""

        try:
            if self._headless:
                return
            if self.app is None:
                return

            if not was_visible:
                self.MainWindow.hide()
                self.app.processEvents()
        except Exception:
            pass

    def add_figure(self, label="", tab_color=None):
        figure = plt.figure(dpi=read_dpi())
        self.addPlot(label, figure, tab_color=tab_color)

        return figure

    def subplots(self, ncols=1, nrows=1, sharey=False, sharex=False, label=""):
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, sharey=sharey, sharex=sharex)

        self.addPlot(label, fig)

        return fig, ax

    def addPlot(self, title, figure, tab_color=None, tab_alpha=0.55):
        """
        tab_color can be a color name or an integer to grab colors in order
        """

        if self._headless:
            self.figure_handles.append(figure)
            self.tab_titles.append(title)
            return

        new_tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        new_tab.setLayout(layout)
        try:
            layout.setContentsMargins(0, 0, 0, 0)
        except Exception:
            pass

        figure.subplots_adjust(wspace=0.2, hspace=0.2)
        new_canvas = FigureCanvas(figure)
        new_toolbar = NavigationToolbar(new_canvas, new_tab)

        layout.addWidget(new_canvas)
        layout.addWidget(new_toolbar)
        try:
            layout.setStretch(0, 1)
            layout.setStretch(1, 0)
        except Exception:
            pass

        # Tabs ~~~~~~~~~
        self.tabs.insertTab(-1, new_tab, title)
        # ~~~~~~~~~~~~~~

        self.toolbar_handles.append(new_toolbar)
        self.canvases.append(new_canvas)
        self.figure_handles.append(figure)
        self.tab_handles.append(new_tab)
        self.tab_titles.append(title)

        # Set the color for the tab if specified
        tab_color_hex = GRAPHICStools.convert_to_hex_soft(tab_color)
        if tab_color_hex:
            tab_color_hex = QtGui.QColor(tab_color_hex)
            tab_color_hex.setAlphaF(tab_alpha)
            self.tabs.tabBar().setTabColor(self.tabs.count() - 1, tab_color_hex)

    def _move_tab(self, from_idx, to_idx):
        '''
        Move one tab inside the QTabWidget and keep every parallel tracking
        list aligned (canvases, figure_handles, toolbar_handles, tab_handles,
        tab_titles). Per-tab colours are keyed by tab_handle identity during
        the move so they stay attached to the right tab after Qt reshuffles.
        No-op in headless mode (no Qt widgets to reorder).
        '''
        if self._headless or from_idx == to_idx:
            return

        tab_bar = self.tabs.tabBar()

        # Snapshot colours by tab-handle identity so the rebuild below
        # re-attaches each colour to the original tab regardless of which
        # index it ends up at.
        color_by_handle = {}
        for i, handle in enumerate(self.tab_handles):
            if i in tab_bar.tab_colors:
                color_by_handle[id(handle)] = tab_bar.tab_colors[i]

        # Reorder the parallel lists the same way Qt is about to reorder
        # its internal tab model (pop from from_idx, insert at to_idx).
        for lst in (self.canvases, self.figure_handles, self.toolbar_handles,
                    self.tab_handles, self.tab_titles):
            lst.insert(to_idx, lst.pop(from_idx))

        self.tabs.tabBar().moveTab(from_idx, to_idx)

        # Rebuild tab_colors keyed by the new indices.
        tab_bar.tab_colors = {}
        for i, handle in enumerate(self.tab_handles):
            c = color_by_handle.get(id(handle))
            if c is not None:
                tab_bar.tab_colors[i] = c
        tab_bar.update()

    def move_tabs_block_to_front(self, block_start, block_count):
        '''
        Move a consecutive block of tabs `[block_start, block_start+block_count)`
        to the front of the notebook, preserving the block's internal order.
        Useful when two plotting passes build the notebook in one order but
        the desired visual order is the reverse (e.g. the OPT generic pass
        runs first for state-hygiene reasons but should appear after the
        module-specific block on screen).
        '''
        if block_count <= 0:
            return
        # Because moveTab(src, dst) only shifts indices in [min, max]
        # between src and dst, tabs at src+1, src+2, ... still sit at
        # those same indices after each move — so we can do the moves in
        # ascending k with a static from-formula.
        for k in range(block_count):
            self._move_tab(block_start + k, k)

    def show(self):
        if self._headless:
            print(
                "\n> MITIM FigureNotebook running headless (no Qt display).", typeMsg="w"
            )
            return
        print(f"\n> MITIM Notebook open, titled: {self.windowtitle}", typeMsg="i")
        print("\t- Close the notebook to continue")
        self.app.exec()

    def tight_layout(self, *args, realize_layout: bool = True, **kwargs):
        """Apply `fig.tight_layout()` to all figures in this notebook.

        Notes:
            - In the Qt notebook, tight layout depends on a renderer + realized widget sizes.
              By default this method temporarily realizes the window off-screen and iterates
              through tabs so each figure gets a correct layout.
        """

        was_visible = False
        if realize_layout and (not self._headless):
            was_visible = self._offscreen_show_begin()

        try:
            for i, fig in enumerate(self.figure_handles):
                if realize_layout and (not self._headless):
                    try:
                        self.tabs.setCurrentIndex(i)
                        self.app.processEvents()
                    except Exception:
                        pass

                # Draw once so text extents exist before tight_layout.
                try:
                    if getattr(fig, "canvas", None) is not None:
                        fig.canvas.draw()
                except Exception:
                    pass

                try:
                    fig.tight_layout(*args, **kwargs)
                except Exception:
                    pass

                # Redraw to apply new layout.
                try:
                    if (not self._headless) and (i < len(self.canvases)):
                        self.canvases[i].draw()
                    elif getattr(fig, "canvas", None) is not None:
                        fig.canvas.draw()
                except Exception:
                    pass
        finally:
            if realize_layout and (not self._headless):
                self._offscreen_show_end(was_visible)

    @staticmethod
    def _sanitize_filename(name: str, max_len: int = 120) -> str:
        name = (name or "").strip()
        if not name:
            return ""
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"[^A-Za-z0-9._-]+", "", name)
        name = name.strip("._-")
        if len(name) > max_len:
            name = name[:max_len].rstrip("._-")
        return name

    def save(
        self,
        folder,
        fmt: str = "png",
        dpi=None,
        prefix: str = "figure",
        include_index: bool = True,
        use_tab_titles: bool = True,
        overwrite: bool = True,
        create_folder: bool = True,
        force_clean_folder: bool = False,
        bbox_inches="tight",
        pad_inches: float = 0.05,
        transparent: bool = False,
        match_canvas_size: bool = True,
        realize_layout: bool = True,
        apply_tight_layout: bool = False,
        **kwargs,
    ):
        """
        Save each tab's Matplotlib figure to disk using `Figure.savefig`.

        Notes:
            - Does not require calling `FigureNotebook.show()` (i.e. no Qt event loop).
            - If the notebook was created with `show=False`, the window stays hidden.
                        - If `match_canvas_size=True`, figures are saved at the same pixel size as
                            their Qt canvas, which typically matches what you see on screen.

        Args:
            folder: Output directory.
            fmt: File extension/format (e.g. "png", "eps", "pdf").
            dpi: Passed to `savefig` (defaults to Matplotlib's figure dpi).
            prefix: Base filename prefix.
            include_index: Prefix filenames with 1-based index.
            use_tab_titles: Append a sanitized tab title to the filename.
            overwrite: If False, raises if a target file already exists.
            create_folder: If True, creates folder if missing.
            force_clean_folder: If True, deletes/recreates folder (MITIM prompt-free).
            bbox_inches/pad_inches/transparent/kwargs: forwarded to `savefig`.
            match_canvas_size: Resize the Matplotlib figure to match the Qt canvas size.
            realize_layout: Force an off-screen Qt layout pass before saving.
            apply_tight_layout: Call `fig.tight_layout()` before saving (optional).

        Returns:
            List of pathlib.Path objects for the saved files.
        """

        out_dir = Path(folder).expanduser()

        print(f"- Saving Notebook to {folder}/")

        was_visible = False
        if realize_layout and (not self._headless):
            was_visible = self._offscreen_show_begin()

        if force_clean_folder:
            IOtools.askNewFolder(out_dir, force=True)
        else:
            if not out_dir.exists():
                if create_folder:
                    out_dir.mkdir(parents=True, exist_ok=True)
                else:
                    raise FileNotFoundError(f"Output folder does not exist: {out_dir}")

        fmt = (fmt or "png").lstrip(".")
        saved = []

        try:
            for i, fig in enumerate(self.figure_handles):
                print(f"\t- Saving figure {i+1}/{len(self.figure_handles)}...")
                if realize_layout and (not self._headless):
                    try:
                        self.tabs.setCurrentIndex(i)
                        self.app.processEvents()
                    except Exception:
                        pass

                title = self.tab_titles[i] if i < len(self.tab_titles) else ""
                title_part = self._sanitize_filename(title) if use_tab_titles else ""

                stem_parts = []
                if include_index:
                    stem_parts.append(f"{i+1:02d}")
                if prefix:
                    stem_parts.append(self._sanitize_filename(prefix) or prefix)
                if title_part:
                    stem_parts.append(title_part)

                stem = "_".join([p for p in stem_parts if p]) or f"figure_{i+1:02d}"
                fpath = out_dir / f"{stem}.{fmt}"

                if (not overwrite) and fpath.exists():
                    raise FileExistsError(f"File already exists: {fpath}")

                # Make saved output match the on-screen tab size.
                if match_canvas_size:
                    try:
                        w_px, h_px = 0, 0
                        if (not self._headless) and (i < len(self.canvases)):
                            canvas = self.canvases[i]
                            size = canvas.size()  # QSize in pixels
                            w_px, h_px = int(size.width()), int(size.height())

                        # If Qt still reports a tiny size, fall back to the notebook geometry.
                        if ((w_px <= 50) or (h_px <= 50)) and (self._geometry_px is not None):
                            w_px, h_px = self._geometry_px

                        if (w_px > 0) and (h_px > 0):
                            # Match the on-screen canvas size at the figure's OWN dpi:
                            # savefig(dpi=dpi) then scales the saved resolution
                            # (pixels = canvas_px * dpi / fig_dpi). Sizing with the
                            # requested dpi here cancelled it out — every --dpi value
                            # produced identical pixel dimensions, only font scale changed.
                            dpi_eff = fig.get_dpi()

                            # Go a bit larger to avoid overlapping of labels, etc
                            w_px = int(w_px * 1.25)
                            h_px = int(h_px * 1.25)
                            fig.set_size_inches(
                                w_px / dpi_eff, h_px / dpi_eff, forward=True
                            )
                    except Exception:
                        pass

                if apply_tight_layout:
                    try:
                        fig.tight_layout()
                    except Exception:
                        pass

                # If a canvas exists, try drawing once so layouts are applied.
                try:
                    if getattr(fig, "canvas", None) is not None:
                        fig.canvas.draw()
                except Exception:
                    pass

                if fpath.exists():
                    fpath.unlink() 
                
                fig.savefig(
                    fpath,
                    format=fmt,
                    dpi=dpi,
                    bbox_inches=bbox_inches,
                    pad_inches=pad_inches,
                    transparent=transparent,
                    **kwargs,
                )
                saved.append(fpath)

            print(f"- Saved {len(saved)} figure(s) to {out_dir}")
            return saved
        finally:
            if realize_layout and (not self._headless):
                self._offscreen_show_end(was_visible)

    def close(self):
        """
        Properly closes the FigureNotebook and its associated resources.
        """
        print(f"\n> Closing MITIM Notebook titled: {self.windowtitle}", typeMsg="i")
        # Disconnect all canvases
        # for canvas in self.canvases:
        #     canvas.mpl_disconnect(canvas.callbacks.connect('draw_event', lambda: None))
        if self._headless:
            return
        self.MainWindow.close()
        self.app.quit()

class TabWidget(QTabWidget):
    def __init__(self, vertical=False, xextend=1600, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setTabBar(TabBar(self, vertical=vertical, xextend=xextend))


class TabBar(QTabBar):
    # Tab font. Constant: we rely on scroll buttons for dense notebooks
    # rather than shrinking text per tab count.
    _FONT_PT = 9

    def __init__(self, parent=None, vertical=False, xextend=1600):
        super().__init__(parent)

        self.vertical = vertical
        self.tab_colors = {}

        if self.vertical:
            self.setFixedSize(xextend, 170)
        else:
            self.setFixedSize(xextend, 30)
        # Scroll buttons apply to both layouts. Elide mode is horizontal
        # only — in the vertical path labels are painted rotated by our
        # own paintEvent and Qt's elision uses the un-rotated tab width,
        # which would drop every label to empty.
        self.setUsesScrollButtons(True)
        if not self.vertical:
            self.setElideMode(QtCore.Qt.TextElideMode.ElideRight)

        self.setStyleSheet(
            f"""
                    QTabBar::tab {{
                        font-size:           {self._FONT_PT}pt;
                        }}
                    QTabBar::tab:selected {{
                        background:          #00FF00;
                        color:               #191970;
                        font:                bold;
                        }}
                    QTabBar::tab:hover {{
                        background:          #90EE90;
                        color:               #191970;
                        }}
                            """
        )

    def setTabColor(self, index, color):
        self.tab_colors[index] = color
        self.update()

    # Constant per-tab width. Below this, Qt squeezes tabs into slivers
    # too narrow to render labels; at this floor, Qt's overflow logic
    # takes over and shows scroll arrows.
    #   horizontal mode: labels run left-to-right, 100px is comfortable.
    #   vertical mode:   labels painted rotated -90 in a 170px-tall tab,
    #                    15px is enough to fit an 11pt glyph column
    #                    with minimal padding.
    _MIN_TAB_WIDTH_PX = 100
    _MIN_TAB_WIDTH_PX_VERTICAL = 22

    def tabSizeHint(self, i):
        if self.vertical:
            # Floor at _MIN_TAB_WIDTH_PX_VERTICAL so dense notebooks push
            # total tab-strip width past xextend and Qt's scroll arrows
            # kick in instead of collapsing every tab to a few pixels.
            natural = int(self.width() / max(1, self.count())) if self.count() else self._MIN_TAB_WIDTH_PX_VERTICAL
            tw = max(self._MIN_TAB_WIDTH_PX_VERTICAL, natural)
            return QtCore.QSize(tw, self.height())

        else:
            # Respect the style's natural hint for short labels but floor
            # the width so dense notebooks tip the total tab-row width
            # past xextend, which is what triggers the scroll buttons we
            # enabled in __init__.
            hint = super().tabSizeHint(i)
            return QtCore.QSize(max(hint.width(), self._MIN_TAB_WIDTH_PX), hint.height())

    def minimumTabSizeHint(self, i):
        '''Qt uses minimumTabSizeHint() during layout to decide whether tabs
        can shrink below tabSizeHint(). We floor it at the same minimum so
        Qt can't bypass our tabSizeHint floor.'''
        if self.vertical:
            return QtCore.QSize(self._MIN_TAB_WIDTH_PX_VERTICAL, self.height())
        hint = super().minimumTabSizeHint(i)
        return QtCore.QSize(max(hint.width(), self._MIN_TAB_WIDTH_PX), hint.height())

    def paintEvent(self, event):
        if self.vertical:
            painter = QtWidgets.QStylePainter(self)
            opt = QtWidgets.QStyleOptionTab()

            for i in range(self.count()):
                self.initStyleOption(opt, i)
                if i in self.tab_colors:
                    opt.palette.setColor(
                        QtGui.QPalette.ColorRole.Button,
                        QtGui.QColor(self.tab_colors[i]),
                    )
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
