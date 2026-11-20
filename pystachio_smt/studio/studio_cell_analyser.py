# -*- coding: utf-8 -*-
import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, interpolate
from scipy.stats import gaussian_kde
import scipy.io as sio

try:
    import tifffile
except ImportError:
    tifffile = None

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QMessageBox, QFileDialog, QProgressDialog, QApplication,
    QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt


class NoScrollSpinBox(QSpinBox):
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoScrollComboBox(QComboBox):
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=9, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, layout='constrained')
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


class CellAnalyserTab(QWidget):
    """
    Results Viewer sub-tab that scans preprocessing results folders (`cell_fitting_parameters*.csv`
    and `cell_*` subdirectories) and generates individual views selected via dropdown:
      1. Heatmap: Frame-Average Intensity (L, R, or Both Channels side-by-side)
      2. Heatmap: Diffusion Coefficient D (L, R, or Both Channels side-by-side)
      3. Heatmap: Jump Distance r (L, R, or Both Channels side-by-side)
      4. Line Profiles (Longitudinal / Transverse; L, R, or Both Channels side-by-side)
      5. Cell Features: Histograms of Length, Width, Orientation Angle, Area, and Cell Fitting Error
    """

    VIEW_CHOICES = [
        ("Heatmap: Frame-Avg Intensity", "intensity"),
        ("Heatmap: Diffusion Coeff (D)", "diffusion"),
        ("Heatmap: Jump Distance (r)", "jump"),
        ("Line Profiles (Longitudinal / Transverse)", "profiles"),
        ("Cell Features (Length, Width, Angle, Area & Fit Error)", "morphology"),
    ]

    CHANNEL_CHOICES = [
        "Left Channel (L)",
        "Right Channel (R)",
        "Both Channels (L & R Side-by-Side)",
    ]

    MASK_SOURCE_CHOICES = [
        "Auto (fits_stack → segmentation)",
        "Fitted Capsule Stack (fits_stack)",
        "Segmentation Mask (all_cells_mask)",
    ]

    REGEN_BTN_CLEAN_STYLE = "font-size: 10px; padding: 2px 7px;"
    REGEN_BTN_STALE_STYLE = (
        "font-size: 10px; padding: 2px 7px; background-color: #fff3cd; "
        "border: 1px solid #d39e00; font-weight: bold; color: #664d03;"
    )

    def __init__(self, parent_viewer=None):
        super().__init__(parent_viewer)
        self.parent = parent_viewer

        # Cached parsed cell records and computed plot matrices
        self.cell_records = []
        self._has_generated = False
        self._cached_results = {}
        self._stale_flags = {
            'records': False,
            'intensity': False,
            'diffusion': False,
            'jump': False,
            'profiles': False,
            'morphology': False,
        }

        self._init_main_canvas()
        self._init_options_group()

        if self.parent is not None:
            self.populate_fields()

    # =========================================================================
    # UI INITIALIZATION
    # =========================================================================

    def _init_main_canvas(self):
        layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        lbl_view = QLabel("Plot View:")
        lbl_view.setStyleSheet("font-weight: bold;")

        self.combo_view_mode = NoScrollComboBox()
        self.combo_view_mode.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        for label, code in self.VIEW_CHOICES:
            self.combo_view_mode.addItem(label, userData=code)
        self.combo_view_mode.currentIndexChanged.connect(self._on_view_mode_changed)

        self.lbl_cell_count_banner = QLabel("Detected cells: 0 (Select plot view & click Generate)")
        self.lbl_cell_count_banner.setStyleSheet("font-size: 11px; color: #333; font-weight: bold;")

        top_bar.addWidget(lbl_view)
        top_bar.addWidget(self.combo_view_mode)
        top_bar.addSpacing(12)
        top_bar.addWidget(self.lbl_cell_count_banner, 1)

        self.canvas = MplCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll_area.setWidget(self.canvas)

        layout.addLayout(top_bar)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.scroll_area, 1)

        self._draw_placeholder(
            "Cell Analyser ready (computation paused to save resources).\n"
            "Choose the desired plot from the 'Plot View' dropdown (or tick checkboxes in the sidebar)\n"
            "and click 'Generate Selected Plot(s)' (or an individual '↻ Regen' button)."
        )

    def _make_plot_check_row(self, checkbox: QCheckBox, plot_key: str, tooltip: str):
        """Creates a compact row pairing a plot checkbox with its specific Regenerate button."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        btn = QPushButton("↻ Regen")
        btn.setStyleSheet(self.REGEN_BTN_CLEAN_STYLE)
        btn.setToolTip(tooltip)
        btn.setFixedWidth(68)
        btn.clicked.connect(lambda _checked=False, k=plot_key: self.regenerate_single_plot(k))

        row.addWidget(checkbox, 1)
        row.addWidget(btn, 0)
        return row, btn

    def _init_options_group(self):
        self.options_group = QGroupBox("Cell Analyser Options")
        opt_layout = QFormLayout()
        opt_layout.setVerticalSpacing(6)

        # 0. Linked Field Selector (two-way synced with ResultsViewerTab.field_selector)
        self.combo_field = NoScrollComboBox()
        self.combo_field.setToolTip("Synced with the main Field Selection dropdown")
        self.combo_field.currentIndexChanged.connect(self._on_local_field_changed)
        opt_layout.addRow("Field Selection:", self.combo_field)

        # 1. Plot Selection Checkboxes + Specific Regenerate Buttons (UNCHECKED BY DEFAULT)
        lbl_plots_hdr = QLabel("<b>Plots to Pre-Compute (Active Dropdown Always Included):</b>")
        opt_layout.addRow(lbl_plots_hdr)

        self.chk_plot_heatmap_intensity = QCheckBox("Heatmap: Frame-Avg Intensity")
        self.chk_plot_heatmap_intensity.setChecked(False)
        row_int, self.btn_regen_intensity = self._make_plot_check_row(
            self.chk_plot_heatmap_intensity,
            'intensity',
            "Regenerate only the Frame-Average Intensity heatmap(s) using current filters"
        )

        self.chk_plot_heatmap_diff = QCheckBox("Heatmap: Diffusion Coeff (D)")
        self.chk_plot_heatmap_diff.setChecked(False)
        row_diff, self.btn_regen_diff = self._make_plot_check_row(
            self.chk_plot_heatmap_diff,
            'diffusion',
            "Regenerate only the Diffusion Coefficient (D) heatmap(s) using current filters"
        )

        self.chk_plot_heatmap_jump = QCheckBox("Heatmap: Jump Distance (r)")
        self.chk_plot_heatmap_jump.setChecked(False)
        row_jump, self.btn_regen_jump = self._make_plot_check_row(
            self.chk_plot_heatmap_jump,
            'jump',
            "Regenerate only the Jump Distance (r) heatmap(s) using current filters"
        )

        self.chk_plot_line_profiles = QCheckBox("Line Profiles (L / R)")
        self.chk_plot_line_profiles.setChecked(False)
        row_prof, self.btn_regen_profiles = self._make_plot_check_row(
            self.chk_plot_line_profiles,
            'profiles',
            "Regenerate only the Line Profiles using current channel/axis filters"
        )

        self.chk_plot_morphology = QCheckBox("Cell Features (Length, Width, Angle, Area & Error)")
        self.chk_plot_morphology.setChecked(False)
        row_morph, self.btn_regen_morphology = self._make_plot_check_row(
            self.chk_plot_morphology,
            'morphology',
            "Regenerate only the Cell Features histograms (Length, Width, Angle, Area, Fit Error)"
        )

        self._regen_buttons = {
            'intensity': self.btn_regen_intensity,
            'diffusion': self.btn_regen_diff,
            'jump': self.btn_regen_jump,
            'profiles': self.btn_regen_profiles,
            'morphology': self.btn_regen_morphology,
        }

        opt_layout.addRow(row_int)
        opt_layout.addRow(row_diff)
        opt_layout.addRow(row_jump)
        opt_layout.addRow(row_prof)
        opt_layout.addRow(row_morph)

        sel_btn_row = QHBoxLayout()
        sel_btn_row.setSpacing(4)
        self.btn_check_all = QPushButton("Check All")
        self.btn_check_all.setStyleSheet("font-size: 10px; padding: 2px 6px;")
        self.btn_check_all.clicked.connect(lambda: self._set_all_plot_checks(True))

        self.btn_uncheck_all = QPushButton("Uncheck All")
        self.btn_uncheck_all.setStyleSheet("font-size: 10px; padding: 2px 6px;")
        self.btn_uncheck_all.clicked.connect(lambda: self._set_all_plot_checks(False))

        sel_btn_row.addWidget(self.btn_check_all)
        sel_btn_row.addWidget(self.btn_uncheck_all)
        opt_layout.addRow(sel_btn_row)

        self.btn_generate = QPushButton("Generate Selected Plot(s)")
        self.btn_generate.setStyleSheet(
            "font-weight: bold; background-color: #008080; color: white; padding: 6px;"
        )
        self.btn_generate.clicked.connect(self.generate_selected_plots)
        opt_layout.addRow(self.btn_generate)

        # 2. Channel, Mask Source & Coordinate Settings
        self.combo_heatmap_channel = NoScrollComboBox()
        self.combo_heatmap_channel.addItems(self.CHANNEL_CHOICES)
        self.combo_heatmap_channel.setToolTip(
            "Select Left (L), Right (R), or Both Channels (L & R Side-by-Side) in the same figure"
        )
        self.combo_heatmap_channel.currentIndexChanged.connect(self._on_heatmap_channel_changed)
        opt_layout.addRow("Channel Selection:", self.combo_heatmap_channel)

        self.combo_mask_source = NoScrollComboBox()
        self.combo_mask_source.addItems(self.MASK_SOURCE_CHOICES)
        self.combo_mask_source.setToolTip(
            "Select whether cell masks are read from fitted capsule stacks (fits_stack.tif) "
            "or raw segmentation masks (all_cells_mask.tif / single_cell_mask*.tif)"
        )
        self.combo_mask_source.currentIndexChanged.connect(
            lambda _i: self._mark_plots_stale(['records', 'intensity', 'profiles', 'morphology'])
        )
        opt_layout.addRow("Cell Mask Source:", self.combo_mask_source)

        self.chk_use_tmat_for_r = QCheckBox("Align Right Channel via transformation_matrix.npy")
        self.chk_use_tmat_for_r.setChecked(True)
        self.chk_use_tmat_for_r.stateChanged.connect(
            lambda _s: self._mark_plots_stale(['intensity', 'diffusion', 'jump', 'profiles'])
        )
        opt_layout.addRow(self.chk_use_tmat_for_r)

        # 3. Heatmap Pixel-Size Grid & Visual Controls
        self.spin_bin_px_scale = NoScrollDoubleSpinBox()
        self.spin_bin_px_scale.setDecimals(2)
        self.spin_bin_px_scale.setRange(0.25, 4.0)
        self.spin_bin_px_scale.setSingleStep(0.25)
        self.spin_bin_px_scale.setValue(1.0)
        self.spin_bin_px_scale.setSuffix(" × px")
        self.spin_bin_px_scale.setToolTip(
            "Spatial grid step relative to configured pixel size (1.0 × px = exact camera pixel size)"
        )
        self.spin_bin_px_scale.valueChanged.connect(
            lambda _v: self._mark_plots_stale(['intensity', 'diffusion', 'jump'])
        )
        opt_layout.addRow("Heatmap Bin Step:", self.spin_bin_px_scale)

        self.combo_cmap = NoScrollComboBox()
        self.combo_cmap.addItems(["turbo", "jet", "viridis", "inferno", "magma", "hot"])
        self.combo_cmap.currentIndexChanged.connect(self._redraw_from_cache)
        opt_layout.addRow("Heatmap Colormap:", self.combo_cmap)

        self.chk_norm_cell_intensity = QCheckBox("Normalize Intensity per Cell (Mean = 1)")
        self.chk_norm_cell_intensity.setChecked(False)
        self.chk_norm_cell_intensity.stateChanged.connect(
            lambda _s: self._mark_plots_stale(['intensity'])
        )
        opt_layout.addRow(self.chk_norm_cell_intensity)

        self.chk_mask_outside_capsule = QCheckBox("Mask Bins Outside Cell Capsule")
        self.chk_mask_outside_capsule.setChecked(True)
        self.chk_mask_outside_capsule.stateChanged.connect(self._redraw_from_cache)
        opt_layout.addRow(self.chk_mask_outside_capsule)

        self.chk_show_grid = QCheckBox("Show Pixel Grid Lines (Paper Style)")
        self.chk_show_grid.setChecked(True)
        self.chk_show_grid.stateChanged.connect(self._redraw_from_cache)
        opt_layout.addRow(self.chk_show_grid)

        self.chk_half_plots = QCheckBox("Also Generate Half Plots (Fold & Combine Halves)")
        self.chk_half_plots.setChecked(False)
        self.chk_half_plots.setToolTip(
            "Cuts heatmaps and line profiles at the midpoint, flips one half, "
            "and combines (averages) it with the other half below the full plot"
        )
        self.chk_half_plots.stateChanged.connect(self._redraw_from_cache)
        opt_layout.addRow(self.chk_half_plots)

        # Optional D subpopulation band
        self.spin_d_band_min = NoScrollDoubleSpinBox()
        self.spin_d_band_min.setDecimals(3)
        self.spin_d_band_min.setRange(0.0, 100.0)
        self.spin_d_band_min.setSingleStep(0.05)
        self.spin_d_band_min.setValue(0.0)
        self.spin_d_band_min.setSpecialValueText("Off")
        self.spin_d_band_min.setSuffix(" μm²/s")
        self.spin_d_band_min.valueChanged.connect(
            lambda _v: self._mark_plots_stale(['diffusion', 'jump'])
        )

        self.spin_d_band_max = NoScrollDoubleSpinBox()
        self.spin_d_band_max.setDecimals(3)
        self.spin_d_band_max.setRange(0.0, 100.0)
        self.spin_d_band_max.setSingleStep(0.05)
        self.spin_d_band_max.setValue(0.0)
        self.spin_d_band_max.setSpecialValueText("Off")
        self.spin_d_band_max.setSuffix(" μm²/s")
        self.spin_d_band_max.valueChanged.connect(
            lambda _v: self._mark_plots_stale(['diffusion', 'jump'])
        )

        opt_layout.addRow("Heatmap Min D:", self.spin_d_band_min)
        opt_layout.addRow("Heatmap Max D:", self.spin_d_band_max)

        # 4. Line Profile & Cell Morphology Filter Controls
        self.combo_profile_axis = NoScrollComboBox()
        self.combo_profile_axis.addItems([
            "Longitudinal (Along Cell Length)",
            "Transverse (Across Cell Width)",
        ])
        self.combo_profile_axis.currentIndexChanged.connect(
            lambda _i: self._mark_plots_stale(['profiles'])
        )
        opt_layout.addRow("Line Profile Axis:", self.combo_profile_axis)

        self.combo_profile_norm = NoScrollComboBox()
        self.combo_profile_norm.addItems([
            "Raw Intensity (a.u.)",
            "Mean-Normalized (I / <I>)",
            "Min-Max Normalized [0–1]",
        ])
        self.combo_profile_norm.currentIndexChanged.connect(self._redraw_from_cache)
        opt_layout.addRow("Profile Scaling:", self.combo_profile_norm)

        self.spin_min_cell_len = NoScrollDoubleSpinBox()
        self.spin_min_cell_len.setDecimals(2)
        self.spin_min_cell_len.setRange(0.0, 50.0)
        self.spin_min_cell_len.setSingleStep(0.25)
        self.spin_min_cell_len.setValue(0.0)
        self.spin_min_cell_len.setSpecialValueText("Off")
        self.spin_min_cell_len.setSuffix(" μm")
        self.spin_min_cell_len.valueChanged.connect(
            lambda _v: self._mark_plots_stale(['records', 'intensity', 'diffusion', 'jump', 'profiles', 'morphology'])
        )

        self.spin_max_cell_len = NoScrollDoubleSpinBox()
        self.spin_max_cell_len.setDecimals(2)
        self.spin_max_cell_len.setRange(0.0, 50.0)
        self.spin_max_cell_len.setSingleStep(0.25)
        self.spin_max_cell_len.setValue(0.0)
        self.spin_max_cell_len.setSpecialValueText("Off")
        self.spin_max_cell_len.setSuffix(" μm")
        self.spin_max_cell_len.valueChanged.connect(
            lambda _v: self._mark_plots_stale(['records', 'intensity', 'diffusion', 'jump', 'profiles', 'morphology'])
        )

        opt_layout.addRow("Min Cell Length:", self.spin_min_cell_len)
        opt_layout.addRow("Max Cell Length:", self.spin_max_cell_len)

        self.btn_export_cells = QPushButton("Export Cell Summary (.csv)")
        self.btn_export_cells.clicked.connect(self.export_cell_summary_csv)
        opt_layout.addRow(self.btn_export_cells)

        self.lbl_analyser_stats = QLabel("No cell analysis computed yet.")
        self.lbl_analyser_stats.setWordWrap(True)
        self.lbl_analyser_stats.setStyleSheet(
            "font-size: 10px; background-color: #f8f9fa; padding: 6px; "
            "border: 1px solid #ccc; border-radius: 4px;"
        )
        opt_layout.addRow(self.lbl_analyser_stats)

        for w in self.options_group.findChildren((NoScrollSpinBox, NoScrollDoubleSpinBox, NoScrollComboBox)):
            w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.options_group.setLayout(opt_layout)
        self.options_group.setVisible(False)

    # =========================================================================
    # VIEW MODE, STALE FILTER TRACKING & CHANNEL HELPERS
    # =========================================================================

    def _on_view_mode_changed(self, _index):
        """Switches the active single-plot view when the top Plot View dropdown changes."""
        view_key = self.combo_view_mode.currentData()
        chk_map = {
            'intensity': self.chk_plot_heatmap_intensity,
            'diffusion': self.chk_plot_heatmap_diff,
            'jump': self.chk_plot_heatmap_jump,
            'profiles': self.chk_plot_line_profiles,
            'morphology': self.chk_plot_morphology,
        }
        if view_key in chk_map and not chk_map[view_key].isChecked():
            chk_map[view_key].setChecked(True)

        if self._has_generated and self._cached_results:
            self._redraw_from_cache()

    def _active_channels_list(self):
        """Returns list of channel names to compute/display based on combo_heatmap_channel."""
        sel = self.combo_heatmap_channel.currentText()
        if sel == "Both Channels (L & R Side-by-Side)":
            return ["Left Channel (L)", "Right Channel (R)"]
        if sel == "Right Channel (R)":
            return ["Right Channel (R)"]
        return ["Left Channel (L)"]

    def _mark_plots_stale(self, keys):
        """Marks specific plot keys as needing regeneration after a relevant filter changes."""
        for k in keys:
            if k in self._stale_flags:
                self._stale_flags[k] = True
        self._refresh_regen_button_styles()

    def _clear_stale_flags(self, keys):
        for k in keys:
            if k in self._stale_flags:
                self._stale_flags[k] = False
        self._refresh_regen_button_styles()

    def _refresh_regen_button_styles(self):
        if not hasattr(self, '_regen_buttons'):
            return
        active_view = self.combo_view_mode.currentData()
        plot_enabled_map = {
            'intensity': self.chk_plot_heatmap_intensity.isChecked() or active_view == 'intensity',
            'diffusion': self.chk_plot_heatmap_diff.isChecked() or active_view == 'diffusion',
            'jump': self.chk_plot_heatmap_jump.isChecked() or active_view == 'jump',
            'profiles': self.chk_plot_line_profiles.isChecked() or active_view == 'profiles',
            'morphology': self.chk_plot_morphology.isChecked() or active_view == 'morphology',
        }
        for key, btn in self._regen_buttons.items():
            is_stale = self._has_generated and self._stale_flags.get(key, False) and plot_enabled_map.get(key, False)
            if is_stale:
                btn.setText("↻ Regen*")
                btn.setStyleSheet(self.REGEN_BTN_STALE_STYLE)
            else:
                btn.setText("↻ Regen")
                btn.setStyleSheet(self.REGEN_BTN_CLEAN_STYLE)

    def _on_heatmap_channel_changed(self, _index):
        """Handles switching between Left, Right, or Both Channels (L & R Side-by-Side)."""
        if not self._has_generated or not self._cached_results:
            return

        needed_channels = set(self._active_channels_list())
        cache = self._cached_results
        missing_any = False

        if cache.get('want_int') and not needed_channels.issubset(cache.get('int_grids', {}).keys()):
            self._stale_flags['intensity'] = True
            missing_any = True
        if cache.get('want_diff') and not needed_channels.issubset(cache.get('diff_grids', {}).keys()):
            self._stale_flags['diffusion'] = True
            missing_any = True
        if cache.get('want_jump') and not needed_channels.issubset(cache.get('jump_grids', {}).keys()):
            self._stale_flags['jump'] = True
            missing_any = True
        if cache.get('want_prof'):
            prof_dict = cache.get('prof_dict', {})
            if any(ch not in prof_dict for ch in needed_channels):
                self._stale_flags['profiles'] = True
                missing_any = True

        self._refresh_regen_button_styles()
        self._redraw_from_cache()
        if missing_any:
            self._update_stats_label()

    # =========================================================================
    # TWO-WAY FIELD SELECTOR SYNC & PARENT HOOKS
    # =========================================================================

    def populate_fields(self):
        """Populates the Cell Analyser field dropdown from parent's active sample/fields."""
        self.combo_field.blockSignals(True)
        self.combo_field.clear()

        if self.parent is not None:
            parent_fs = getattr(self.parent, 'field_selector', None)
            if parent_fs is not None and parent_fs.count() > 0:
                for i in range(parent_fs.count()):
                    self.combo_field.addItem(parent_fs.itemText(i))
            else:
                raw_fields = self._discover_all_field_dirs()
                fields = sorted({fld for _, fld, _ in raw_fields if fld})
                if fields:
                    self.combo_field.addItem("-- All Fields --")
                    for fld in fields:
                        self.combo_field.addItem(fld)

        self.combo_field.blockSignals(False)
        self.sync_field_from_parent()

    def sync_field_from_parent(self):
        """Aligns local field dropdown when parent's sidebar field_selector changes."""
        if self.parent is None or self.combo_field.count() == 0:
            self.on_parent_data_updated()
            return

        parent_fs = getattr(self.parent, 'field_selector', None)
        if parent_fs is not None and parent_fs.count() > 0:
            target_text = parent_fs.currentText()
            idx = self.combo_field.findText(target_text)
            if idx >= 0 and self.combo_field.currentIndex() != idx:
                self.combo_field.blockSignals(True)
                self.combo_field.setCurrentIndex(idx)
                self.combo_field.blockSignals(False)

        self.on_parent_data_updated()

    def _on_local_field_changed(self, index):
        """Pushes local field selection back to parent's sidebar field_selector."""
        if index < 0:
            return
        field_text = self.combo_field.currentText()
        if not field_text:
            return

        if self.parent is not None:
            parent_fs = getattr(self.parent, 'field_selector', None)
            if parent_fs is not None:
                idx = parent_fs.findText(field_text)
                if idx >= 0 and parent_fs.currentIndex() != idx:
                    parent_fs.setCurrentIndex(idx)
                    self._mark_plots_stale(['records', 'intensity', 'diffusion', 'jump', 'profiles', 'morphology'])
                    return

        self.on_parent_data_updated()

    def plot(self, summary_df=None):
        """Standard entry point called by ResultsViewerTab._render_tab_widget()."""
        self.on_parent_data_updated()

    def _set_all_plot_checks(self, checked: bool):
        for chk in (
            self.chk_plot_heatmap_intensity,
            self.chk_plot_heatmap_diff,
            self.chk_plot_heatmap_jump,
            self.chk_plot_line_profiles,
            self.chk_plot_morphology,
        ):
            chk.setChecked(checked)
        self._refresh_regen_button_styles()

    def _draw_placeholder(self, text: str):
        fig = self.canvas.fig
        fig.clf()
        fig.set_layout_engine('constrained')
        self.canvas.setMinimumSize(640, 420)
        ax = fig.add_subplot(111)
        ax.text(
            0.5, 0.5, text,
            ha='center', va='center', transform=ax.transAxes,
            fontsize=10, color='#555'
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw()

    # =========================================================================
    # DIRECTORY & CELL_* DISCOVERY (LIGHTWEIGHT METADATA SCAN)
    # =========================================================================

    def _discover_all_field_dirs(self):
        """Discovers all (sample_name, field_name, results_dir) containing `cell_*` folders or fitting CSVs."""
        discovered = []
        seen_dirs = set()

        if self.parent is None:
            return discovered

        field_dirs_map = getattr(self.parent, 'field_dirs_map', {})

        for field_name, info in field_dirs_map.items():
            sample_name = os.path.basename(info.get("sample_dir", "")) if isinstance(info, dict) else ""
            search_dirs = self.parent.get_search_dirs_for_field(field_name)
            for s_dir in search_dirs:
                cell_subdirs = [
                    d for d in glob.glob(os.path.join(s_dir, "cell_*"))
                    if os.path.isdir(d)
                ]
                fit_csvs = glob.glob(os.path.join(s_dir, "*cell_fitting_parameter*.csv"))
                if cell_subdirs or fit_csvs:
                    real_dir = os.path.realpath(s_dir)
                    if real_dir not in seen_dirs:
                        seen_dirs.add(real_dir)
                        discovered.append((sample_name, field_name, s_dir))
                    break

        root_dir = getattr(self.parent, 'root_dir', "")
        if not discovered and root_dir and os.path.exists(root_dir):
            direct_cells = [
                d for d in glob.glob(os.path.join(root_dir, "cell_*"))
                if os.path.isdir(d)
            ]
            direct_csvs = glob.glob(os.path.join(root_dir, "*cell_fitting_parameter*.csv"))
            if direct_cells or direct_csvs:
                discovered.append((
                    os.path.basename(os.path.dirname(root_dir)),
                    os.path.basename(root_dir),
                    root_dir,
                ))
            else:
                for res_dir in glob.glob(os.path.join(root_dir, "**", "results"), recursive=True):
                    if os.path.isdir(res_dir) and (
                        any(os.path.isdir(c) for c in glob.glob(os.path.join(res_dir, "cell_*")))
                        or len(glob.glob(os.path.join(res_dir, "*cell_fitting_parameter*.csv"))) > 0
                    ):
                        real_dir = os.path.realpath(res_dir)
                        if real_dir not in seen_dirs:
                            seen_dirs.add(real_dir)
                            fld_dir = os.path.dirname(res_dir)
                            discovered.append((
                                os.path.basename(os.path.dirname(fld_dir)),
                                os.path.basename(fld_dir),
                                res_dir,
                            ))

        return discovered

    def _get_active_field_dirs(self):
        """Returns list of (sample_name, field_name, results_dir) matching current Sample & Field selection."""
        if self.parent is None:
            return []

        selected_sample = self.parent.sample_selector.currentText() if hasattr(self.parent, 'sample_selector') else ""
        parent_fs = getattr(self.parent, 'field_selector', None)
        if parent_fs is not None and parent_fs.count() > 0:
            selected_field = parent_fs.currentText()
        else:
            selected_field = self.combo_field.currentText() or "-- All Fields --"

        allowed_fields = None
        if selected_field == "-- Selected Fields --":
            allowed_fields = set(self.parent.get_selected_fields())
        elif selected_field and selected_field != "-- All Fields --":
            allowed_fields = {selected_field}

        sample_allowed_fields = None
        traj_df = getattr(self.parent, 'trajectories_df', pd.DataFrame())
        if (
            not traj_df.empty
            and selected_sample
            and selected_sample != "All Samples Combined"
        ):
            sample_allowed_fields = set(
                traj_df[traj_df['sample'] == selected_sample]['field'].unique()
            )

        all_fields = self._discover_all_field_dirs()
        active = []
        for sample_name, field_name, r_dir in all_fields:
            if allowed_fields is not None and field_name not in allowed_fields:
                continue
            if sample_allowed_fields is not None and field_name not in sample_allowed_fields:
                continue
            active.append((sample_name, field_name, r_dir))

        return active

    def on_parent_data_updated(self):
        """Lightweight update when parent dataset/filters change; marks cached plots stale without forced recompute."""
        active_fields = self._get_active_field_dirs()
        total_cell_folders = sum(
            len([d for d in glob.glob(os.path.join(r_dir, "cell_*")) if os.path.isdir(d)])
            for _, _, r_dir in active_fields
        )
        if self._has_generated:
            self._mark_plots_stale(['records', 'intensity', 'diffusion', 'jump', 'profiles', 'morphology'])

        stale_note = " (Filters updated — click '↻ Regen*' or 'Generate Selected Plot(s)')" if self._has_generated else ""
        self.lbl_cell_count_banner.setText(
            f"Detected {total_cell_folders} cell_* folder(s) across {len(active_fields)} field(s){stale_note}"
        )

        # Sync channel selector with parent channel selector when appropriate
        if self.parent is not None and hasattr(self.parent, 'channel_selector') and self.parent.channel_selector.isVisible():
            parent_ch = self.parent.channel_selector.currentText()
            target_ch = None
            if parent_ch in ("Left Channel (L)", "Right Channel (R)"):
                target_ch = parent_ch
            elif "Both" in parent_ch or "All" in parent_ch:
                target_ch = "Both Channels (L & R Side-by-Side)"
            if target_ch:
                idx = self.combo_heatmap_channel.findText(target_ch)
                if idx >= 0 and self.combo_heatmap_channel.currentIndex() != idx:
                    self.combo_heatmap_channel.blockSignals(True)
                    self.combo_heatmap_channel.setCurrentIndex(idx)
                    self.combo_heatmap_channel.blockSignals(False)

        if not self._has_generated:
            self._draw_placeholder(
                f"Detected {total_cell_folders} cell_* folder(s) across {len(active_fields)} field(s).\n"
                "Choose a plot from the 'Plot View' dropdown and click 'Generate Selected Plot(s)'\n"
                "(or click an individual '↻ Regen' button in the sidebar)."
            )

    # =========================================================================
    # PARSING CELL FITTING PARAMETERS CSV, MASKS & LINE PROFILES
    # =========================================================================

    @staticmethod
    def _natural_cell_num(folder_name: str) -> int:
        m = re.search(r'(\d+)', folder_name)
        return int(m.group(1)) if m else -1

    @staticmethod
    def _angle_to_degrees(val: float) -> float:
        """Converts orientation angle to degrees wrapped to [-90, 90]."""
        v = float(val)
        if abs(v) <= (2.0 * np.pi + 1e-4) and not float(v).is_integer():
            deg = np.degrees(v)
        else:
            deg = v
        deg = ((deg + 90.0) % 180.0) - 90.0
        return float(deg)

    @staticmethod
    def _capsule_area_um2(length_um: float, width_um: float) -> float:
        """Computes 2D spherocylinder (capsule) area in um^2 from total length and width."""
        if not (np.isfinite(length_um) and np.isfinite(width_um) and length_um > 0 and width_um > 0):
            return np.nan
        r = 0.5 * width_um
        cyl_len = max(0.0, length_um - width_um)
        return float(cyl_len * width_um + np.pi * (r ** 2))

    @staticmethod
    def _normalize_channel_code(raw_ch, fallback_filename: str = "") -> str:
        """Maps channel identifiers ('L', 'R', 'Left Channel (L)', etc.) to 'L' or 'R'."""
        if raw_ch is not None and str(raw_ch).strip() != "":
            s = str(raw_ch).strip().upper()
            if s in ("R", "RIGHT") or "RIGHT" in s or s.endswith("(R)"):
                return "R"
            if s in ("L", "LEFT") or "LEFT" in s or s.endswith("(L)"):
                return "L"
        fn_u = os.path.basename(fallback_filename).upper()
        if "_R" in fn_u or "RIGHT" in fn_u:
            return "R"
        return "L"

    def _load_cell_fitting_parameters(self, res_dir: str, pixel_size: float):
        """
        Loads `cell_fitting_parameters*.csv` inside the preprocessing results folder (`res_dir`).
        Expected standard columns:
            Cell_Number, Channel, Length_nm, Width_nm, Center_X, Center_Y, Rotation_deg, Error_nm
        Returns a list of normalized parameter dicts for all rows found in `res_dir`.
        """
        search_dirs = [res_dir]
        parent_dir = os.path.dirname(res_dir)
        if parent_dir and os.path.isdir(parent_dir) and parent_dir != res_dir:
            search_dirs.append(parent_dir)

        csv_paths = []
        seen_paths = set()
        for d in search_dirs:
            for pat in ("*cell_fitting_parameter*.csv", "*fitting_parameter*.csv"):
                for p in sorted(glob.glob(os.path.join(d, pat))):
                    rp = os.path.realpath(p)
                    if rp not in seen_paths and os.path.isfile(p):
                        seen_paths.add(rp)
                        csv_paths.append(p)
            if csv_paths:
                break

        parsed_rows = []
        for csv_p in csv_paths:
            try:
                df = pd.read_csv(csv_p)
            except Exception:
                continue
            if df.empty:
                continue

            cols_l = {str(c).strip().lower(): c for c in df.columns}
            id_col = next(
                (cols_l[k] for k in ('cell_number', 'cell_num', 'label', 'cell_id', 'id') if k in cols_l),
                None
            )
            ch_col = next((cols_l[k] for k in ('channel', 'chan', 'ch') if k in cols_l), None)
            len_nm_col = next((cols_l[k] for k in ('length_nm', 'cell_length_nm') if k in cols_l), None)
            len_um_col = next((cols_l[k] for k in ('length_um', 'cell_length_um') if k in cols_l), None)
            len_px_col = next((cols_l[k] for k in ('length_px', 'length', 'major_axis_length') if k in cols_l), None)
            wid_nm_col = next((cols_l[k] for k in ('width_nm', 'cell_width_nm') if k in cols_l), None)
            wid_um_col = next((cols_l[k] for k in ('width_um', 'cell_width_um') if k in cols_l), None)
            wid_px_col = next((cols_l[k] for k in ('width_px', 'width', 'minor_axis_length') if k in cols_l), None)
            cx_col = next((cols_l[k] for k in ('center_x', 'centroid_x', 'xc', 'x0', 'x') if k in cols_l), None)
            cy_col = next((cols_l[k] for k in ('center_y', 'centroid_y', 'yc', 'y0', 'y') if k in cols_l), None)
            rot_col = next((cols_l[k] for k in ('rotation_deg', 'angle_deg', 'angle', 'orientation', 'theta') if k in cols_l), None)
            err_nm_col = next((cols_l[k] for k in ('error_nm', 'fit_error_nm', 'fitting_error_nm') if k in cols_l), None)
            err_px_col = next((cols_l[k] for k in ('error_px', 'fit_error', 'cell_fit_err', 'fit_err', 'error') if k in cols_l), None)
            area_um2_col = next((cols_l[k] for k in ('area_um2', 'cell_area_um2') if k in cols_l), None)
            area_px_col = next((cols_l[k] for k in ('area_px', 'area') if k in cols_l), None)

            for r_idx, row in df.iterrows():
                try:
                    c_num = int(float(row[id_col])) if id_col is not None and pd.notna(row[id_col]) else (r_idx + 1)
                except Exception:
                    c_num = r_idx + 1

                ch_code = self._normalize_channel_code(row[ch_col] if ch_col is not None else None, csv_p)

                length_um = np.nan
                if len_nm_col is not None and pd.notna(row[len_nm_col]):
                    length_um = float(row[len_nm_col]) / 1000.0
                elif len_um_col is not None and pd.notna(row[len_um_col]):
                    length_um = float(row[len_um_col])
                elif len_px_col is not None and pd.notna(row[len_px_col]):
                    length_um = float(row[len_px_col]) * pixel_size

                width_um = np.nan
                if wid_nm_col is not None and pd.notna(row[wid_nm_col]):
                    width_um = float(row[wid_nm_col]) / 1000.0
                elif wid_um_col is not None and pd.notna(row[wid_um_col]):
                    width_um = float(row[wid_um_col])
                elif wid_px_col is not None and pd.notna(row[wid_px_col]):
                    width_um = float(row[wid_px_col]) * pixel_size

                cx = float(row[cx_col]) if (cx_col is not None and pd.notna(row[cx_col])) else np.nan
                cy = float(row[cy_col]) if (cy_col is not None and pd.notna(row[cy_col])) else np.nan

                angle_deg = np.nan
                if rot_col is not None and pd.notna(row[rot_col]):
                    angle_deg = self._angle_to_degrees(float(row[rot_col]))

                fit_error_nm = np.nan
                if err_nm_col is not None and pd.notna(row[err_nm_col]):
                    fit_error_nm = float(row[err_nm_col])
                elif err_px_col is not None and pd.notna(row[err_px_col]):
                    fit_error_nm = float(row[err_px_col]) * pixel_size * 1000.0

                area_px = float(row[area_px_col]) if (area_px_col is not None and pd.notna(row[area_px_col])) else np.nan
                if area_um2_col is not None and pd.notna(row[area_um2_col]):
                    area_um2 = float(row[area_um2_col])
                elif np.isfinite(length_um) and np.isfinite(width_um):
                    area_um2 = self._capsule_area_um2(length_um, width_um)
                elif np.isfinite(area_px):
                    area_um2 = area_px * (pixel_size ** 2)
                else:
                    area_um2 = np.nan

                parsed_rows.append({
                    'cell_num': c_num,
                    'channel_code': ch_code,
                    'length_um': length_um,
                    'width_um': width_um,
                    'length_px': (length_um / pixel_size) if np.isfinite(length_um) else np.nan,
                    'width_px': (width_um / pixel_size) if np.isfinite(width_um) else np.nan,
                    'cx': cx,
                    'cy': cy,
                    'angle_deg': angle_deg,
                    'fit_error_nm': fit_error_nm,
                    'area_um2': area_um2,
                    'area_px': area_px,
                    'csv_path': csv_p,
                })

        return parsed_rows

    def _extract_cell_geometry_from_dict(self, data_dict: dict, pixel_size: float):
        """Extracts centroid (cx=col/x_px, cy=row/y_px), length_px, width_px, angle_deg, area_px, fit_error_nm from a dict."""
        lower_map = {str(k).lower(): v for k, v in data_dict.items()}

        cx, cy = None, None
        length_px, width_px, angle_deg = None, None, None
        area_px, fit_error_nm = None, None

        # 1. Capsule endpoints p1, p2 + radius
        if 'p1' in lower_map and 'p2' in lower_map:
            try:
                p1 = np.asarray(lower_map['p1'], dtype=float).ravel()
                p2 = np.asarray(lower_map['p2'], dtype=float).ravel()
                if p1.size >= 2 and p2.size >= 2:
                    cy = 0.5 * (p1[0] + p2[0])
                    cx = 0.5 * (p1[1] + p2[1])
                    dy = p2[0] - p1[0]
                    dx = p2[1] - p1[1]
                    cyl_len = float(np.hypot(dx, dy))
                    rad = float(np.squeeze(lower_map.get('radius', lower_map.get('r', 0.0))))
                    length_px = cyl_len + 2.0 * rad
                    width_px = 2.0 * rad if rad > 0 else None
                    angle_deg = self._angle_to_degrees(np.degrees(np.arctan2(dy, dx)))
            except Exception:
                pass

        # 2. Centroid keys
        if cx is None or cy is None:
            if 'centroid' in lower_map:
                c_arr = np.asarray(lower_map['centroid'], dtype=float).ravel()
                if c_arr.size >= 2:
                    cy, cx = float(c_arr[0]), float(c_arr[1])
            elif 'centroid-1' in lower_map and 'centroid-0' in lower_map:
                cy = float(np.squeeze(lower_map['centroid-0']))
                cx = float(np.squeeze(lower_map['centroid-1']))
            else:
                for kx, ky in [('center_x', 'center_y'), ('x', 'y'), ('xc', 'yc'), ('x0', 'y0'), ('centroid_x', 'centroid_y')]:
                    if kx in lower_map and ky in lower_map:
                        cx = float(np.squeeze(lower_map[kx]))
                        cy = float(np.squeeze(lower_map[ky]))
                        break

        # 3. Length keys
        if length_px is None:
            for lk in ['length_nm', 'length_um', 'length_px', 'cell_length_px', 'major_axis_length', 'length', 'cell_length', 'l']:
                if lk in lower_map:
                    try:
                        val = float(np.squeeze(lower_map[lk]))
                        if np.isfinite(val) and val > 0:
                            if lk.endswith('_nm'):
                                length_px = (val / 1000.0) / pixel_size
                            elif lk.endswith('_um'):
                                length_px = val / pixel_size
                            else:
                                length_px = val
                            break
                    except Exception:
                        pass

        # 4. Width keys
        if width_px is None:
            for wk in ['width_nm', 'width_um', 'width_px', 'cell_width_px', 'minor_axis_length', 'width', 'cell_width', 'w']:
                if wk in lower_map:
                    try:
                        val = float(np.squeeze(lower_map[wk]))
                        if np.isfinite(val) and val > 0:
                            if wk.endswith('_nm'):
                                width_px = (val / 1000.0) / pixel_size
                            elif wk.endswith('_um'):
                                width_px = val / pixel_size
                            else:
                                width_px = val
                            break
                    except Exception:
                        pass
            if width_px is None and 'radius' in lower_map:
                try:
                    rad = float(np.squeeze(lower_map['radius']))
                    if np.isfinite(rad) and rad > 0:
                        width_px = 2.0 * rad
                except Exception:
                    pass

        # 5. Angle keys
        if angle_deg is None:
            for ak in ['rotation_deg', 'angle_deg', 'angle', 'orientation', 'theta', 'cell_angle', 'angle_rad']:
                if ak in lower_map:
                    try:
                        val = float(np.squeeze(lower_map[ak]))
                        if np.isfinite(val):
                            if ak in ('rotation_deg', 'angle_deg'):
                                angle_deg = ((val + 90.0) % 180.0) - 90.0
                            elif ak == 'angle_rad':
                                angle_deg = ((np.degrees(val) + 90.0) % 180.0) - 90.0
                            else:
                                angle_deg = self._angle_to_degrees(val)
                            break
                    except Exception:
                        pass

        # 6. Area and Fit Error keys
        for ark in ['area_px', 'area']:
            if ark in lower_map:
                try:
                    val = float(np.squeeze(lower_map[ark]))
                    if np.isfinite(val) and val > 0:
                        area_px = val
                        break
                except Exception:
                    pass

        for erk in ['error_nm', 'fit_error_nm', 'fitting_error_nm', 'fit_error', 'cell_fit_err', 'fit_err', 'error']:
            if erk in lower_map:
                try:
                    val = float(np.squeeze(lower_map[erk]))
                    if np.isfinite(val):
                        fit_error_nm = val if erk.endswith('_nm') else (val * pixel_size * 1000.0)
                        break
                except Exception:
                    pass

        return cx, cy, length_px, width_px, angle_deg, area_px, fit_error_nm

    @staticmethod
    def _geometry_from_binary_mask(cell_mask: np.ndarray):
        """Computes centroid (cx=col, cy=row), major length (px), minor width (px), angle (deg), and area (px) from a 2D boolean mask."""
        ys, xs = np.nonzero(cell_mask)
        if xs.size < 3:
            return None, None, None, None, None, None

        area_px = float(xs.size)
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))

        coords = np.column_stack([xs - cx, ys - cy])
        cov = np.cov(coords, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]

        major_vec = eigvecs[:, 0]
        angle_deg = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
        angle_deg = ((angle_deg + 90.0) % 180.0) - 90.0

        proj_major = coords @ eigvecs[:, 0]
        proj_minor = coords @ eigvecs[:, 1]
        length_px = float(np.ptp(proj_major) + 1.0)
        width_px = float(np.ptp(proj_minor) + 1.0)

        return cx, cy, length_px, width_px, angle_deg, area_px

    @staticmethod
    def _read_tiff_stack_or_2d(path: str):
        """Reads a 2D or multi-page 3D TIFF image into a numpy array."""
        if not path or not os.path.exists(path):
            return None
        try:
            if tifffile is not None:
                arr = tifffile.imread(path)
                return np.asarray(arr)
            with Image.open(path) as im:
                n_frames = getattr(im, 'n_frames', 1)
                if n_frames <= 1:
                    return np.array(im)
                frames = []
                for idx in range(n_frames):
                    im.seek(idx)
                    frames.append(np.array(im))
                return np.stack(frames, axis=0)
        except Exception:
            return None

    def _load_fits_stack_data(self, res_dir: str):
        """Loads `fits_stack*.tif` from a results directory and indexes slices for fast cell lookup."""
        candidates = [
            "fits_stack.tif", "fits_stack.tiff",
            "L_fits_stack.tif", "L_fits_stack.tiff",
            "R_fits_stack.tif", "R_fits_stack.tiff",
            "all_cells_fits.tif", "all_cells_fits.tiff",
        ]
        fits_path = next(
            (os.path.join(res_dir, n) for n in candidates if os.path.exists(os.path.join(res_dir, n))),
            None
        )
        if fits_path is None:
            glob_matches = sorted(glob.glob(os.path.join(res_dir, "*fits_stack*.tif*")))
            fits_path = glob_matches[0] if glob_matches else None

        if fits_path is None:
            return None

        arr = self._read_tiff_stack_or_2d(fits_path)
        if arr is None or arr.size == 0:
            return None

        if arr.ndim == 2:
            return {'mode': '2d', 'arr': arr, 'path': fits_path}

        if arr.ndim == 3:
            slices = []
            for s_idx in range(arr.shape[0]):
                sl = arr[s_idx]
                ys, xs = np.nonzero(sl > 0)
                if xs.size == 0:
                    slices.append(None)
                    continue
                vals = sl[ys, xs]
                unique_vals = np.unique(vals)
                label_val = int(unique_vals[0]) if unique_vals.size == 1 and unique_vals[0] > 1 else None
                slices.append({
                    'slice_idx': s_idx,
                    'mask': sl > 0,
                    'cx': float(np.mean(xs)),
                    'cy': float(np.mean(ys)),
                    'label_val': label_val,
                })
            return {'mode': '3d', 'slices': slices, 'path': fits_path}

        return None

    @staticmethod
    def _extract_mask_from_fits_stack(fits_data, cell_num: int, cell_order_idx: int, cx=None, cy=None, length_px=None):
        """Retrieves the 2D boolean capsule mask for a given cell from `fits_stack`."""
        if fits_data is None:
            return None

        if fits_data['mode'] == '2d':
            arr = fits_data['arr']
            if cell_num > 0:
                m_bool = (arr == cell_num)
                if np.any(m_bool):
                    return m_bool
            m_ord = (arr == (cell_order_idx + 1))
            if np.any(m_ord):
                return m_ord
            return None

        slices = fits_data.get('slices', [])
        if not slices:
            return None

        if cell_num > 0:
            for item in slices:
                if item is not None and item['label_val'] == cell_num:
                    return item['mask']

        if cx is not None and cy is not None and np.isfinite(cx) and np.isfinite(cy):
            max_tol = max(12.0, 0.75 * float(length_px or 20.0))
            best_item = None
            best_dist = float('inf')
            for item in slices:
                if item is None:
                    continue
                d = float(np.hypot(item['cx'] - cx, item['cy'] - cy))
                if d < best_dist:
                    best_dist = d
                    best_item = item
            if best_item is not None and best_dist <= max_tol:
                return best_item['mask']

        if 0 <= cell_order_idx < len(slices) and slices[cell_order_idx] is not None:
            return slices[cell_order_idx]['mask']

        return None

    def _load_line_profiles_from_cell_folder(self, cell_folder: str, mat_dict: dict):
        """Extracts saved line profile arrays for L and R channels from a cell_* folder."""
        profiles = {'Left Channel (L)': None, 'Right Channel (R)': None}
        if not cell_folder or not os.path.isdir(cell_folder):
            return profiles

        if mat_dict:
            for k, v in mat_dict.items():
                kl = str(k).lower()
                if 'profile' not in kl or 'location' in kl:
                    continue
                arr = np.asarray(v, dtype=float).squeeze()
                if arr.ndim == 1 and arr.size >= 4 and np.all(np.isfinite(arr)):
                    if '_r' in kl or kl.endswith('r') or 'right' in kl:
                        profiles['Right Channel (R)'] = arr
                    elif '_l' in kl or kl.endswith('l') or 'left' in kl:
                        profiles['Left Channel (L)'] = arr
                    elif profiles['Left Channel (L)'] is None:
                        profiles['Left Channel (L)'] = arr

        def _profile_file_priority(fn: str):
            fl = fn.lower()
            if fl.startswith('line_profile_l.'):
                return (0, fl)
            if fl.startswith('line_profile_r.'):
                return (1, fl)
            if 'combined' in fl:
                return (3, fl)
            return (2, fl)

        sorted_files = sorted(os.listdir(cell_folder), key=_profile_file_priority)

        for fname in sorted_files:
            flower = fname.lower()
            if 'profile' not in flower or flower.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                continue
            fpath = os.path.join(cell_folder, fname)
            try:
                if flower.endswith('.npy'):
                    arr = np.load(fpath, allow_pickle=True)
                    arr = np.asarray(arr, dtype=float).squeeze()
                    if arr.ndim == 1 and arr.size >= 4:
                        ch = 'Right Channel (R)' if ('_r' in flower or 'r_' in flower) else 'Left Channel (L)'
                        if profiles[ch] is None:
                            profiles[ch] = arr
                elif flower.endswith(('.csv', '.tsv', '.dat', '.txt')):
                    sep = '\t' if flower.endswith('.tsv') else (',' if flower.endswith('.csv') else None)
                    df_p = pd.read_csv(fpath, sep=sep, engine='python')
                    cols_l = {c.lower(): c for c in df_p.columns}
                    l_col = next((cols_l[c] for c in cols_l if c in ('l', 'left', 'l_intensity', 'intensity_l', 'l_profile')), None)
                    r_col = next((cols_l[c] for c in cols_l if c in ('r', 'right', 'r_intensity', 'intensity_r', 'r_profile')), None)
                    if l_col:
                        vals = pd.to_numeric(df_p[l_col], errors='coerce').dropna().values
                        if vals.size >= 4:
                            profiles['Left Channel (L)'] = vals
                    if r_col:
                        vals = pd.to_numeric(df_p[r_col], errors='coerce').dropna().values
                        if vals.size >= 4:
                            profiles['Right Channel (R)'] = vals
                    if not l_col and not r_col and 'combined' not in flower:
                        num_df = df_p.apply(pd.to_numeric, errors='coerce').dropna(how='all')
                        if not num_df.empty:
                            vals = num_df.iloc[:, -1].dropna().values
                            if vals.size >= 4:
                                ch = 'Right Channel (R)' if ('_r' in flower or 'r_' in flower) else 'Left Channel (L)'
                                if profiles[ch] is None:
                                    profiles[ch] = vals
            except Exception:
                pass

        return profiles

    def _compute_profile_from_avg_image(self, avg_img: np.ndarray, record: dict, axis_mode: str, tmat=None):
        """Computes 1D longitudinal or transverse intensity profile directly from L_avg/R_avg image."""
        if avg_img is None:
            return None

        cx, cy = record['cx'], record['cy']
        length_px = record['length_px']
        width_px = record['width_px']
        theta = np.radians(record['angle_deg'])

        if not (np.isfinite(cx) and np.isfinite(cy) and length_px > 2 and width_px > 1):
            return None

        n_long = max(12, int(round(length_px * 2)))
        n_trans = max(7, int(round(width_px * 2)))

        u_grid = np.linspace(-0.5, 0.5, n_long) * length_px
        v_grid = np.linspace(-0.5, 0.5, n_trans) * width_px
        UU, VV = np.meshgrid(u_grid, v_grid)

        XX = cx + UU * np.cos(theta) - VV * np.sin(theta)
        YY = cy + UU * np.sin(theta) + VV * np.cos(theta)

        if tmat is not None:
            tmat_3x3 = np.vstack([tmat, [0, 0, 1]]) if tmat.shape == (2, 3) else tmat
            if tmat_3x3.shape == (3, 3):
                pts = np.vstack([XX.ravel(), YY.ravel(), np.ones(XX.size)])
                res = tmat_3x3 @ pts
                denom = np.where(res[2] != 0, res[2], 1.0)
                XX = (res[0] / denom).reshape(XX.shape)
                YY = (res[1] / denom).reshape(YY.shape)

        sampled = ndimage.map_coordinates(avg_img.astype(np.float64), [YY, XX], order=1, mode='nearest')

        r_cap = 0.5 * width_px
        cyl_half = max(0.0, 0.5 * length_px - r_cap)
        dx_cap = np.maximum(0.0, np.abs(UU) - cyl_half)
        inside_cap = (dx_cap**2 + VV**2) <= (r_cap**2)
        sampled = np.where(inside_cap, sampled, np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if axis_mode.startswith("Longitudinal"):
                prof = np.nanmean(sampled, axis=0)
            else:
                prof = np.nanmean(sampled, axis=1)

        return prof if np.any(np.isfinite(prof)) else None

    @staticmethod
    def _load_2d_image(path: str):
        if not path or not os.path.exists(path):
            return None
        try:
            with Image.open(path) as im:
                arr = np.array(im, dtype=np.float64)
            if arr.ndim == 3:
                arr = np.mean(arr[..., :3], axis=-1)
            return arr
        except Exception:
            return None

    def _scan_and_parse_cells(self, progress_dlg=None):
        """
        Scans all active field results folders, reads `cell_fitting_parameters*.csv`
        (Length_nm, Width_nm, Center_X, Center_Y, Rotation_deg, Error_nm, Channel) and
        `cell_*` folders, and builds unified cell records.
        """
        pixel_size = self.parent.spin_pixel_size.value() if (self.parent and hasattr(self.parent, 'spin_pixel_size')) else 0.051
        min_len_um = self.spin_min_cell_len.value()
        max_len_um = self.spin_max_cell_len.value()
        mask_source_mode = self.combo_mask_source.currentText()
        prefer_fits = not mask_source_mode.startswith("Segmentation Mask")
        allow_seg_fallback = not mask_source_mode.startswith("Fitted Capsule Stack")

        active_fields = self._get_active_field_dirs()
        records = []

        for f_idx, (sample_name, field_name, res_dir) in enumerate(active_fields):
            if progress_dlg is not None:
                if progress_dlg.wasCanceled():
                    break
                progress_dlg.setLabelText(f"Scanning cells in field {field_name} ({f_idx + 1}/{len(active_fields)})...")
                QApplication.processEvents()

            # Load cell_fitting_parameters*.csv inside the preprocessing results folder
            fit_csv_rows = self._load_cell_fitting_parameters(res_dir, pixel_size)
            fit_by_num_ch = {}
            fit_by_num = {}
            for fr in fit_csv_rows:
                fit_by_num_ch[(fr['cell_num'], fr['channel_code'])] = fr
                fit_by_num.setdefault(fr['cell_num'], []).append(fr)

            df_objs = None
            obj_csv = os.path.join(res_dir, "all_objects.csv")
            if os.path.exists(obj_csv):
                try:
                    df_objs = pd.read_csv(obj_csv)
                except Exception:
                    df_objs = None

            fits_data = self._load_fits_stack_data(res_dir) if prefer_fits else None

            labeled_mask = None
            if allow_seg_fallback:
                for m_name in ("all_cells_mask.tif", "all_cells_mask_L.tif", "all_cells_mask_R.tif"):
                    m_path = os.path.join(res_dir, m_name)
                    if os.path.exists(m_path):
                        try:
                            with Image.open(m_path) as im:
                                labeled_mask = np.array(im)
                            break
                        except Exception:
                            pass

            l_avg_path = next(
                (os.path.join(res_dir, n) for n in ("L_avg.tif", "L_avg.png") if os.path.exists(os.path.join(res_dir, n))),
                None
            )
            r_avg_path = next(
                (os.path.join(res_dir, n) for n in ("R_avg.tif", "R_avg.png") if os.path.exists(os.path.join(res_dir, n))),
                None
            )

            tmat, _ = (
                self.parent.find_transformation_matrix(field_name)
                if (self.parent and hasattr(self.parent, 'find_transformation_matrix'))
                else (None, None)
            )

            cell_dirs = sorted(
                [d for d in glob.glob(os.path.join(res_dir, "cell_*")) if os.path.isdir(d)],
                key=lambda p: self._natural_cell_num(os.path.basename(p))
            )
            cell_dir_by_num = {self._natural_cell_num(os.path.basename(d)): d for d in cell_dirs}

            # Build union of cell numbers from cell_* folders and cell_fitting_parameters*.csv
            ordered_cell_nums = sorted(set(cell_dir_by_num.keys()) | set(fit_by_num.keys()))

            for c_idx, cell_num in enumerate(ordered_cell_nums):
                c_dir = cell_dir_by_num.get(cell_num, "")
                folder_name = os.path.basename(c_dir) if c_dir else f"cell_{cell_num}"

                merged_meta = {}
                if df_objs is not None and not df_objs.empty:
                    id_col = next((c for c in ('label', 'Cell_Number', 'cell_num', 'label_id', 'object_id') if c in df_objs.columns), None)
                    if id_col is not None:
                        match = df_objs[df_objs[id_col] == cell_num]
                        if not match.empty:
                            merged_meta.update(match.iloc[0].to_dict())

                mat_dict = {}
                if c_dir and os.path.isdir(c_dir):
                    mat_files = glob.glob(os.path.join(c_dir, "*.mat"))
                    if mat_files:
                        try:
                            mat_dict = sio.loadmat(mat_files[0], squeeze_me=True, struct_as_record=False)
                            merged_meta.update({k: v for k, v in mat_dict.items() if not str(k).startswith('__')})
                        except Exception:
                            pass

                # Choose primary fitting row (prefer L for canonical cell coordinates; keep channel-specific rows too)
                cell_fit_rows = fit_by_num.get(cell_num, [])
                primary_fit = fit_by_num_ch.get((cell_num, 'L')) or (cell_fit_rows[0] if cell_fit_rows else None)

                cx, cy, length_px, width_px, angle_deg, area_px, fit_error_nm = (
                    self._extract_cell_geometry_from_dict(merged_meta, pixel_size)
                )

                if primary_fit is not None:
                    if np.isfinite(primary_fit['cx']):
                        cx = primary_fit['cx']
                    if np.isfinite(primary_fit['cy']):
                        cy = primary_fit['cy']
                    if np.isfinite(primary_fit['length_px']) and primary_fit['length_px'] > 0:
                        length_px = primary_fit['length_px']
                    if np.isfinite(primary_fit['width_px']) and primary_fit['width_px'] > 0:
                        width_px = primary_fit['width_px']
                    if np.isfinite(primary_fit['angle_deg']):
                        angle_deg = primary_fit['angle_deg']
                    if np.isfinite(primary_fit['fit_error_nm']):
                        fit_error_nm = primary_fit['fit_error_nm']
                    if area_px is None and np.isfinite(primary_fit['area_px']):
                        area_px = primary_fit['area_px']

                cell_mask = None
                mask_used_source = "cell_fitting_parameters.csv" if primary_fit is not None else "none"

                # 1. Try fitted capsule stack (`fits_stack.tif`)
                if prefer_fits and fits_data is not None:
                    cell_mask = self._extract_mask_from_fits_stack(
                        fits_data, cell_num, c_idx, cx=cx, cy=cy, length_px=length_px
                    )
                    if cell_mask is not None:
                        mask_used_source = "fits_stack"

                # 2. Fallback to segmentation masks (`all_cells_mask*.tif` -> `single_cell_mask*.tif`)
                if cell_mask is None and allow_seg_fallback:
                    if labeled_mask is not None and labeled_mask.ndim == 2 and cell_num > 0:
                        m_bool = (labeled_mask == cell_num)
                        if np.any(m_bool):
                            cell_mask = m_bool
                            mask_used_source = "all_cells_mask"

                    if cell_mask is None and c_dir and os.path.isdir(c_dir):
                        explicit_single_masks = [
                            "single_cell_mask_L.tif",
                            "single_cell_mask.tif",
                            "single_cell_mask_R.tif",
                        ]
                        candidate_paths = [os.path.join(c_dir, n) for n in explicit_single_masks]
                        extra_masks = sorted(
                            p for p in glob.glob(os.path.join(c_dir, "*mask*.tif*"))
                            if not os.path.basename(p).lower().startswith("cellmasked")
                        )
                        candidate_paths.extend(extra_masks)

                        for mask_f in candidate_paths:
                            if not os.path.exists(mask_f):
                                continue
                            try:
                                with Image.open(mask_f) as im:
                                    arr_m = np.array(im) > 0
                                if arr_m.ndim == 2 and np.any(arr_m):
                                    cell_mask = arr_m
                                    mask_used_source = os.path.basename(mask_f)
                                    break
                            except Exception:
                                pass

                # Cross-validate and fill missing geometry against the 2D binary cell mask
                if cell_mask is not None:
                    m_cx, m_cy, m_len, m_wid, m_ang, m_area_px = self._geometry_from_binary_mask(cell_mask)
                    if cx is None:
                        cx = m_cx
                    if cy is None:
                        cy = m_cy
                    if length_px is None:
                        length_px = m_len
                    if width_px is None:
                        width_px = m_wid
                    if angle_deg is None:
                        angle_deg = m_ang
                    if area_px is None:
                        area_px = m_area_px

                    # Safeguard against transposed (row, col) centroids in external metadata
                    if m_cx is not None and m_cy is not None and cx is not None and cy is not None:
                        dist_direct = np.hypot(cx - m_cx, cy - m_cy)
                        dist_swapped = np.hypot(cy - m_cx, cx - m_cy)
                        if dist_swapped + 2.0 < dist_direct:
                            cx, cy = cy, cx

                    # Safeguard against transposed atan2(dx, dy) orientation angles
                    if m_ang is not None and angle_deg is not None:
                        diff_rad = np.radians(angle_deg - m_ang)
                        if abs(np.cos(diff_rad)) < abs(np.sin(diff_rad)):
                            angle_deg = m_ang

                if cx is None or cy is None or length_px is None or width_px is None or length_px <= 0 or width_px <= 0:
                    continue

                if angle_deg is None:
                    angle_deg = 0.0

                length_um = float(length_px * pixel_size)
                width_um = float(width_px * pixel_size)

                if min_len_um > 0 and length_um < min_len_um:
                    continue
                if max_len_um > 0 and length_um > max_len_um:
                    continue

                if primary_fit is not None and np.isfinite(primary_fit['area_um2']):
                    area_um2 = float(primary_fit['area_um2'])
                else:
                    area_um2 = self._capsule_area_um2(length_um, width_um)
                    if not np.isfinite(area_um2) and area_px is not None:
                        area_um2 = float(area_px) * (pixel_size ** 2)

                if area_px is None and np.isfinite(area_um2):
                    area_px = area_um2 / (pixel_size ** 2)

                # Store per-channel fitting entries (from cell_fitting_parameters*.csv) so L and R can be plotted separately or together
                channel_fits = {}
                for fr in cell_fit_rows:
                    ch_key = "Right Channel (R)" if fr['channel_code'] == 'R' else "Left Channel (L)"
                    fr_len = fr['length_um'] if np.isfinite(fr['length_um']) else length_um
                    fr_wid = fr['width_um'] if np.isfinite(fr['width_um']) else width_um
                    fr_ang = fr['angle_deg'] if np.isfinite(fr['angle_deg']) else float(angle_deg)
                    fr_area = fr['area_um2'] if np.isfinite(fr['area_um2']) else self._capsule_area_um2(fr_len, fr_wid)
                    fr_err = fr['fit_error_nm'] if np.isfinite(fr['fit_error_nm']) else (float(fit_error_nm) if fit_error_nm is not None else np.nan)
                    channel_fits[ch_key] = {
                        'length_um': float(fr_len),
                        'width_um': float(fr_wid),
                        'angle_deg': float(fr_ang),
                        'area_um2': float(fr_area) if np.isfinite(fr_area) else np.nan,
                        'fit_error_nm': float(fr_err) if np.isfinite(fr_err) else np.nan,
                    }

                saved_profiles = self._load_line_profiles_from_cell_folder(c_dir, mat_dict)

                records.append({
                    'sample': sample_name,
                    'field': field_name,
                    'res_dir': res_dir,
                    'cell_dir': c_dir,
                    'cell_folder': folder_name,
                    'cell_num': cell_num,
                    'channel_code': primary_fit['channel_code'] if primary_fit is not None else 'L',
                    'cx': float(cx),
                    'cy': float(cy),
                    'length_px': float(length_px),
                    'width_px': float(width_px),
                    'length_um': length_um,
                    'width_um': width_um,
                    'angle_deg': float(angle_deg),
                    'area_px': float(area_px) if area_px is not None else np.nan,
                    'area_um2': float(area_um2) if np.isfinite(area_um2) else np.nan,
                    'fit_error_nm': float(fit_error_nm) if (fit_error_nm is not None and np.isfinite(fit_error_nm)) else np.nan,
                    'channel_fits': channel_fits,
                    'cell_mask': cell_mask,
                    'mask_source': mask_used_source,
                    'l_avg_path': l_avg_path,
                    'r_avg_path': r_avg_path,
                    'tmat': tmat,
                    'saved_profiles': saved_profiles,
                })

        self.cell_records = records
        return records

    # =========================================================================
    # CANONICAL CELL HEATMAP & LINE PROFILE COMPUTATIONS
    # =========================================================================

    def _build_canonical_grid(self, records):
        """Constructs pixel-size-calibrated canonical cell grid dimensions."""
        pixel_size = self.parent.spin_pixel_size.value() if (self.parent and hasattr(self.parent, 'spin_pixel_size')) else 0.051
        bin_step_um = pixel_size * self.spin_bin_px_scale.value()

        mean_len_um = float(np.mean([r['length_um'] for r in records]))
        mean_wid_um = float(np.mean([r['width_um'] for r in records]))

        nx = max(6, int(round(mean_len_um / bin_step_um)))
        ny = max(4, int(round(mean_wid_um / bin_step_um)))

        u_edges = np.linspace(-0.5, 0.5, nx + 1)
        v_edges = np.linspace(-0.5, 0.5, ny + 1)
        u_centers = 0.5 * (u_edges[:-1] + u_edges[1:])
        v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])

        UU_um, VV_um = np.meshgrid(u_centers * mean_len_um, v_centers * mean_wid_um)
        r_cap = 0.5 * mean_wid_um
        cyl_half = max(0.0, 0.5 * mean_len_um - r_cap)
        dx_cap = np.maximum(0.0, np.abs(UU_um) - cyl_half)
        capsule_mask = (dx_cap**2 + VV_um**2) <= (r_cap**2)

        return {
            'pixel_size': pixel_size,
            'bin_step_um': bin_step_um,
            'mean_len_um': mean_len_um,
            'mean_wid_um': mean_wid_um,
            'nx': nx,
            'ny': ny,
            'u_edges': u_edges,
            'v_edges': v_edges,
            'capsule_mask': capsule_mask,
        }

    @staticmethod
    def _bin_points_to_cell_grid(x_px, y_px, values, record, grid_info):
        """Transforms image pixel coordinates (x_px, y_px) into normalized cell coordinates (u, v)
        and returns a (ny, nx) grid of mean values per bin for a single cell.
        """
        if len(x_px) == 0:
            return None

        theta = np.radians(record['angle_deg'])
        dx = np.asarray(x_px, dtype=np.float64) - record['cx']
        dy = np.asarray(y_px, dtype=np.float64) - record['cy']

        x_rot = dx * np.cos(theta) + dy * np.sin(theta)
        y_rot = -dx * np.sin(theta) + dy * np.cos(theta)

        u = x_rot / record['length_px']
        v = y_rot / record['width_px']

        in_bounds = (u >= -0.5) & (u <= 0.5) & (v >= -0.5) & (v <= 0.5) & np.isfinite(values)
        if not np.any(in_bounds):
            return None

        u_b = u[in_bounds]
        v_b = v[in_bounds]
        val_b = np.asarray(values, dtype=np.float64)[in_bounds]

        ny, nx = grid_info['ny'], grid_info['nx']
        sum_grid, _, _ = np.histogram2d(
            v_b, u_b,
            bins=[grid_info['v_edges'], grid_info['u_edges']],
            weights=val_b
        )
        cnt_grid, _, _ = np.histogram2d(
            v_b, u_b,
            bins=[grid_info['v_edges'], grid_info['u_edges']]
        )

        with np.errstate(divide='ignore', invalid='ignore'):
            mean_grid = np.where(cnt_grid > 0, sum_grid / cnt_grid, np.nan)
        return mean_grid

    def _compute_intensity_heatmap(self, records, grid_info, channel: str, progress_dlg=None):
        """Computes cell-averaged frame-average intensity heatmap for the specified channel."""
        img_cache = {}
        per_cell_grids = []
        norm_per_cell = self.chk_norm_cell_intensity.isChecked()
        use_tmat = self.chk_use_tmat_for_r.isChecked() and (channel == "Right Channel (R)")

        for idx, rec in enumerate(records):
            if progress_dlg is not None and idx % 20 == 0:
                if progress_dlg.wasCanceled():
                    break
                progress_dlg.setLabelText(f"Computing intensity heatmap [{channel}] ({idx + 1}/{len(records)} cells)...")
                QApplication.processEvents()

            img_path = rec['r_avg_path'] if channel == "Right Channel (R)" else rec['l_avg_path']
            if not img_path:
                continue

            if img_path not in img_cache:
                img_cache[img_path] = self._load_2d_image(img_path)
            avg_img = img_cache[img_path]
            if avg_img is None:
                continue

            h, w = avg_img.shape
            if rec['cell_mask'] is not None and rec['cell_mask'].shape == (h, w):
                ys, xs = np.nonzero(rec['cell_mask'])
            else:
                half_box = int(np.ceil(0.6 * max(rec['length_px'], rec['width_px'])))
                x0 = max(0, int(np.floor(rec['cx'] - half_box)))
                x1 = min(w, int(np.ceil(rec['cx'] + half_box + 1)))
                y0 = max(0, int(np.floor(rec['cy'] - half_box)))
                y1 = min(h, int(np.ceil(rec['cy'] + half_box + 1)))
                yy, xx = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing='ij')
                xs, ys = xx.ravel(), yy.ravel()

            if xs.size == 0:
                continue

            if use_tmat and rec['tmat'] is not None:
                tmat = rec['tmat']
                tmat_3x3 = np.vstack([tmat, [0, 0, 1]]) if tmat.shape == (2, 3) else tmat
                if tmat_3x3.shape == (3, 3):
                    pts = np.vstack([xs, ys, np.ones_like(xs, dtype=float)])
                    res = tmat_3x3 @ pts
                    denom = np.where(res[2] != 0, res[2], 1.0)
                    sx = res[0] / denom
                    sy = res[1] / denom
                    vals = ndimage.map_coordinates(avg_img, [sy, sx], order=1, mode='nearest')
                else:
                    vals = avg_img[ys, xs]
            else:
                vals = avg_img[ys, xs]

            if norm_per_cell:
                mean_v = np.nanmean(vals)
                if np.isfinite(mean_v) and mean_v > 0:
                    vals = vals / mean_v

            cell_grid = self._bin_points_to_cell_grid(xs, ys, vals, rec, grid_info)
            if cell_grid is not None:
                per_cell_grids.append(cell_grid)

        if not per_cell_grids:
            return None, 0

        stack = np.stack(per_cell_grids, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_grid = np.nanmean(stack, axis=0)
        return avg_grid, len(per_cell_grids)

    def _compute_track_heatmaps(self, records, grid_info, channel: str, want_diff: bool, want_jump: bool, progress_dlg=None):
        """Computes cell-averaged Diffusion Coefficient (D) and Jump Distance (r) heatmaps for one channel."""
        if self.parent is None or not hasattr(self.parent, 'trajectories_df') or self.parent.trajectories_df.empty:
            return None, 0, None, 0

        summary_df = self.parent.get_filtered_summary_df()
        if summary_df.empty:
            return None, 0, None, 0

        use_tmat = self.chk_use_tmat_for_r.isChecked() and (channel == "Right Channel (R)")
        if 'channel' in summary_df.columns:
            ch_vals = set(summary_df['channel'].dropna().unique())
            if len(ch_vals) > 1 or (len(ch_vals) == 1 and channel not in ch_vals and not use_tmat):
                summary_df = summary_df[summary_df['channel'] == channel]

        d_min_band = self.spin_d_band_min.value()
        d_max_band = self.spin_d_band_max.value()
        if d_min_band > 0:
            summary_df = summary_df[summary_df['D'] >= d_min_band]
        if d_max_band > 0:
            summary_df = summary_df[summary_df['D'] <= d_max_band]

        if summary_df.empty:
            return None, 0, None, 0

        allowed_ids = set(summary_df['global_track_id'].unique())
        track_d_map = dict(zip(summary_df['global_track_id'], summary_df['D']))

        traj_df = self.parent.trajectories_df[
            self.parent.trajectories_df['global_track_id'].isin(allowed_ids)
        ].copy()

        if traj_df.empty:
            return None, 0, None, 0

        pixel_size = grid_info['pixel_size']
        lag_N = self.parent.spin_jump_lag.value() if hasattr(self.parent, 'spin_jump_lag') else 1

        field_loc_data = {}
        field_jump_data = {}

        for field_name, f_df in traj_df.groupby('field'):
            tmat, _ = self.parent.find_transformation_matrix(field_name) if hasattr(self.parent, 'find_transformation_matrix') else (None, None)

            if want_diff:
                x_arr = f_df['x'].values.astype(float)
                y_arr = f_df['y'].values.astype(float)
                d_arr = np.array([track_d_map.get(tid, np.nan) for tid in f_df['global_track_id']], dtype=float)
                if use_tmat and tmat is not None:
                    x_arr, y_arr = self.parent.transform_track_coordinates(x_arr, y_arr, tmat)
                field_loc_data[field_name] = (x_arr, y_arr, d_arr)

            if want_jump:
                jx, jy, jr = [], [], []
                for tid, tr in f_df.groupby('global_track_id'):
                    if len(tr) <= lag_N:
                        continue
                    tr_s = tr.sort_values('frame')
                    frames = tr_s['frame'].values
                    tx = tr_s['x'].values.astype(float)
                    ty = tr_s['y'].values.astype(float)
                    if use_tmat and tmat is not None:
                        tx, ty = self.parent.transform_track_coordinates(tx, ty, tmat)
                    valid_step = (frames[lag_N:] - frames[:-lag_N]) == lag_N
                    if not np.any(valid_step):
                        continue
                    dx_um = (tx[lag_N:] - tx[:-lag_N])[valid_step] * pixel_size
                    dy_um = (ty[lag_N:] - ty[:-lag_N])[valid_step] * pixel_size
                    r_um = np.sqrt(dx_um**2 + dy_um**2)
                    mid_x = 0.5 * (tx[lag_N:] + tx[:-lag_N])[valid_step]
                    mid_y = 0.5 * (ty[lag_N:] + ty[:-lag_N])[valid_step]
                    jx.extend(mid_x)
                    jy.extend(mid_y)
                    jr.extend(r_um)
                if jx:
                    field_jump_data[field_name] = (np.array(jx), np.array(jy), np.array(jr))

        diff_grids = []
        jump_grids = []

        for idx, rec in enumerate(records):
            if progress_dlg is not None and idx % 20 == 0:
                if progress_dlg.wasCanceled():
                    break
                progress_dlg.setLabelText(f"Mapping tracks [{channel}] ({idx + 1}/{len(records)} cells)...")
                QApplication.processEvents()

            fld = rec['field']
            half_box = 0.65 * max(rec['length_px'], rec['width_px'])

            if want_diff and fld in field_loc_data:
                fx, fy, fd = field_loc_data[fld]
                near = (np.abs(fx - rec['cx']) <= half_box) & (np.abs(fy - rec['cy']) <= half_box)
                if np.any(near):
                    g_d = self._bin_points_to_cell_grid(fx[near], fy[near], fd[near], rec, grid_info)
                    if g_d is not None:
                        diff_grids.append(g_d)

            if want_jump and fld in field_jump_data:
                jx, jy, jr = field_jump_data[fld]
                near = (np.abs(jx - rec['cx']) <= half_box) & (np.abs(jy - rec['cy']) <= half_box)
                if np.any(near):
                    g_j = self._bin_points_to_cell_grid(jx[near], jy[near], jr[near], rec, grid_info)
                    if g_j is not None:
                        jump_grids.append(g_j)

        diff_avg = None
        if diff_grids:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                diff_avg = np.nanmean(np.stack(diff_grids, axis=0), axis=0)

        jump_avg = None
        if jump_grids:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                jump_avg = np.nanmean(np.stack(jump_grids, axis=0), axis=0)

        return diff_avg, len(diff_grids), jump_avg, len(jump_grids)

    def _collect_channel_line_profiles(self, records, channels, progress_dlg=None):
        """Collects & resamples line profiles onto a common normalized axis [0, 1] for the requested channel(s)."""
        axis_mode = self.combo_profile_axis.currentText()
        use_tmat = self.chk_use_tmat_for_r.isChecked()
        n_common = 100
        x_common = np.linspace(0.0, 1.0, n_common)

        img_cache = {}
        profiles_by_chan = {ch: [] for ch in channels}

        channel_path_map = {
            'Left Channel (L)': 'l_avg_path',
            'Right Channel (R)': 'r_avg_path',
        }

        for idx, rec in enumerate(records):
            if progress_dlg is not None and idx % 20 == 0:
                if progress_dlg.wasCanceled():
                    break
                progress_dlg.setLabelText(f"Extracting line profiles ({idx + 1}/{len(records)} cells)...")
                QApplication.processEvents()

            for ch_name in channels:
                path_key = channel_path_map[ch_name]
                raw_prof = None
                if axis_mode.startswith("Longitudinal") and rec['saved_profiles'].get(ch_name) is not None:
                    raw_prof = rec['saved_profiles'][ch_name]
                else:
                    img_p = rec.get(path_key)
                    if img_p:
                        if img_p not in img_cache:
                            img_cache[img_p] = self._load_2d_image(img_p)
                        tmat = rec['tmat'] if (use_tmat and ch_name == 'Right Channel (R)') else None
                        raw_prof = self._compute_profile_from_avg_image(img_cache[img_p], rec, axis_mode, tmat=tmat)

                if raw_prof is not None and len(raw_prof) >= 4:
                    valid = np.isfinite(raw_prof)
                    if np.sum(valid) >= 4:
                        x_orig = np.linspace(0.0, 1.0, len(raw_prof))[valid]
                        y_orig = raw_prof[valid]
                        interp_fn = interpolate.interp1d(x_orig, y_orig, kind='linear', fill_value='extrapolate')
                        profiles_by_chan[ch_name].append(interp_fn(x_common))

        return x_common, profiles_by_chan

    # =========================================================================
    # FULL & SPECIFIC-PLOT GENERATION + RENDERING
    # =========================================================================

    def _update_stats_label(self):
        if not self._has_generated or not self._cached_results:
            return
        cache = self._cached_results
        records = cache['records']
        grid_info = cache['grid_info']
        lengths = np.array([r['length_um'] for r in records])
        widths = np.array([r['width_um'] for r in records])
        angles = np.array([r['angle_deg'] for r in records])
        areas = np.array([r['area_um2'] for r in records if np.isfinite(r.get('area_um2', np.nan))])
        errors = np.array([r['fit_error_nm'] for r in records if np.isfinite(r.get('fit_error_nm', np.nan))])

        mask_sources = sorted({r.get('mask_source', 'unknown') for r in records})
        mask_src_str = ", ".join(mask_sources)

        chans = self._active_channels_list()
        hm_lines = []
        for ch in chans:
            short = "L" if "Left" in ch else "R"
            n_i = cache.get('int_grids', {}).get(ch, (None, 0))[1]
            n_d = cache.get('diff_grids', {}).get(ch, (None, 0))[1]
            n_j = cache.get('jump_grids', {}).get(ch, (None, 0))[1]
            hm_lines.append(f"<b>{short}:</b> Int={n_i}, D={n_d}, Jump={n_j}")

        area_str = f"{np.mean(areas):.2f} ± {np.std(areas):.2f} μm²" if areas.size > 0 else "N/A"
        err_str = f"{np.mean(errors):.2f} ± {np.std(errors):.2f} nm" if errors.size > 0 else "N/A"

        self.lbl_cell_count_banner.setText(
            f"Analyzed {len(records)} cell(s) [{mask_src_str}] | Pixel size: {grid_info['pixel_size']:.4f} μm/px "
            f"| Mean L×W: {grid_info['mean_len_um']:.2f} × {grid_info['mean_wid_um']:.2f} μm"
        )
        self.lbl_analyser_stats.setText(
            f"<b>Cells Analyzed:</b> {len(records)} (Source: {mask_src_str})<br>"
            f"<b>Pixel Size:</b> {grid_info['pixel_size']:.4f} μm/px (Bin: {grid_info['bin_step_um']:.4f} μm)<br>"
            f"<b>Mean Length:</b> {np.mean(lengths):.2f} ± {np.std(lengths):.2f} μm<br>"
            f"<b>Mean Width:</b> {np.mean(widths):.2f} ± {np.std(widths):.2f} μm<br>"
            f"<b>Mean Area:</b> {area_str}<br>"
            f"<b>Mean Angle:</b> {np.mean(angles):.1f} ± {np.std(angles):.1f}°<br>"
            f"<b>Mean Fit Error:</b> {err_str}<br>"
            + "<br>".join(hm_lines)
        )

    def generate_selected_plots(self):
        """Generates the active dropdown plot plus any checked plots across the selected channel(s)."""
        active_view = self.combo_view_mode.currentData()

        want_int = self.chk_plot_heatmap_intensity.isChecked() or (active_view == 'intensity')
        want_diff = self.chk_plot_heatmap_diff.isChecked() or (active_view == 'diffusion')
        want_jump = self.chk_plot_heatmap_jump.isChecked() or (active_view == 'jump')
        want_prof = self.chk_plot_line_profiles.isChecked() or (active_view == 'profiles')
        want_morph = self.chk_plot_morphology.isChecked() or (active_view == 'morphology')

        # Ensure the active view checkbox reflects what was generated
        chk_map = {
            'intensity': self.chk_plot_heatmap_intensity,
            'diffusion': self.chk_plot_heatmap_diff,
            'jump': self.chk_plot_heatmap_jump,
            'profiles': self.chk_plot_line_profiles,
            'morphology': self.chk_plot_morphology,
        }
        if active_view in chk_map and not chk_map[active_view].isChecked():
            chk_map[active_view].setChecked(True)

        progress = QProgressDialog("Scanning preprocessing results & cell_* folders...", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(150)
        progress.show()
        QApplication.processEvents()

        try:
            records = self._scan_and_parse_cells(progress_dlg=progress)
            if not records:
                progress.close()
                self._has_generated = False
                self.lbl_cell_count_banner.setText("Detected cells: 0")
                self.lbl_analyser_stats.setText("No valid cell records found in active selection.")
                self._draw_placeholder("No valid cell_fitting_parameters*.csv or cell_* folders found in the selected fields.")
                return

            grid_info = self._build_canonical_grid(records)
            channels = self._active_channels_list()

            int_grids = {}
            if want_int:
                for ch in channels:
                    if progress.wasCanceled():
                        break
                    int_grids[ch] = self._compute_intensity_heatmap(records, grid_info, ch, progress_dlg=progress)

            diff_grids = {}
            jump_grids = {}
            if want_diff or want_jump:
                for ch in channels:
                    if progress.wasCanceled():
                        break
                    d_g, n_d, j_g, n_j = self._compute_track_heatmaps(
                        records, grid_info, ch, want_diff, want_jump, progress_dlg=progress
                    )
                    if want_diff:
                        diff_grids[ch] = (d_g, n_d)
                    if want_jump:
                        jump_grids[ch] = (j_g, n_j)

            x_prof, prof_dict = (None, {})
            if want_prof and not progress.wasCanceled():
                x_prof, prof_dict = self._collect_channel_line_profiles(records, channels, progress_dlg=progress)

            self._cached_results = {
                'records': records,
                'grid_info': grid_info,
                'want_int': want_int,
                'int_grids': int_grids,
                'want_diff': want_diff,
                'diff_grids': diff_grids,
                'want_jump': want_jump,
                'jump_grids': jump_grids,
                'want_prof': want_prof,
                'x_prof': x_prof,
                'prof_dict': prof_dict,
                'want_morph': want_morph,
            }
            self._has_generated = True
            self._clear_stale_flags(['records', 'intensity', 'diffusion', 'jump', 'profiles', 'morphology'])

            self._update_stats_label()
            self._redraw_from_cache()
        finally:
            progress.close()

    def regenerate_single_plot(self, plot_key: str):
        """Regenerates ONLY the requested plot (`intensity`, `diffusion`, `jump`, `profiles`, or `morphology`)
        using current channel and filter settings, preserving all other cached plots.
        """
        chk_map = {
            'intensity': self.chk_plot_heatmap_intensity,
            'diffusion': self.chk_plot_heatmap_diff,
            'jump': self.chk_plot_heatmap_jump,
            'profiles': self.chk_plot_line_profiles,
            'morphology': self.chk_plot_morphology,
        }
        if plot_key in chk_map and not chk_map[plot_key].isChecked():
            chk_map[plot_key].setChecked(True)

        # Switch dropdown to the regenerated plot so the user immediately sees the updated output
        idx = self.combo_view_mode.findData(plot_key)
        if idx >= 0 and self.combo_view_mode.currentIndex() != idx:
            self.combo_view_mode.blockSignals(True)
            self.combo_view_mode.setCurrentIndex(idx)
            self.combo_view_mode.blockSignals(False)

        progress = QProgressDialog(f"Regenerating {plot_key} plot...", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(150)
        progress.show()
        QApplication.processEvents()

        try:
            need_rescan = (
                not self._has_generated
                or not self._cached_results
                or not self.cell_records
                or self._stale_flags.get('records', False)
            )

            if need_rescan:
                records = self._scan_and_parse_cells(progress_dlg=progress)
                if not records:
                    progress.close()
                    self._has_generated = False
                    self.lbl_cell_count_banner.setText("Detected cells: 0")
                    self.lbl_analyser_stats.setText("No valid cell records found in active selection.")
                    self._draw_placeholder("No valid cell_fitting_parameters*.csv or cell_* folders found in the selected fields.")
                    return

                if self._has_generated:
                    other_keys = [k for k in ('intensity', 'diffusion', 'jump', 'profiles', 'morphology') if k != plot_key]
                    for k in other_keys:
                        self._stale_flags[k] = True
                self._stale_flags['records'] = False
            else:
                records = self._cached_results['records']

            grid_info = self._build_canonical_grid(records)
            channels = self._active_channels_list()

            if not self._cached_results:
                self._cached_results = {
                    'records': records,
                    'grid_info': grid_info,
                    'want_int': self.chk_plot_heatmap_intensity.isChecked(),
                    'int_grids': {},
                    'want_diff': self.chk_plot_heatmap_diff.isChecked(),
                    'diff_grids': {},
                    'want_jump': self.chk_plot_heatmap_jump.isChecked(),
                    'jump_grids': {},
                    'want_prof': self.chk_plot_line_profiles.isChecked(),
                    'x_prof': None,
                    'prof_dict': {},
                    'want_morph': self.chk_plot_morphology.isChecked(),
                }
            else:
                self._cached_results['records'] = records
                self._cached_results['grid_info'] = grid_info
                self._cached_results['want_int'] = self.chk_plot_heatmap_intensity.isChecked()
                self._cached_results['want_diff'] = self.chk_plot_heatmap_diff.isChecked()
                self._cached_results['want_jump'] = self.chk_plot_heatmap_jump.isChecked()
                self._cached_results['want_prof'] = self.chk_plot_line_profiles.isChecked()
                self._cached_results['want_morph'] = self.chk_plot_morphology.isChecked()

            if plot_key == 'intensity' and not progress.wasCanceled():
                int_grids = {}
                for ch in channels:
                    if progress.wasCanceled():
                        break
                    int_grids[ch] = self._compute_intensity_heatmap(records, grid_info, ch, progress_dlg=progress)
                self._cached_results['int_grids'] = int_grids

            elif plot_key == 'diffusion' and not progress.wasCanceled():
                diff_grids = {}
                for ch in channels:
                    if progress.wasCanceled():
                        break
                    d_g, n_d, _, _ = self._compute_track_heatmaps(
                        records, grid_info, ch, want_diff=True, want_jump=False, progress_dlg=progress
                    )
                    diff_grids[ch] = (d_g, n_d)
                self._cached_results['diff_grids'] = diff_grids

            elif plot_key == 'jump' and not progress.wasCanceled():
                jump_grids = {}
                for ch in channels:
                    if progress.wasCanceled():
                        break
                    _, _, j_g, n_j = self._compute_track_heatmaps(
                        records, grid_info, ch, want_diff=False, want_jump=True, progress_dlg=progress
                    )
                    jump_grids[ch] = (j_g, n_j)
                self._cached_results['jump_grids'] = jump_grids

            elif plot_key == 'profiles' and not progress.wasCanceled():
                x_prof, prof_dict = self._collect_channel_line_profiles(records, channels, progress_dlg=progress)
                self._cached_results['x_prof'] = x_prof
                self._cached_results['prof_dict'] = prof_dict

            elif plot_key == 'morphology':
                pass

            self._has_generated = True
            self._clear_stale_flags([plot_key])
            self._update_stats_label()
            self._redraw_from_cache()
        finally:
            progress.close()

    def _redraw_from_cache(self):
        """Renders ONLY the single plot view currently selected in `combo_view_mode`."""
        if not self._has_generated or not self._cached_results:
            return

        cache = self._cached_results
        view_mode = self.combo_view_mode.currentData() or "intensity"
        channels = self._active_channels_list()
        include_half = self.chk_half_plots.isChecked()

        fig = self.canvas.fig
        fig.clf()
        fig.set_layout_engine('constrained')
        try:
            fig.get_layout_engine().set(w_pad=0.06, h_pad=0.08, hspace=0.06, wspace=0.06)
        except Exception:
            pass

        if view_mode in ("intensity", "diffusion", "jump"):
            grid_key_map = {
                "intensity": ("want_int", "int_grids"),
                "diffusion": ("want_diff", "diff_grids"),
                "jump": ("want_jump", "jump_grids"),
            }
            want_flag, grid_dict_key = grid_key_map[view_mode]
            if not cache.get(want_flag):
                self._draw_placeholder(
                    f"Plot '{self.combo_view_mode.currentText()}' has not been generated yet.\n"
                    "Click 'Generate Selected Plot(s)' or '↻ Regen' to compute it."
                )
                return

            hm_map = cache.get(grid_dict_key, {})
            n_cols = len(channels)
            n_rows = 2 if include_half else 1
            total_width_in = max(6.8, n_cols * 4.8)
            total_height_in = 4.6 * n_rows
            dpi = fig.dpi or 100
            self.canvas.setMinimumSize(int(total_width_in * dpi), int(total_height_in * dpi))
            fig.set_size_inches(total_width_in, total_height_in, forward=True)

            gs = fig.add_gridspec(n_rows, n_cols)
            for c_idx, ch_name in enumerate(channels):
                hm_grid, n_cells_used = hm_map.get(ch_name, (None, 0))
                ax_full = fig.add_subplot(gs[0, c_idx])
                self._render_single_heatmap(
                    ax_full, view_mode, hm_grid, n_cells_used, cache['grid_info'], ch_name, half_mode=False
                )
                if include_half:
                    ax_half = fig.add_subplot(gs[1, c_idx])
                    self._render_single_heatmap(
                        ax_half, view_mode, hm_grid, n_cells_used, cache['grid_info'], ch_name, half_mode=True
                    )

        elif view_mode == "profiles":
            if not cache.get('want_prof'):
                self._draw_placeholder(
                    "Line Profiles have not been generated yet.\n"
                    "Click 'Generate Selected Plot(s)' or '↻ Regen' to compute them."
                )
                return

            n_cols = len(channels)
            n_rows = 2 if include_half else 1
            total_width_in = max(6.8, n_cols * 4.8)
            total_height_in = 4.4 * n_rows
            dpi = fig.dpi or 100
            self.canvas.setMinimumSize(int(total_width_in * dpi), int(total_height_in * dpi))
            fig.set_size_inches(total_width_in, total_height_in, forward=True)

            gs = fig.add_gridspec(n_rows, n_cols)
            self._render_line_profiles(
                fig, gs, cache.get('x_prof'), cache.get('prof_dict', {}), channels, row_idx=0, half_mode=False
            )
            if include_half:
                self._render_line_profiles(
                    fig, gs, cache.get('x_prof'), cache.get('prof_dict', {}), channels, row_idx=1, half_mode=True
                )

        elif view_mode == "morphology":
            if not cache.get('want_morph'):
                self._draw_placeholder(
                    "Cell Features have not been generated yet.\n"
                    "Click 'Generate Selected Plot(s)' or '↻ Regen' to compute them."
                )
                return

            total_width_in = 12.6
            total_height_in = 7.2
            dpi = fig.dpi or 100
            self.canvas.setMinimumSize(int(total_width_in * dpi), int(total_height_in * dpi))
            fig.set_size_inches(total_width_in, total_height_in, forward=True)

            gs = fig.add_gridspec(2, 3)
            self._render_morphology_plots(fig, gs, cache['records'], channels)

        else:
            self._draw_placeholder("Select a valid plot view from the dropdown.")
            return

        self.canvas.draw()

    @staticmethod
    def _fold_grid_half(grid: np.ndarray) -> np.ndarray:
        """Cuts a 2D (ny, nx) heatmap grid at the longitudinal midpoint, flips the right half,
        and combines (nanmean) it with the left half.
        """
        if grid is None or grid.ndim != 2 or grid.shape[1] < 2:
            return grid
        nx = grid.shape[1]
        half_nx = (nx + 1) // 2
        left_half = grid[:, :half_nx]
        right_half_flipped = grid[:, -half_nx:][:, ::-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(np.stack([left_half, right_half_flipped], axis=0), axis=0)

    def _render_single_heatmap(
        self,
        ax,
        hm_kind: str,
        grid: np.ndarray,
        n_cells: int,
        grid_info: dict,
        channel: str,
        half_mode: bool = False,
    ):
        mean_L = grid_info['mean_len_um']
        mean_W = grid_info['mean_wid_um']
        px_um = grid_info['pixel_size']
        bin_um = grid_info['bin_step_um']

        base_titles = {
            'intensity': f"Frame-Avg Intensity [{channel}]",
            'diffusion': f"Diffusion Coeff D [{channel}]",
            'jump': f"Jump Distance r [{channel}]",
        }
        title_str = f"Half {base_titles[hm_kind]}" if half_mode else base_titles[hm_kind]

        cbar_labels = {
            'intensity': "Norm. Intensity" if self.chk_norm_cell_intensity.isChecked() else "Intensity (a.u.)",
            'diffusion': r"$D\ [\mu\mathrm{m}^2/\mathrm{s}]$",
            'jump': r"Jump $r\ [\mu\mathrm{m}]$",
        }

        if grid is None or not np.any(np.isfinite(grid)):
            ax.text(
                0.5, 0.5,
                f"No data for {title_str}\n(Click '↻ Regen' if channel was newly selected)",
                ha='center', va='center', transform=ax.transAxes, fontsize=9, color='#555'
            )
            ax.set_title(title_str, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        plot_grid = grid.copy()
        if self.chk_mask_outside_capsule.isChecked():
            plot_grid = np.where(grid_info['capsule_mask'], plot_grid, np.nan)

        if half_mode:
            plot_grid = self._fold_grid_half(plot_grid)

        cmap = matplotlib.colormaps.get_cmap(self.combo_cmap.currentText()).copy()
        cmap.set_bad(color='#111111' if self.chk_show_grid.isChecked() else '#f4f4f4')

        x_max = 0.0 if half_mode else 0.5 * mean_L
        extent = [-0.5 * mean_L, x_max, -0.5 * mean_W, 0.5 * mean_W]
        im = ax.imshow(
            plot_grid,
            origin='lower',
            extent=extent,
            aspect='equal',
            cmap=cmap,
            interpolation='nearest',
        )

        n_x_bins = plot_grid.shape[1]
        if self.chk_show_grid.isChecked():
            x_lines = np.linspace(-0.5 * mean_L, x_max, n_x_bins + 1)
            y_lines = np.linspace(-0.5 * mean_W, 0.5 * mean_W, grid_info['ny'] + 1)
            for xl in x_lines:
                ax.axvline(xl, color='black', linewidth=0.35, alpha=0.45)
            for yl in y_lines:
                ax.axhline(yl, color='black', linewidth=0.35, alpha=0.45)

        r_cap = 0.5 * mean_W
        cyl_len = max(0.0, mean_L - mean_W)
        theta_left = np.linspace(0.5 * np.pi, 1.5 * np.pi, 60)
        if half_mode:
            cap_x = np.concatenate([
                -0.5 * cyl_len + r_cap * np.cos(theta_left),
                [0.0, 0.0, -0.5 * cyl_len]
            ])
            cap_y = np.concatenate([
                r_cap * np.sin(theta_left),
                [-r_cap, r_cap, r_cap]
            ])
        else:
            theta_right = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 60)
            cap_x = np.concatenate([
                -0.5 * cyl_len + r_cap * np.cos(theta_left),
                0.5 * cyl_len + r_cap * np.cos(theta_right),
                [-0.5 * cyl_len]
            ])
            cap_y = np.concatenate([
                r_cap * np.sin(theta_left),
                r_cap * np.sin(theta_right),
                [r_cap]
            ])
        ax.plot(cap_x, cap_y, color='white', linewidth=1.6)

        cbar = self.canvas.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(cbar_labels[hm_kind], fontsize=8)
        cbar.ax.tick_params(labelsize=8)

        ax.set_title(
            f"{title_str} (n={n_cells} cells)\n"
            f"px={px_um:.3f} μm | bin={bin_um:.3f} μm ({n_x_bins}×{grid_info['ny']})",
            fontsize=9.5
        )
        ax.set_xlabel(
            r"Half Cell Long Axis (Pole $\rightarrow$ Midcell) $[\mu\mathrm{m}]$"
            if half_mode else r"Cell Long Axis $[\mu\mathrm{m}]$",
            fontsize=9
        )
        ax.set_ylabel(r"Cell Short Axis $[\mu\mathrm{m}]$", fontsize=9)
        ax.tick_params(labelsize=8)

    @staticmethod
    def _fold_profiles_half(x_prof: np.ndarray, mat: np.ndarray):
        """Cuts a 2D (n_cells, n_pts) profile matrix at the midpoint, flips the second half,
        and combines (nanmean) it with the first half onto [0.0, 0.5].
        """
        n_pts = mat.shape[1]
        if n_pts < 2:
            return x_prof, mat
        half_n = (n_pts + 1) // 2
        left_half = mat[:, :half_n]
        right_half_flipped = mat[:, -half_n:][:, ::-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            folded_mat = np.nanmean(np.stack([left_half, right_half_flipped], axis=0), axis=0)
        x_half = np.linspace(0.0, 0.5, half_n)
        return x_half, folded_mat

    def _render_line_profiles(
        self,
        fig,
        gs,
        x_prof,
        prof_dict,
        channels,
        row_idx: int = 0,
        half_mode: bool = False,
    ):
        if x_prof is None or not prof_dict:
            if row_idx == 0:
                ax = fig.add_subplot(gs[0, 0])
                ax.text(0.5, 0.5, "No line profiles found.", ha='center', va='center', transform=ax.transAxes)
            return

        norm_mode = self.combo_profile_norm.currentText()
        is_long = self.combo_profile_axis.currentText().startswith("Longitudinal")
        if half_mode:
            axis_lbl = (
                "Half Normalized Cell Length (0 = Pole, 0.5 = Midcell)"
                if is_long else "Half Normalized Cell Width (0 = Edge, 0.5 = Center)"
            )
        else:
            axis_lbl = "Normalized Cell Length" if is_long else "Normalized Cell Width"

        color_map = {
            "Left Channel (L)": "#008080",
            "Right Channel (R)": "#D81B60",
        }

        for c_idx, ch_name in enumerate(channels):
            color = color_map.get(ch_name, "#008080")
            ax = fig.add_subplot(gs[row_idx, c_idx])
            curves = prof_dict.get(ch_name, [])
            title_prefix = "Half Line Profiles (Folded)" if half_mode else "Line Profiles"

            if not curves:
                ax.text(
                    0.5, 0.5,
                    f"No line profiles for {ch_name}\n(Click '↻ Regen' if channel was newly selected)",
                    ha='center', va='center', transform=ax.transAxes, fontsize=9, color='#555'
                )
                ax.set_title(f"{title_prefix} — {ch_name}", fontsize=10)
                continue

            mat = np.vstack(curves)
            x_plot = x_prof
            if half_mode:
                x_plot, mat = self._fold_profiles_half(x_prof, mat)

            if norm_mode == "Mean-Normalized (I / <I>)":
                means = np.nanmean(mat, axis=1, keepdims=True)
                means = np.where(means > 0, means, 1.0)
                mat = mat / means
                y_label = "Normalized Intensity (I / <I>)"
            elif norm_mode == "Min-Max Normalized [0–1]":
                mins = np.nanmin(mat, axis=1, keepdims=True)
                spans = np.nanmax(mat, axis=1, keepdims=True) - mins
                spans = np.where(spans > 0, spans, 1.0)
                mat = (mat - mins) / spans
                y_label = "Min-Max Normalized Intensity"
            else:
                y_label = "Intensity (a.u.)"

            for row in mat:
                ax.plot(x_plot, row, color=color, alpha=0.18, linewidth=0.8)

            mean_curve = np.nanmean(mat, axis=0)
            sd_curve = np.nanstd(mat, axis=0)
            ax.plot(x_plot, mean_curve, color=color, linewidth=2.3, label=f"Mean (n={mat.shape[0]} cells)")
            ax.fill_between(x_plot, mean_curve - sd_curve, mean_curve + sd_curve, color=color, alpha=0.22, label="±1 SD")

            if half_mode:
                ax.set_xlim(0.0, 0.5)

            ax.set_title(f"{title_prefix} — {ch_name}", fontsize=10)
            ax.set_xlabel(axis_lbl, fontsize=9)
            ax.set_ylabel(y_label, fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.tick_params(labelsize=8)

    def _extract_feature_arrays_for_channel(self, records, ch_name: str):
        """Extracts Length, Width, Angle, Area, and Fit Error arrays for a specific channel from cell records."""
        vals_by_key = {
            'length_um': [],
            'width_um': [],
            'angle_deg': [],
            'area_um2': [],
            'fit_error_nm': [],
        }
        for r in records:
            ch_fits = r.get('channel_fits', {})
            if ch_name in ch_fits:
                src = ch_fits[ch_name]
            elif ch_fits and ch_name == "Right Channel (R)":
                # Skip L-only entries when explicitly inspecting R-only channel_fits
                continue
            else:
                src = r

            for k in vals_by_key:
                v = src.get(k, np.nan)
                if v is not None and np.isfinite(v):
                    vals_by_key[k].append(float(v))

        return {k: np.asarray(v, dtype=float) for k, v in vals_by_key.items()}

    def _render_morphology_plots(self, fig, gs, records, channels):
        """
        Renders 5 Cell Feature histograms in a 2x3 grid:
          (0,0) Cell Lengths [um]
          (0,1) Cell Widths [um]
          (0,2) Cell Orientation Angles [deg]
          (1,0) Cell Areas [um^2]
          (1,1) Cell Fitting Errors [nm]
          (1,2) Summary table / text panel
        Supports Left Channel (L), Right Channel (R), or Both Channels (L & R) in the same figure.
        """
        channel_data = {}
        for ch in channels:
            arrs = self._extract_feature_arrays_for_channel(records, ch)
            if any(a.size > 0 for a in arrs.values()):
                channel_data[ch] = arrs

        if not channel_data:
            # Fallback to record-level values if channel-tagged entries were empty
            channel_data[channels[0]] = {
                'length_um': np.array([r['length_um'] for r in records if np.isfinite(r.get('length_um', np.nan))]),
                'width_um': np.array([r['width_um'] for r in records if np.isfinite(r.get('width_um', np.nan))]),
                'angle_deg': np.array([r['angle_deg'] for r in records if np.isfinite(r.get('angle_deg', np.nan))]),
                'area_um2': np.array([r['area_um2'] for r in records if np.isfinite(r.get('area_um2', np.nan))]),
                'fit_error_nm': np.array([r['fit_error_nm'] for r in records if np.isfinite(r.get('fit_error_nm', np.nan))]),
            }

        panel_specs = [
            (0, 0, 'length_um', "Cell Lengths", r"Length $[\mu\mathrm{m}]$", "#008080"),
            (0, 1, 'width_um', "Cell Widths", r"Width $[\mu\mathrm{m}]$", "#2e6da4"),
            (0, 2, 'angle_deg', "Cell Orientation Angles", r"Angle $[^\circ]$", "#6f42c1"),
            (1, 0, 'area_um2', "Cell Areas", r"Area $[\mu\mathrm{m}^2]$", "#2b9348"),
            (1, 1, 'fit_error_nm', "Cell Fitting Error", r"Fitting Error $[\mathrm{nm}]$", "#d9534f"),
        ]

        chan_colors = {
            "Left Channel (L)": "#008080",
            "Right Channel (R)": "#D81B60",
        }
        multi_chan_active = len(channel_data) > 1

        for r_idx, c_idx, key, title, xlabel, single_color in panel_specs:
            ax = fig.add_subplot(gs[r_idx, c_idx])

            all_valid = [arrs[key][np.isfinite(arrs[key])] for arrs in channel_data.values() if arrs[key].size > 0]
            all_valid = [a for a in all_valid if a.size > 0]
            if not all_valid:
                ax.text(0.5, 0.5, f"No {title} data available", ha='center', va='center', transform=ax.transAxes, fontsize=9, color='#555')
                ax.set_title(title, fontsize=10)
                continue

            combined_vals = np.concatenate(all_valid)
            bins = min(25, max(6, int(np.sqrt(combined_vals.size) * 1.5)))
            bin_edges = np.histogram_bin_edges(combined_vals, bins=bins)
            bw = float(np.mean(np.diff(bin_edges))) if len(bin_edges) > 1 else 1.0

            for ch_name, arrs in channel_data.items():
                vals = arrs[key]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue

                color = chan_colors.get(ch_name, single_color) if multi_chan_active else single_color
                short_lbl = "L" if "Left" in ch_name else "R"
                prefix = f"[{short_lbl}] " if multi_chan_active else ""

                ax.hist(
                    vals,
                    bins=bin_edges,
                    color=color,
                    alpha=0.38 if multi_chan_active else 0.45,
                    edgecolor='black',
                    linewidth=0.5,
                )

                if vals.size > 2 and np.std(vals) > 1e-6:
                    x_g = np.linspace(bin_edges[0], bin_edges[-1], 200)
                    kde = gaussian_kde(vals)
                    ax.plot(x_g, kde(x_g) * vals.size * bw, color=color if multi_chan_active else 'black', linewidth=1.6)

                mean_v = float(np.mean(vals))
                sd_v = float(np.std(vals))
                ax.axvline(
                    mean_v,
                    color=color if multi_chan_active else 'darkred',
                    linestyle='--',
                    linewidth=1.4,
                    label=f"{prefix}Mean: {mean_v:.2f} ± {sd_v:.2f} (n={vals.size})"
                )

            total_n = int(combined_vals.size)
            ch_tag = "L & R" if multi_chan_active else ("L" if "Left" in next(iter(channel_data)) else "R")
            ax.set_title(f"{title} [{ch_tag}] (n={total_n})", fontsize=10)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("Cell Count", fontsize=9)
            ax.legend(loc='upper right', fontsize=7.5)
            ax.tick_params(labelsize=8)

        # Panel (1, 2): Parameter Summary Card
        ax_info = fig.add_subplot(gs[1, 2])
        ax_info.axis('off')
        summary_lines = ["Cell Fitting Parameter Summary", "────────────────────────"]
        for ch_name, arrs in channel_data.items():
            short_lbl = "Left Channel (L)" if "Left" in ch_name else "Right Channel (R)"
            summary_lines.append(f"\n{short_lbl}:")
            for k, lbl, unit in [
                ('length_um', 'Length', 'μm'),
                ('width_um', 'Width', 'μm'),
                ('area_um2', 'Area', 'μm²'),
                ('angle_deg', 'Angle', '°'),
                ('fit_error_nm', 'Fit Error', 'nm'),
            ]:
                v = arrs[k][np.isfinite(arrs[k])]
                if v.size > 0:
                    summary_lines.append(f"  • {lbl}: {np.mean(v):.2f} ± {np.std(v):.2f} {unit} (n={v.size})")
                else:
                    summary_lines.append(f"  • {lbl}: N/A")

        ax_info.text(
            0.04, 0.96, "\n".join(summary_lines),
            transform=ax_info.transAxes,
            va='top', ha='left', fontsize=8.5, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#cccccc')
        )

    def export_cell_summary_csv(self):
        if not self.cell_records:
            self._scan_and_parse_cells()
        if not self.cell_records:
            QMessageBox.warning(self, "No Cells", "No valid cell records found to export.")
            return

        root_dir = getattr(self.parent, 'root_dir', ".") if self.parent is not None else "."
        default_path = os.path.join(root_dir or ".", "cell_analyser_summary.csv")
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Cell Summary CSV", default_path, "CSV Files (*.csv);;All Files (*)"
        )
        if not filepath:
            return

        export_rows = [
            {
                'sample': r['sample'],
                'field': r['field'],
                'cell_folder': r['cell_folder'],
                'cell_num': r['cell_num'],
                'channel': r.get('channel_code', 'L'),
                'mask_source': r.get('mask_source', ''),
                'centroid_x_px': r['cx'],
                'centroid_y_px': r['cy'],
                'length_px': r['length_px'],
                'width_px': r['width_px'],
                'length_um': r['length_um'],
                'width_um': r['width_um'],
                'area_px': r.get('area_px', np.nan),
                'area_um2': r.get('area_um2', np.nan),
                'angle_deg': r['angle_deg'],
                'fit_error_nm': r.get('fit_error_nm', np.nan),
            }
            for r in self.cell_records
        ]
        pd.DataFrame(export_rows).to_csv(filepath, index=False)
        QMessageBox.information(self, "Export Complete", f"Exported {len(export_rows)} cell records to:\n{filepath}")