import sys
import os
import glob
import re
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PIL import Image
from scipy.stats import gaussian_kde

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QComboBox,
    QSplitter, QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox,
    QCheckBox, QScrollArea
)
from PyQt6.QtCore import Qt


class _NoScrollComboBox(QComboBox):
    """QComboBox that ignores mouse wheel scrolling unless explicitly focused by clicking."""
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _NoScrollSpinBox(QSpinBox):
    """QSpinBox that ignores mouse wheel scrolling unless explicitly focused by clicking."""
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _NoScrollDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that ignores mouse wheel scrolling unless explicitly focused by clicking."""
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _MplCanvas(FigureCanvasQTAgg):
    """Reusable Matplotlib Canvas widget."""
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class SingleFieldInspectorTab(QWidget):
    """
    Single Field Inspector tab widget and callable standalone inspector.

    Can be embedded inside ResultsViewerTab (`SingleFieldInspectorTab(viewer)`)
    or invoked directly as a standalone widget/script (`SingleFieldInspectorTab(root_dir="...")`
    or `python single_field_inspector_tab.py [root_dir]`).
    """

    PLOT_CHOICES = [
        ("Microscopy Image / Static Output", "IMAGE"),
        ("Diffusion Coefficient (D)", "LIVE_DIFF"),
        ("Jump Distance (N frames)", "LIVE_JUMP"),
        ("Stoichiometry Scatter (N vs D)", "LIVE_SCATTER"),
        ("Trajectory Intensity Traces", "LIVE_TRACES"),
        ("Stoichiometry Histogram", "LIVE_STOICH_HIST"),
        ("Spot Intensity vs Frame", "LIVE_SPOTS"),
        ("All Live Field Plots (2x3 Grid)", "LIVE_ALL"),
        ("None (Disable Plotting)", "NONE"),
    ]

    def __init__(self, parent=None, root_dir=None):
        # Support calling SingleFieldInspectorTab("path/to/dir") directly
        if isinstance(parent, (str, os.PathLike)):
            root_dir = os.fspath(parent)
            parent = None

        super().__init__(parent)

        # Determine whether attached to ResultsViewerTab or running standalone
        self._is_embedded = (
            parent is not None
            and hasattr(parent, "trajectories_df")
            and hasattr(parent, "get_filtered_summary_df")
        )
        self.viewer = parent if self._is_embedded else None

        # Standalone state (used only when not embedded in ResultsViewerTab)
        self.root_dir = os.path.abspath(root_dir) if root_dir else ""
        self.trajectories_df = pd.DataFrame()
        self.diffusion_summary_df = pd.DataFrame()
        self.field_dirs_map = {}

        self._build_options_group()
        self._build_main_ui()

        if not self._is_embedded and self.root_dir and os.path.exists(self.root_dir):
            self.scan_and_load_directory()

    # ------------------------------------------------------------------
    # UI CONSTRUCTION
    # ------------------------------------------------------------------

    def _build_options_group(self):
        """Creates the sidebar QGroupBox ('Field Inspector Options') for ResultsViewerTab or standalone sidebar."""
        self.options_group = QGroupBox("Field Inspector Options")
        layout = QFormLayout()
        layout.setVerticalSpacing(6)

        self.combo_plot_selector_sidebar = _NoScrollComboBox()
        self.combo_plot_selector_sidebar.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        for label, code in self.PLOT_CHOICES:
            self.combo_plot_selector_sidebar.addItem(label, userData=code)
        self.combo_plot_selector_sidebar.currentIndexChanged.connect(
            self._on_sidebar_plot_selector_changed
        )

        self.chk_overlay_field_tracks = QCheckBox("Overlay Trajectories on Microscopy Images")
        self.chk_overlay_field_tracks.setChecked(True)
        self.chk_overlay_field_tracks.stateChanged.connect(self.plot)

        layout.addRow("Plot to Produce:", self.combo_plot_selector_sidebar)
        layout.addRow(self.chk_overlay_field_tracks)
        self.options_group.setLayout(layout)
        self.options_group.setVisible(False)

    def _build_main_ui(self):
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        plot_container = QWidget()
        layout_field = QVBoxLayout(plot_container)

        # Top bar with Plot Selector dropdown & Static Image File dropdown
        field_top_bar = QHBoxLayout()
        field_top_bar.setSpacing(8)

        lbl_plot_sel = QLabel("Plot Selector:")
        lbl_plot_sel.setStyleSheet("font-weight: bold;")

        self.combo_plot_selector = _NoScrollComboBox()
        self.combo_plot_selector.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        for label, code in self.PLOT_CHOICES:
            self.combo_plot_selector.addItem(label, userData=code)
        self.combo_plot_selector.currentIndexChanged.connect(
            self._on_top_plot_selector_changed
        )

        self.lbl_field_image = QLabel("Image / Output File:")
        self.lbl_field_image.setStyleSheet("font-weight: bold;")

        self.combo_field_image = _NoScrollComboBox()
        self.combo_field_image.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.combo_field_image.currentIndexChanged.connect(self.plot)

        field_top_bar.addWidget(lbl_plot_sel)
        field_top_bar.addWidget(self.combo_plot_selector, 1)
        field_top_bar.addWidget(self.lbl_field_image)
        field_top_bar.addWidget(self.combo_field_image, 1)

        self.field_canvas = _MplCanvas(self)
        self.field_toolbar = NavigationToolbar(self.field_canvas, self)

        layout_field.addLayout(field_top_bar)
        layout_field.addWidget(self.field_toolbar)
        layout_field.addWidget(self.field_canvas)

        if self._is_embedded:
            root_layout.addWidget(plot_container)
        else:
            # Standalone mode: add a right-hand control sidebar so the script is fully usable standalone
            self.options_group.setVisible(True)
            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.addWidget(plot_container)
            splitter.addWidget(self._build_standalone_sidebar())
            splitter.setSizes([960, 320])
            root_layout.addWidget(splitter)

    def _build_standalone_sidebar(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        sidebar = QWidget()
        s_layout = QVBoxLayout(sidebar)
        s_layout.setContentsMargins(8, 8, 8, 8)
        s_layout.setSpacing(10)

        # Directory box
        dir_group = QGroupBox("Experiment Directory")
        dir_form = QFormLayout()
        self.lbl_root_dir = QLabel(self.root_dir or "No directory selected")
        self.lbl_root_dir.setWordWrap(True)
        self.lbl_root_dir.setStyleSheet("font-size: 10px; color: #444;")
        btn_open = QPushButton("Open Directory...")
        btn_open.clicked.connect(self.select_new_directory)
        btn_rescan = QPushButton("Rescan Directory")
        btn_rescan.clicked.connect(self.scan_and_load_directory)
        dir_form.addRow("Target:", self.lbl_root_dir)
        dir_form.addRow(btn_open)
        dir_form.addRow(btn_rescan)
        dir_group.setLayout(dir_form)
        s_layout.addWidget(dir_group)

        # Field & Channel box
        sel_group = QGroupBox("Sample, Channel & Field")
        sel_form = QFormLayout()
        self.sample_selector = _NoScrollComboBox()
        self.sample_selector.currentIndexChanged.connect(self._on_standalone_sample_changed)
        self.channel_selector = _NoScrollComboBox()
        self.channel_selector.addItems(["Both Channels (Overlay)", "Left Channel (L)", "Right Channel (R)"])
        self.channel_selector.currentIndexChanged.connect(self.plot)
        self.field_selector = _NoScrollComboBox()
        self.field_selector.currentIndexChanged.connect(self._on_standalone_field_changed)
        sel_form.addRow("Sample:", self.sample_selector)
        sel_form.addRow("Channel:", self.channel_selector)
        sel_form.addRow("Field:", self.field_selector)
        sel_group.setLayout(sel_form)
        s_layout.addWidget(sel_group)

        # Basic Acquisition & Filter box
        param_group = QGroupBox("Acquisition & Filters")
        param_form = QFormLayout()
        self.spin_pixel_size = _NoScrollDoubleSpinBox()
        self.spin_pixel_size.setDecimals(4)
        self.spin_pixel_size.setRange(0.0001, 10.0)
        self.spin_pixel_size.setValue(0.120)
        self.spin_pixel_size.setSuffix(" μm/px")

        self.spin_exposure = _NoScrollDoubleSpinBox()
        self.spin_exposure.setDecimals(3)
        self.spin_exposure.setRange(0.001, 10000.0)
        self.spin_exposure.setValue(5.0)
        self.spin_exposure.setSuffix(" ms")

        self.chk_alex = QCheckBox("ALEX Imaging")
        self.chk_alex.setChecked(False)

        self.spin_min_len = _NoScrollSpinBox()
        self.spin_min_len.setRange(2, 100)
        self.spin_min_len.setValue(4)

        self.spin_isingle_L = _NoScrollDoubleSpinBox()
        self.spin_isingle_L.setDecimals(1)
        self.spin_isingle_L.setRange(1.0, 1000000.0)
        self.spin_isingle_L.setValue(10000.0)
        self.spin_isingle_L.setSuffix(" a.u.")

        self.spin_isingle_R = _NoScrollDoubleSpinBox()
        self.spin_isingle_R.setDecimals(1)
        self.spin_isingle_R.setRange(1.0, 1000000.0)
        self.spin_isingle_R.setValue(10000.0)
        self.spin_isingle_R.setSuffix(" a.u.")

        btn_recalc = QPushButton("Recalculate & Plot")
        btn_recalc.setStyleSheet("font-weight: bold; background-color: #008080; color: white; padding: 5px;")
        btn_recalc.clicked.connect(self._recalculate_standalone_summary)

        param_form.addRow("Pixel Size:", self.spin_pixel_size)
        param_form.addRow("Frame Interval:", self.spin_exposure)
        param_form.addRow(self.chk_alex)
        param_form.addRow("Min Track Length:", self.spin_min_len)
        param_form.addRow("iSingle (L):", self.spin_isingle_L)
        param_form.addRow("iSingle (R):", self.spin_isingle_R)
        param_form.addRow(btn_recalc)
        param_group.setLayout(param_form)
        s_layout.addWidget(param_group)

        # Inspector options group
        s_layout.addWidget(self.options_group)
        s_layout.addStretch()

        scroll.setWidget(sidebar)
        return scroll

    # ------------------------------------------------------------------
    # DROPDOWN SYNCHRONIZATION
    # ------------------------------------------------------------------

    def _on_top_plot_selector_changed(self, index):
        self.combo_plot_selector_sidebar.blockSignals(True)
        self.combo_plot_selector_sidebar.setCurrentIndex(index)
        self.combo_plot_selector_sidebar.blockSignals(False)
        self._update_image_selector_enabled_state()
        self.plot()

    def _on_sidebar_plot_selector_changed(self, index):
        self.combo_plot_selector.blockSignals(True)
        self.combo_plot_selector.setCurrentIndex(index)
        self.combo_plot_selector.blockSignals(False)
        self._update_image_selector_enabled_state()
        self.plot()

    def _update_image_selector_enabled_state(self):
        mode = self.combo_plot_selector.currentData()
        is_image_mode = (mode == "IMAGE")
        self.lbl_field_image.setEnabled(is_image_mode)
        self.combo_field_image.setEnabled(is_image_mode)
        self.chk_overlay_field_tracks.setEnabled(is_image_mode)

    def get_selected_plot_mode(self):
        """Returns the active plot mode code (e.g. 'IMAGE', 'LIVE_DIFF', 'NONE', etc.)."""
        return self.combo_plot_selector.currentData() or "IMAGE"

    # ------------------------------------------------------------------
    # DATA & PARAMETER ACCESSORS (EMBEDDED VS STANDALONE)
    # ------------------------------------------------------------------

    def _ctx(self):
        """Returns the active state provider (ResultsViewerTab when embedded, self when standalone)."""
        return self.viewer if self._is_embedded else self

    def populate_field_images(self):
        """Populates the static disk image/output dropdown for the currently selected field."""
        ctx = self._ctx()
        selected_field = ctx.field_selector.currentText() if hasattr(ctx, "field_selector") else ""

        self.combo_field_image.blockSignals(True)
        self.combo_field_image.clear()

        if (
            selected_field
            and selected_field not in ("-- All Fields --", "-- Selected Fields --")
            and selected_field in ctx.field_dirs_map
        ):
            search_dirs = self._get_search_dirs_for_field(selected_field)
            img_files = []
            for s_dir in search_dirs:
                if os.path.exists(s_dir):
                    for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.PNG', '*.JPG', '*.JPEG', '*.TIF', '*.TIFF'):
                        img_files.extend(glob.glob(os.path.join(s_dir, ext)))

            unique_images = sorted(list(set(img_files)))
            if unique_images:
                def sort_key(filepath):
                    fname = os.path.basename(filepath).lower()
                    if 'avg' in fname:
                        return (0, fname)
                    return (1, fname)

                for fpath in sorted(unique_images, key=sort_key):
                    rel_name = os.path.basename(fpath)
                    parent_dir = os.path.basename(os.path.dirname(fpath))
                    if parent_dir == "results":
                        rel_name = f"results/{rel_name}"
                    self.combo_field_image.addItem(rel_name, userData=fpath)
            else:
                self.combo_field_image.addItem("(No static images found on disk)", userData=None)
        else:
            self.combo_field_image.addItem("(Select a single field first)", userData=None)

        self.combo_field_image.blockSignals(False)
        self._update_image_selector_enabled_state()

    def _get_search_dirs_for_field(self, field_name):
        ctx = self._ctx()
        if hasattr(ctx, "get_search_dirs_for_field") and ctx is not self:
            return ctx.get_search_dirs_for_field(field_name)

        if not field_name or field_name not in self.field_dirs_map:
            return []
        info = self.field_dirs_map[field_name]
        if isinstance(info, dict):
            sample_dir = info.get("sample_dir", self.root_dir)
            traj_dir = info.get("traj_dir", "")
        else:
            traj_dir = info
            sample_dir = os.path.dirname(traj_dir)

        search_dirs = []
        if sample_dir and field_name:
            search_dirs.append(os.path.join(sample_dir, field_name, "results"))
            search_dirs.append(os.path.join(sample_dir, f"{field_name}_results"))
            search_dirs.append(os.path.join(sample_dir, field_name))
        if traj_dir and traj_dir not in search_dirs:
            search_dirs.append(traj_dir)
            search_dirs.append(os.path.join(traj_dir, "results"))
        return [d for d in list(dict.fromkeys(search_dirs)) if os.path.exists(d)]

    def _get_filtered_summary_df(self):
        ctx = self._ctx()
        if hasattr(ctx, "get_filtered_summary_df") and ctx is not self:
            return ctx.get_filtered_summary_df()

        if self.diffusion_summary_df.empty:
            return pd.DataFrame()
        df = self.diffusion_summary_df.copy()
        chan = self.channel_selector.currentText()
        if chan == "Left Channel (L)":
            df = df[df['channel'] == "Left Channel (L)"]
        elif chan == "Right Channel (R)":
            df = df[df['channel'] == "Right Channel (R)"]
        return df

    def _extract_jump_distances(self, summary_df):
        ctx = self._ctx()
        if hasattr(ctx, "extract_jump_distances") and ctx is not self:
            return ctx.extract_jump_distances(summary_df)

        if self.trajectories_df.empty or summary_df.empty:
            return np.array([])
        allowed_ids = set(summary_df['global_track_id'].unique())
        df = self.trajectories_df[self.trajectories_df['global_track_id'].isin(allowed_ids)]
        pixel_size = self.spin_pixel_size.value()
        jumps = []
        for _, track in df.groupby('global_track_id'):
            if len(track) <= 1:
                continue
            track_sorted = track.sort_values('frame')
            frames = track_sorted['frame'].values
            x = track_sorted['x'].values * pixel_size
            y = track_sorted['y'].values * pixel_size
            valid = (frames[1:] - frames[:-1]) == 1
            if np.any(valid):
                dx = (x[1:] - x[:-1])[valid]
                dy = (y[1:] - y[:-1])[valid]
                jumps.extend(np.sqrt(dx**2 + dy**2))
        return np.array(jumps)

    # ------------------------------------------------------------------
    # MAIN PLOTTING ENTRY POINT
    # ------------------------------------------------------------------

    def plot(self, *_args):
        """Produces only the plot selected in `combo_plot_selector` to minimize lag."""
        fig = self.field_canvas.fig
        plot_mode = self.get_selected_plot_mode()

        # Fast exit if plotting is disabled via dropdown selector
        if plot_mode == "NONE":
            fig.clf()
            ax = fig.add_subplot(111)
            self.field_canvas.axes = ax
            ax.text(
                0.5, 0.5,
                "Single Field Inspector plotting is paused ('None' selected).\n"
                "Choose a plot or image from the Plot Selector dropdown above to render.",
                ha='center', va='center', transform=ax.transAxes, fontsize=10, color='#555'
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            self.field_canvas.draw()
            return

        ctx = self._ctx()
        selected_field = ctx.field_selector.currentText() if hasattr(ctx, "field_selector") else ""

        if not selected_field or selected_field in ("-- All Fields --", "-- Selected Fields --"):
            fig.clf()
            ax = fig.add_subplot(111)
            self.field_canvas.axes = ax
            ax.text(
                0.5, 0.5,
                "Select a single specific field from the 'Field Selection' dropdown to view images & outputs.",
                ha='center', va='center', transform=ax.transAxes, fontsize=11
            )
            ax.grid(False)
            self.field_canvas.draw()
            return

        selected_chan = (
            ctx.channel_selector.currentText()
            if hasattr(ctx, "channel_selector") and ctx.channel_selector.isVisible()
            else "Single Channel"
        )

        # Helper to lazily slice field summary & trajectories only when needed by the selected plot
        def get_field_data():
            summary_df = self._get_filtered_summary_df()
            field_summary = (
                summary_df[summary_df['field'] == selected_field]
                if not summary_df.empty else pd.DataFrame()
            )
            if not ctx.trajectories_df.empty:
                allowed_ids = (
                    set(field_summary['global_track_id'].unique())
                    if not field_summary.empty else set()
                )
                df_field = ctx.trajectories_df[
                    (ctx.trajectories_df['field'] == selected_field)
                    & (ctx.trajectories_df['global_track_id'].isin(allowed_ids))
                ]
            else:
                df_field = pd.DataFrame()
            return field_summary, df_field

        if plot_mode == "LIVE_ALL":
            fig.clf()
            axes = [fig.add_subplot(2, 3, i + 1) for i in range(6)]
            self.field_canvas.axes = axes[0]
            field_summary, df_field = get_field_data()
            self._render_live_diff_coeff(axes[0], field_summary, selected_field, selected_chan)
            self._render_live_jump_distance(axes[1], field_summary, df_field, selected_field, selected_chan)
            self._render_live_stoich_scatter(axes[2], field_summary, selected_field, selected_chan)
            self._render_live_trajectory_traces(axes[3], df_field, selected_field, selected_chan)
            self._render_live_stoich_hist(axes[4], field_summary, selected_field, selected_chan)
            self._render_live_spot_intensity(axes[5], df_field, selected_field, selected_chan)
            fig.tight_layout()
            self.field_canvas.draw()
            return

        # Single-axis views
        fig.clf()
        ax = fig.add_subplot(111)
        self.field_canvas.axes = ax

        if plot_mode == "IMAGE":
            img_path = self.combo_field_image.currentData()
            if not img_path or not os.path.exists(img_path):
                ax.text(
                    0.5, 0.5,
                    f"No image selected or file not found for field '{selected_field}'.",
                    ha='center', va='center', transform=ax.transAxes, fontsize=11
                )
                ax.grid(False)
                self.field_canvas.draw()
                return

            try:
                img_data = np.array(Image.open(img_path))
            except Exception as e:
                ax.text(
                    0.5, 0.5, f"Error loading image:\n{e}",
                    ha='center', va='center', transform=ax.transAxes, fontsize=11
                )
                ax.grid(False)
                self.field_canvas.draw()
                return

            filename = os.path.basename(img_path)
            is_plot_png = filename.lower().endswith(('.png', '.jpg', '.jpeg'))

            if img_data.ndim == 3 and img_data.shape[0] < 10:
                img_data = img_data[0]

            if is_plot_png:
                ax.imshow(img_data, origin='upper', aspect='equal')
                ax.axis('off')
            else:
                ax.imshow(img_data, cmap='gray', origin='lower', aspect='equal')
                ax.set_xlabel("X (px)")
                ax.set_ylabel("Y (px)")
                ax.grid(False)

                if self.chk_overlay_field_tracks.isChecked():
                    _, df_field = get_field_data()
                    if not df_field.empty:
                        for _, track in df_field.groupby('global_track_id'):
                            channel = track['channel'].iloc[0]
                            color = '#00FFFF' if channel == "Left Channel (L)" else '#FF00FF'
                            ax.plot(track['x'], track['y'], color=color, alpha=0.8, linewidth=1.0)

            ax.set_title(f"Single Field Inspector: {selected_field} | {filename} [{selected_chan}]", fontsize=10)

        elif plot_mode == "LIVE_DIFF":
            field_summary, _ = get_field_data()
            self._render_live_diff_coeff(ax, field_summary, selected_field, selected_chan)

        elif plot_mode == "LIVE_JUMP":
            field_summary, df_field = get_field_data()
            self._render_live_jump_distance(ax, field_summary, df_field, selected_field, selected_chan)

        elif plot_mode == "LIVE_SCATTER":
            field_summary, _ = get_field_data()
            self._render_live_stoich_scatter(ax, field_summary, selected_field, selected_chan)

        elif plot_mode == "LIVE_TRACES":
            _, df_field = get_field_data()
            self._render_live_trajectory_traces(ax, df_field, selected_field, selected_chan)

        elif plot_mode == "LIVE_STOICH_HIST":
            field_summary, _ = get_field_data()
            self._render_live_stoich_hist(ax, field_summary, selected_field, selected_chan)

        elif plot_mode == "LIVE_SPOTS":
            _, df_field = get_field_data()
            self._render_live_spot_intensity(ax, df_field, selected_field, selected_chan)

        fig.tight_layout()
        self.field_canvas.draw()

    # ------------------------------------------------------------------
    # LIVE PLOT RENDERING HELPERS
    # ------------------------------------------------------------------

    def _render_live_diff_coeff(self, ax, field_summary, field_name, chan_label):
        ctx = self._ctx()
        if field_summary.empty or field_summary['D'].dropna().empty:
            ax.text(0.5, 0.5, f"No valid tracks match filter criteria in {field_name}", ha='center', va='center', transform=ax.transAxes)
            ax.grid(False)
            return

        d_vals = field_summary['D'].dropna().values
        remove_neg = ctx.chk_remove_neg.isChecked() if hasattr(ctx, "chk_remove_neg") else False
        d_clean = d_vals[d_vals > 0] if remove_neg else d_vals

        bins_count = ctx.spin_bins.value() if hasattr(ctx, "spin_bins") else 30
        use_kde = ctx.chk_kde.isChecked() if hasattr(ctx, "chk_kde") else False

        ax.hist(d_clean, bins=bins_count, color='#008080', alpha=0.55, edgecolor='black', linewidth=0.6, label=f"Filtered Tracks (n={len(d_clean)})")

        if len(d_clean) > 1 and use_kde:
            kde = gaussian_kde(d_clean)
            x_grid = np.linspace(d_clean.min(), d_clean.max(), 300)
            b_w = (d_clean.max() - d_clean.min()) / bins_count if d_clean.max() > d_clean.min() else 1.0
            ax.plot(x_grid, kde(x_grid) * len(d_clean) * b_w, color='#004D4D', linewidth=2, label="KDE Fit")

        mean_d = np.mean(d_clean)
        ax.axvline(mean_d, color='red', linestyle='--', linewidth=1.5, label=f"Mean D: {mean_d:.3f} μm²/s")

        ax.set_title(f"Diffusion Coefficient — {field_name} [{chan_label}]", fontsize=10)
        ax.set_xlabel(r"Diffusion Coefficient $D$ $[\mu\mathrm{m}^2/\mathrm{s}]$", fontsize=10)
        ax.set_ylabel("Track Count", fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(False)

    def _render_live_jump_distance(self, ax, field_summary, df_field, field_name, chan_label):
        ctx = self._ctx()
        if df_field.empty or field_summary.empty:
            ax.text(0.5, 0.5, f"No valid tracks match filter criteria in {field_name}", ha='center', va='center', transform=ax.transAxes)
            ax.grid(False)
            return

        jumps = self._extract_jump_distances(field_summary)
        if len(jumps) == 0:
            ax.text(0.5, 0.5, f"No jump distance data for filtered tracks in {field_name}", ha='center', va='center', transform=ax.transAxes)
            ax.grid(False)
            return

        lag_N = ctx.spin_jump_lag.value() if hasattr(ctx, "spin_jump_lag") else 1
        bins_count = ctx.spin_bins_jump.value() if hasattr(ctx, "spin_bins_jump") else 30
        use_kde = ctx.chk_kde_jump.isChecked() if hasattr(ctx, "chk_kde_jump") else False

        ax.hist(jumps, bins=bins_count, color='#008080', alpha=0.55, edgecolor='black', linewidth=0.6, label=f"Jump Steps (n={len(jumps)})")

        if len(jumps) > 1 and use_kde:
            kde = gaussian_kde(jumps)
            x_grid = np.linspace(jumps.min(), jumps.max(), 300)
            b_w = (jumps.max() - jumps.min()) / bins_count if jumps.max() > jumps.min() else 1.0
            ax.plot(x_grid, kde(x_grid) * len(jumps) * b_w, color='#004D4D', linewidth=2, label="KDE Fit")

        mean_j = np.mean(jumps)
        ax.axvline(mean_j, color='red', linestyle='--', linewidth=1.5, label=f"Mean Jump: {mean_j:.3f} μm")

        ax.set_title(f"Jump Distance (N={lag_N} frames) — {field_name} [{chan_label}]", fontsize=10)
        ax.set_xlabel(r"Jump Distance $r$ $[\mu\mathrm{m}]$", fontsize=10)
        ax.set_ylabel("Step Count", fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(False)

    def _render_live_stoich_scatter(self, ax, field_summary, field_name, chan_label):
        ctx = self._ctx()
        if field_summary.empty or field_summary['mean_intensity'].dropna().empty:
            ax.text(0.5, 0.5, f"No valid tracks match filter criteria in {field_name}", ha='center', va='center', transform=ax.transAxes)
            ax.grid(False)
            return

        isingle = ctx.spin_isingle_L.value() if "Right" not in chan_label else ctx.spin_isingle_R.value()
        stoich = field_summary['mean_intensity'] / isingle
        d_vals = field_summary['D']

        ax.scatter(d_vals, stoich, color='#008080', alpha=0.65, s=25, edgecolors='black', linewidth=0.3, label=f"Tracks (n={len(field_summary)})")

        max_stoich = int(np.ceil(stoich.max())) if not stoich.empty and not np.isnan(stoich.max()) else 5
        for i in range(1, min(max_stoich + 1, 10)):
            ax.axhline(i, color='gray', linestyle=':', alpha=0.6, linewidth=0.8)

        ax.set_title(f"Stoichiometry vs Diffusion — {field_name} [{chan_label}]", fontsize=10)
        ax.set_xlabel(r"Diffusion Coefficient $D$ $[\mu\mathrm{m}^2/\mathrm{s}]$", fontsize=10)
        ax.set_ylabel(r"Stoichiometry ($I_{\mathrm{mean}} / I_{\mathrm{single}}$)", fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(False)

    def _render_live_trajectory_traces(self, ax, df_field, field_name, chan_label):
        ctx = self._ctx()
        if df_field.empty:
            ax.text(0.5, 0.5, f"No spot data for filtered trajectories in {field_name}", ha='center', va='center', transform=ax.transAxes)
            ax.grid(False)
            return

        intensity_col = 'integrated_intensity' if 'integrated_intensity' in df_field.columns else (
            'intensity' if 'intensity' in df_field.columns else 'spot_intensity'
        )
        isingle = ctx.spin_isingle_L.value() if "Right" not in chan_label else ctx.spin_isingle_R.value()

        for _, track in df_field.groupby('global_track_id'):
            track_sorted = track.sort_values('frame')
            ax.plot(track_sorted['frame'], track_sorted[intensity_col], alpha=0.6, linewidth=1.1)

        ax.axhline(isingle, color='black', linestyle='--', linewidth=1.5, label=f"iSingle ({isingle:.1f} a.u.)")

        ax.set_title(f"Trajectory Intensity Traces — {field_name} [{chan_label}]", fontsize=10)
        ax.set_xlabel("Frame Number", fontsize=10)
        ax.set_ylabel("Integrated Intensity (a.u.)", fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(False)

    def _render_live_stoich_hist(self, ax, field_summary, field_name, chan_label):
        ctx = self._ctx()
        if field_summary.empty or field_summary['mean_intensity'].dropna().empty:
            ax.text(0.5, 0.5, f"No valid tracks match filter criteria in {field_name}", ha='center', va='center', transform=ax.transAxes)
            ax.grid(False)
            return

        isingle = ctx.spin_isingle_L.value() if "Right" not in chan_label else ctx.spin_isingle_R.value()
        stoich = (field_summary['mean_intensity'] / isingle).dropna()

        bins = np.arange(0, stoich.max() + 0.5, 0.25) if not stoich.empty else 20
        ax.hist(stoich, bins=bins, color='#008080', alpha=0.55, edgecolor='black', linewidth=0.6, label=f"Filtered Tracks (n={len(stoich)})")

        for i in range(1, int(np.ceil(stoich.max())) + 1 if not stoich.empty else 5):
            ax.axvline(i, color='darkred', linestyle=':', alpha=0.7, linewidth=1.0)

        ax.set_title(f"Stoichiometry Histogram — {field_name} [{chan_label}]", fontsize=10)
        ax.set_xlabel(r"Stoichiometry ($N = I_{\mathrm{mean}} / I_{\mathrm{single}}$)", fontsize=10)
        ax.set_ylabel("Trajectory Count", fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(False)

    def _render_live_spot_intensity(self, ax, df_field, field_name, chan_label):
        ctx = self._ctx()
        if df_field.empty:
            ax.text(0.5, 0.5, f"No spot data for filtered trajectories in {field_name}", ha='center', va='center', transform=ax.transAxes)
            ax.grid(False)
            return

        intensity_col = 'integrated_intensity' if 'integrated_intensity' in df_field.columns else (
            'intensity' if 'intensity' in df_field.columns else 'spot_intensity'
        )

        ax.scatter(df_field['frame'], df_field[intensity_col], color='#008080', alpha=0.35, s=10, edgecolors='none', label=f"Spots (n={len(df_field)})")

        isingle = ctx.spin_isingle_L.value() if "Right" not in chan_label else ctx.spin_isingle_R.value()
        ax.axhline(isingle, color='red', linestyle='--', linewidth=1.5, label=f"iSingle ({isingle:.1f} a.u.)")

        ax.set_title(f"Spot Intensity vs Frame — {field_name} [{chan_label}]", fontsize=10)
        ax.set_xlabel("Frame Number", fontsize=10)
        ax.set_ylabel("Spot Integrated Intensity (a.u.)", fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(False)

    # ------------------------------------------------------------------
    # STANDALONE DIRECTORY SCANNING & STATE MANAGEMENT
    # ------------------------------------------------------------------

    def select_new_directory(self):
        selected_dir = QFileDialog.getExistingDirectory(self, "Select Experiment Directory", self.root_dir or ".")
        if selected_dir:
            self.root_dir = os.path.abspath(selected_dir)
            if hasattr(self, "lbl_root_dir"):
                self.lbl_root_dir.setText(self.root_dir)
            self.scan_and_load_directory()

    def scan_and_load_directory(self):
        """Loads trajectory TSVs and field directories when running standalone."""
        if self._is_embedded or not self.root_dir or not os.path.exists(self.root_dir):
            return

        all_files = glob.glob(os.path.join(self.root_dir, "**", "*_trajectories.tsv"), recursive=True)
        traj_list = []
        self.field_dirs_map = {}

        for traj_path in all_files:
            norm_path = os.path.normpath(traj_path)
            path_parts = norm_path.split(os.sep)
            if any(part.startswith("cell_") or part.startswith("cell") for part in path_parts[:-1]):
                continue

            filename = os.path.basename(traj_path)
            parent_dir = os.path.dirname(norm_path)
            parent_folder = os.path.basename(parent_dir)

            if parent_folder == "results":
                field_dir = os.path.dirname(parent_dir)
                field_name = os.path.basename(field_dir)
                sample_dir = os.path.dirname(field_dir)
                sample_label = os.path.basename(sample_dir)
            elif parent_folder.endswith("_results"):
                field_dir = parent_dir
                field_name = parent_folder[:-8].rstrip('_')
                sample_dir = os.path.dirname(parent_dir)
                sample_label = os.path.basename(sample_dir)
            else:
                field_name = parent_folder[:-8].rstrip('_') if parent_folder.endswith("_results") else parent_folder
                field_dir = parent_dir
                sample_dir = os.path.dirname(parent_dir)
                sample_label = os.path.basename(sample_dir)

            self.field_dirs_map[field_name] = {
                "traj_dir": field_dir,
                "sample_dir": sample_dir
            }

            if "R_channel" in filename or "_R_" in filename or "_R." in filename:
                channel, chan_code = "Right Channel (R)", "R"
            else:
                channel, chan_code = "Left Channel (L)", "L"

            try:
                df = pd.read_csv(traj_path, sep='\t')
                df['sample'] = sample_label
                df['field'] = field_name
                df['channel'] = channel
                raw_id_col = 'trajectory' if 'trajectory' in df.columns else (
                    'track_id' if 'track_id' in df.columns else None
                )
                if raw_id_col and raw_id_col in df.columns:
                    df['global_track_id'] = field_name + "_" + chan_code + "_" + df[raw_id_col].astype(str)
                else:
                    df['global_track_id'] = field_name + "_" + chan_code + "_tr" + df.index.astype(str)
                traj_list.append(df)
            except Exception as e:
                print(f"Error reading {traj_path}: {e}")

        self.trajectories_df = pd.concat(traj_list, ignore_index=True) if traj_list else pd.DataFrame()

        self.sample_selector.blockSignals(True)
        self.sample_selector.clear()
        if not self.trajectories_df.empty:
            samples = sorted(self.trajectories_df['sample'].unique().tolist())
            if len(samples) > 1:
                self.sample_selector.addItem("All Samples Combined")
            for s in samples:
                self.sample_selector.addItem(s)
        self.sample_selector.blockSignals(False)

        self._on_standalone_sample_changed()

    def _on_standalone_sample_changed(self):
        self.field_selector.blockSignals(True)
        self.field_selector.clear()
        if not self.trajectories_df.empty:
            sel = self.sample_selector.currentText()
            sub = (
                self.trajectories_df
                if sel in ("", "All Samples Combined")
                else self.trajectories_df[self.trajectories_df['sample'] == sel]
            )
            for f in sorted(sub['field'].unique().tolist()):
                self.field_selector.addItem(f)
        self.field_selector.blockSignals(False)
        self._recalculate_standalone_summary()
        self.populate_field_images()
        self.plot()

    def _on_standalone_field_changed(self):
        self.populate_field_images()
        self.plot()

    def _recalculate_standalone_summary(self):
        if self.trajectories_df.empty:
            self.diffusion_summary_df = pd.DataFrame()
            self.plot()
            return

        sel = self.sample_selector.currentText()
        active_df = (
            self.trajectories_df
            if sel in ("", "All Samples Combined")
            else self.trajectories_df[self.trajectories_df['sample'] == sel]
        )

        pixel_size = self.spin_pixel_size.value()
        dt = (self.spin_exposure.value() / 1000.0) * (2.0 if self.chk_alex.isChecked() else 1.0)
        min_len = self.spin_min_len.value()

        results = []
        for track_id, track in active_df.groupby('global_track_id'):
            if len(track) < min_len:
                continue
            track_sorted = track.sort_values('frame')
            dx = np.diff(track_sorted['x'].values) * pixel_size
            dy = np.diff(track_sorted['y'].values) * pixel_size
            D = np.mean(dx**2 + dy**2) / (4.0 * dt) if dt > 0 else np.nan
            intensity_col = 'integrated_intensity' if 'integrated_intensity' in track.columns else (
                'intensity' if 'intensity' in track.columns else (
                    'spot_intensity' if 'spot_intensity' in track.columns else None
                )
            )
            mean_int = track[intensity_col].mean() if intensity_col else np.nan
            results.append({
                'global_track_id': track_id,
                'sample': track['sample'].iloc[0],
                'field': track['field'].iloc[0],
                'channel': track['channel'].iloc[0],
                'D': D,
                'mean_intensity': mean_int,
            })

        self.diffusion_summary_df = pd.DataFrame(results)
        self.plot()


def launch_single_field_inspector(root_dir=".", parent=None, show=True):
    """Callable entry point to launch the Single Field Inspector window."""
    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)

    inspector = SingleFieldInspectorTab(parent=parent, root_dir=root_dir)
    inspector.setWindowTitle("PySTACHIO Studio - Single Field Inspector")
    inspector.resize(1280, 820)
    if show:
        inspector.show()

    if owns_app:
        sys.exit(app.exec())
    return inspector


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    launch_single_field_inspector(root_dir=target_dir)
