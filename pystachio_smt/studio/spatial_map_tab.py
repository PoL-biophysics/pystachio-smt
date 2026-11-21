import os
import glob
import numpy as np

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from PIL import Image
from scipy.ndimage import affine_transform

try:
    from pystackreg import StackReg
except ImportError:
    StackReg = None

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QGroupBox, QFormLayout, QCheckBox, QSpinBox
)
from PyQt6.QtCore import Qt


class _NoScrollSpinBox(QSpinBox):
    """QSpinBox that ignores mouse wheel scrolling unless explicitly focused by clicking."""
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _NoScrollComboBox(QComboBox):
    """QComboBox that ignores mouse wheel scrolling unless explicitly focused by clicking."""
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _SpatialMplCanvas(FigureCanvasQTAgg):
    """Matplotlib Canvas widget for the Spatial Map tab."""
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class SpatialMapTab(QWidget):
    """Encapsulates the Spatial Map tab UI, sidebar options group, and plotting logic."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self._init_tab_ui()
        self._init_options_group()

    def _init_tab_ui(self):
        layout_spatial = QVBoxLayout(self)

        spatial_top_bar = QHBoxLayout()

        lbl_spatial_field = QLabel("Field:")
        lbl_spatial_field.setStyleSheet("font-weight: bold;")
        self.combo_spatial_field = _NoScrollComboBox()
        self.combo_spatial_field.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.combo_spatial_field.currentIndexChanged.connect(self.refresh)

        lbl_spatial_bg = QLabel("Background Overlay:")
        lbl_spatial_bg.setStyleSheet("font-weight: bold;")

        self.combo_spatial_bg = _NoScrollComboBox()
        self.combo_spatial_bg.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.combo_spatial_bg.addItems([
            "L_channel Average (L_avg.tif)",
            "R_channel Average (R_avg.tif)",
            "Merged Left and Right (Merge)",
            "Brightfield Image (BF.tif)",
            "None (Blank Grid)"
        ])
        self.combo_spatial_bg.currentIndexChanged.connect(self.refresh)

        spatial_top_bar.addWidget(lbl_spatial_field)
        spatial_top_bar.addWidget(self.combo_spatial_field, 1)
        spatial_top_bar.addWidget(lbl_spatial_bg)
        spatial_top_bar.addWidget(self.combo_spatial_bg, 1)

        self.spatial_canvas = _SpatialMplCanvas(self)
        self.spatial_toolbar = NavigationToolbar(self.spatial_canvas, self)

        layout_spatial.addLayout(spatial_top_bar)
        layout_spatial.addWidget(self.spatial_toolbar)
        layout_spatial.addWidget(self.spatial_canvas)

    def _init_options_group(self):
        self.options_group = QGroupBox("Spatial Map Options")
        spatial_opt_layout = QFormLayout()
        spatial_opt_layout.setVerticalSpacing(6)

        self.chk_spatial_show_L = QCheckBox("Show Left Channel (L) Tracks")
        self.chk_spatial_show_L.setChecked(True)
        self.chk_spatial_show_L.stateChanged.connect(self.refresh)

        self.chk_spatial_show_R = QCheckBox("Show Right Channel (R) Tracks")
        self.chk_spatial_show_R.setChecked(True)
        self.chk_spatial_show_R.stateChanged.connect(self.refresh)

        self.chk_spatial_transform_R = QCheckBox("Transform R Channel (transformation_matrix.npy)")
        self.chk_spatial_transform_R.setChecked(True)
        self.chk_spatial_transform_R.stateChanged.connect(self.refresh)

        self.chk_spatial_show_radial_circle = QCheckBox("Show Radial Filter Circle (radial_filter_circle.npy)")
        self.chk_spatial_show_radial_circle.setChecked(False)
        self.chk_spatial_show_radial_circle.stateChanged.connect(self.refresh)

        self.chk_spatial_show_mask = QCheckBox("Show Mask Overlay (Dotted Green)")
        self.chk_spatial_show_mask.setChecked(True)
        self.chk_spatial_show_mask.stateChanged.connect(self.refresh)

        self.lbl_spatial_entity = QLabel("Spatial Entity Mapping:")
        self.combo_spatial_entity = _NoScrollComboBox()
        self.combo_spatial_entity.addItems([
            "Map Full Trajectories",
            "Map Single Jump Steps",
            "Map All Spots"
        ])
        self.combo_spatial_entity.currentIndexChanged.connect(self.on_spatial_entity_changed)

        self.lbl_spatial_color_mode = QLabel("Track Color Mode:")
        self.combo_spatial_color_mode = _NoScrollComboBox()
        self.combo_spatial_color_mode.addItems([
            "Channel Color (Default)",
            "Color by Diffusion Coefficient (D)",
            "Color by Jump Distance (N frames)",
            "Color by Spot Intensity"
        ])
        self.combo_spatial_color_mode.currentIndexChanged.connect(self.on_spatial_color_mode_changed)

        self.lbl_spatial_jump_lag = QLabel("Spatial Jump Lag (N):")
        self.spin_spatial_jump_lag = _NoScrollSpinBox()
        self.spin_spatial_jump_lag.setRange(1, 100)
        self.spin_spatial_jump_lag.setValue(1)
        self.spin_spatial_jump_lag.setSuffix(" frames")
        self.spin_spatial_jump_lag.valueChanged.connect(self.refresh)

        for widget in (self.combo_spatial_entity, self.combo_spatial_color_mode, self.spin_spatial_jump_lag):
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        spatial_opt_layout.addRow(self.chk_spatial_show_L)
        spatial_opt_layout.addRow(self.chk_spatial_show_R)
        spatial_opt_layout.addRow(self.chk_spatial_transform_R)
        spatial_opt_layout.addRow(self.chk_spatial_show_radial_circle)
        spatial_opt_layout.addRow(self.chk_spatial_show_mask)
        spatial_opt_layout.addRow(self.lbl_spatial_entity, self.combo_spatial_entity)
        spatial_opt_layout.addRow(self.lbl_spatial_color_mode, self.combo_spatial_color_mode)
        spatial_opt_layout.addRow(self.lbl_spatial_jump_lag, self.spin_spatial_jump_lag)
        self.options_group.setLayout(spatial_opt_layout)
        self.options_group.setVisible(False)

    # ================= STATE & EVENT SYNCHRONIZATION =================
    
    def populate_fields(self):
        """Populates Spatial Map's independent field dropdown for the active sample."""
        prev_field = self.combo_spatial_field.currentText()
        self.combo_spatial_field.blockSignals(True)
        self.combo_spatial_field.clear()

        if not self.viewer.trajectories_df.empty:
            selected_sample = self.viewer.sample_selector.currentText()
            if selected_sample == "All Samples Combined" or not selected_sample:
                fields = sorted(self.viewer.trajectories_df['field'].unique().tolist())
            else:
                sample_df = self.viewer.trajectories_df[self.viewer.trajectories_df['sample'] == selected_sample]
                fields = sorted(sample_df['field'].unique().tolist())

            items = ["-- All Fields --"] + fields
            self.combo_spatial_field.addItems(items)
            if prev_field in items:
                self.combo_spatial_field.setCurrentText(prev_field)

        self.combo_spatial_field.blockSignals(False)

    def refresh(self):
        """Re-renders the spatial map using the viewer's non-field-filtered summary DataFrame."""
        self.plot(self.viewer.get_filtered_summary_df(apply_field_filter=False))

    def sync_from_analysis_mode(self, analysis_mode, jump_lag=None, redraw=False):
        """Synchronizes entity/color dropdowns and jump lag with the active analysis tab."""
        target_mode = (
            "Color by Jump Distance (N frames)"
            if analysis_mode == 'jump'
            else "Color by Diffusion Coefficient (D)"
        )
        self.combo_spatial_color_mode.blockSignals(True)
        self.combo_spatial_color_mode.setCurrentText(target_mode)
        self.combo_spatial_color_mode.blockSignals(False)

        if analysis_mode == 'jump' and jump_lag is not None:
            self.spin_spatial_jump_lag.blockSignals(True)
            self.spin_spatial_jump_lag.setValue(jump_lag)
            self.spin_spatial_jump_lag.blockSignals(False)

        self.combo_spatial_entity.blockSignals(True)
        self.combo_spatial_entity.setCurrentText("Map Full Trajectories")
        self.combo_spatial_entity.blockSignals(False)

        if redraw:
            self.refresh()

    def on_spatial_entity_changed(self):
        """Synchronizes spatial color mode when switching to or from Map All Spots."""
        entity_mode = self.combo_spatial_entity.currentText()
        color_mode = self.combo_spatial_color_mode.currentText()

        self.combo_spatial_color_mode.blockSignals(True)
        if entity_mode == "Map All Spots" and color_mode != "Color by Spot Intensity":
            self.combo_spatial_color_mode.setCurrentText("Color by Spot Intensity")
        elif entity_mode != "Map All Spots" and color_mode == "Color by Spot Intensity":
            self.combo_spatial_color_mode.setCurrentText("Channel Color (Default)")
        self.combo_spatial_color_mode.blockSignals(False)

        self.refresh()

    def on_spatial_color_mode_changed(self):
        """Synchronizes spatial entity mode when switching to or from Color by Spot Intensity."""
        color_mode = self.combo_spatial_color_mode.currentText()
        entity_mode = self.combo_spatial_entity.currentText()

        self.combo_spatial_entity.blockSignals(True)
        if color_mode == "Color by Spot Intensity" and entity_mode != "Map All Spots":
            self.combo_spatial_entity.setCurrentText("Map All Spots")
        elif color_mode in ("Color by Diffusion Coefficient (D)", "Color by Jump Distance (N frames)") and entity_mode == "Map All Spots":
            self.combo_spatial_entity.setCurrentText("Map Full Trajectories")
        self.combo_spatial_entity.blockSignals(False)

        self.refresh()

    # ================= TRANSFORMATION MATRIX HELPERS =================

    def find_transformation_matrix(self, field_name=None):
        """Looks for transformation_matrix.npy in field results folders, then in root_dir."""
        candidates = []
        if field_name and field_name in self.viewer.field_dirs_map:
            for s_dir in self.viewer.get_search_dirs_for_field(field_name):
                candidates.append(os.path.join(s_dir, "transformation_matrix.npy"))

        if self.viewer.root_dir:
            candidates.append(os.path.join(self.viewer.root_dir, "transformation_matrix.npy"))
            glob_matches = glob.glob(os.path.join(self.viewer.root_dir, "**", "transformation_matrix.npy"), recursive=True)
            for gm in glob_matches:
                if gm not in candidates:
                    candidates.append(gm)

        for path in candidates:
            if os.path.exists(path):
                try:
                    tmat = np.load(path)
                    return tmat, path
                except Exception as e:
                    print(f"Error loading transformation matrix from {path}: {e}")
        return None, None

    def transform_track_coordinates(self, x, y, tmat):
        """Transforms coordinate arrays (x, y) using transformation_matrix.npy."""
        if tmat is None:
            return x, y
        try:
            x_arr = np.asarray(x, dtype=np.float64)
            y_arr = np.asarray(y, dtype=np.float64)
            pts = np.vstack([x_arr, y_arr, np.ones_like(x_arr)])

            if tmat.shape == (2, 3):
                tmat_3x3 = np.vstack([tmat, [0, 0, 1]])
            elif tmat.shape == (3, 3):
                tmat_3x3 = tmat
            else:
                return x, y

            # StackReg matrices map Target (L) -> Source (R) for image resampling.
            # Invert the matrix to map track coordinates from Source (R) -> Target (L).
            inv_tmat = np.linalg.inv(tmat_3x3)
            res = inv_tmat @ pts

            if np.any(res[2] != 0):
                return res[0] / res[2], res[1] / res[2]
            return res[0], res[1]
        except Exception as e:
            print(f"Error transforming track coordinates: {e}")
            return x, y

    def _transform_single_object_mask(self, obj_mask, tmat, direction="R_to_L"):
        """Transforms an individual 2D boolean object mask using transformation_matrix.npy."""
        if tmat is None:
            return obj_mask
        try:
            if tmat.shape == (2, 3):
                tmat_3x3 = np.vstack([tmat, [0.0, 0.0, 1.0]])
            elif tmat.shape == (3, 3):
                tmat_3x3 = np.asarray(tmat, dtype=np.float64)
            else:
                return obj_mask

            # tmat_3x3 maps Target (L) -> Source (R) for image resampling.
            # R_to_L resampling uses tmat_3x3; L_to_R resampling uses inv(tmat_3x3).
            mat = tmat_3x3 if direction == "R_to_L" else np.linalg.inv(tmat_3x3)

            # Swap (x, y) -> (row=y, col=x) for scipy.ndimage.affine_transform
            M_yx = np.array([
                [mat[1, 1], mat[1, 0]],
                [mat[0, 1], mat[0, 0]],
            ], dtype=np.float64)
            offset_yx = np.array([mat[1, 2], mat[0, 2]], dtype=np.float64)

            warped = affine_transform(
                obj_mask.astype(np.uint8),
                M_yx,
                offset=offset_yx,
                output_shape=obj_mask.shape,
                order=0,
                mode='constant',
                cval=0,
            )
            return warped > 0
        except Exception as e:
            print(f"Error transforming individual object mask: {e}")
            return obj_mask

    # ================= SPATIAL MAP PLOTTING =================

    def plot(self, summary_df=None):
        fig = self.spatial_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)

        if summary_df is None:
            summary_df = self.viewer.get_filtered_summary_df(apply_field_filter=False)

        selected_field = self.combo_spatial_field.currentText()
        if not summary_df.empty and 'field' in summary_df.columns:
            if selected_field and selected_field != "-- All Fields --":
                summary_df = summary_df[summary_df['field'] == selected_field]

        if self.viewer.trajectories_df.empty or summary_df.empty:
            ax.grid(False)
            self.spatial_canvas.draw()
            return

        bg_choice = self.combo_spatial_bg.currentText()
        img_data = None

        tmat, tmat_path = self.find_transformation_matrix(selected_field)

        if selected_field and selected_field != "-- All Fields --" and selected_field in self.viewer.field_dirs_map:
            search_dirs = self.viewer.get_search_dirs_for_field(selected_field)

            def find_image_file(patterns):
                for s_dir in search_dirs:
                    if os.path.exists(s_dir):
                        for pat in patterns:
                            matches = glob.glob(os.path.join(s_dir, pat))
                            if matches:
                                return matches[0]
                return None

            if bg_choice == "L_channel Average (L_avg.tif)":
                l_file = find_image_file(["*L_avg*.[tT][iI][fF]*", "l_avg.[tT][iI][fF]*"])
                if l_file:
                    try:
                        img_data = np.array(Image.open(l_file))
                    except Exception as e:
                        print(f"Error loading {l_file}: {e}")

            elif bg_choice == "R_channel Average (R_avg.tif)":
                r_file = find_image_file(["*R_avg*.[tT][iI][fF]*", "r_avg.[tT][iI][fF]*"])
                if r_file:
                    try:
                        img_data = np.array(Image.open(r_file)).astype(np.float64)
                        if self.chk_spatial_transform_R.isChecked() and tmat is not None and StackReg is not None:
                            try:
                                sr = StackReg(StackReg.RIGID_BODY)
                                bkg = np.mean(img_data)
                                img_data = sr.transform(img_data, tmat)
                                img_data[img_data == 0] = bkg
                                img_data = img_data - bkg
                                img_data[img_data < 0] = 0
                            except Exception as e:
                                print(f"StackReg transform failed: {e}")
                    except Exception as e:
                        print(f"Error loading {r_file}: {e}")

            elif bg_choice == "Merged Left and Right (Merge)":
                l_file = find_image_file(["*L_avg*.[tT][iI][fF]*", "l_avg.[tT][iI][fF]*"])
                r_file = find_image_file(["*R_avg*.[tT][iI][fF]*", "r_avg.[tT][iI][fF]*"])

                if l_file and r_file:
                    try:
                        img_L = np.array(Image.open(l_file)).astype(np.float64)
                        img_R = np.array(Image.open(r_file)).astype(np.float64)

                        img_L = img_L - np.mean(img_L)
                        img_L[img_L < 0] = 0

                        if self.chk_spatial_transform_R.isChecked() and tmat is not None and StackReg is not None:
                            try:
                                sr = StackReg(StackReg.RIGID_BODY)
                                bkg = np.mean(img_R)
                                img_R = sr.transform(img_R, tmat)
                                img_R[img_R == 0] = bkg
                                img_R = img_R - bkg
                                img_R[img_R < 0] = 0
                            except Exception as e:
                                print(f"StackReg transform failed: {e}")

                        max_R = np.max(img_R) if np.max(img_R) > 0 else 1.0
                        max_L = np.max(img_L) if np.max(img_L) > 0 else 1.0

                        img_R_norm = (np.clip(img_R, 0, None) / max_R * 255).astype(np.uint8)
                        img_L_norm = (np.clip(img_L, 0, None) / max_L * 255).astype(np.uint8)

                        composite_merge = np.zeros((img_L.shape[0], img_L.shape[1], 3), dtype=np.uint8)
                        composite_merge[:, :, 0] = img_R_norm  # Red Channel
                        composite_merge[:, :, 1] = img_L_norm  # Green Channel
                        img_data = composite_merge
                    except Exception as e:
                        print(f"Error merging L and R images: {e}")

            elif bg_choice == "Brightfield Image (BF.tif)":
                bf_file = find_image_file(["*BF*.[tT][iI][fF]*", "bf.[tT][iI][fF]*"])
                if bf_file:
                    try:
                        img_data = np.array(Image.open(bf_file))
                    except Exception as e:
                        print(f"Error loading {bf_file}: {e}")

        if img_data is not None:
            if img_data.ndim == 3 and img_data.shape[0] < 10 and img_data.shape[2] != 3:
                img_data = img_data[0]
            if img_data.ndim == 2:
                ax.imshow(img_data, cmap='gray', origin='lower')
            else:
                ax.imshow(img_data, origin='lower')

        allowed_ids = set(summary_df['global_track_id'].unique())
        df = self.viewer.trajectories_df[self.viewer.trajectories_df['global_track_id'].isin(allowed_ids)]

        show_L = self.chk_spatial_show_L.isChecked()
        show_R = self.chk_spatial_show_R.isChecked()
        transform_R = self.chk_spatial_transform_R.isChecked()

        entity_mode = self.combo_spatial_entity.currentText()
        color_mode = self.combo_spatial_color_mode.currentText()

        try:
            cmap = matplotlib.colormaps['viridis']
        except AttributeError:
            import matplotlib.cm as cm
            cmap = cm.get_cmap('viridis')

        if color_mode == "Color by Diffusion Coefficient (D)":
            d_map = dict(zip(summary_df['global_track_id'], summary_df['D']))
            segments = []
            segment_vals = []

            for track_id, track in df.groupby('global_track_id'):
                channel = track['channel'].iloc[0]
                if channel == "Left Channel (L)" and not show_L:
                    continue
                if channel == "Right Channel (R)" and not show_R:
                    continue

                d_val = d_map.get(track_id, np.nan)
                if np.isnan(d_val) or d_val <= 0:
                    continue

                track_sorted = track.sort_values('frame')
                x_coords = track_sorted['x'].values
                y_coords = track_sorted['y'].values

                if channel == "Right Channel (R)" and transform_R and tmat is not None:
                    x_coords, y_coords = self.transform_track_coordinates(x_coords, y_coords, tmat)

                pts = np.column_stack([x_coords, y_coords])

                if len(pts) > 1:
                    for i in range(len(pts) - 1):
                        segments.append(pts[i:i+2])
                        segment_vals.append(d_val)

            if segments:
                segment_vals = np.array(segment_vals)
                vmin = np.percentile(segment_vals, 2) if len(segment_vals) > 10 else segment_vals.min()
                vmax = np.percentile(segment_vals, 98) if len(segment_vals) > 10 else segment_vals.max()
                if vmin == vmax:
                    vmax = vmin + 1e-5

                norm = Normalize(vmin=vmin, vmax=vmax)
                lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.2, alpha=0.85)
                lc.set_array(segment_vals)
                ax.add_collection(lc)

                cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(r"Diffusion Coefficient $D$ ($\mu\mathrm{m}^2/\mathrm{s}$)", fontsize=10)

        elif color_mode == "Color by Jump Distance (N frames)":
            pixel_size = self.viewer.spin_pixel_size.value()
            lag_N = self.spin_spatial_jump_lag.value()

            segments = []
            segment_vals = []

            for track_id, track in df.groupby('global_track_id'):
                channel = track['channel'].iloc[0]
                if channel == "Left Channel (L)" and not show_L:
                    continue
                if channel == "Right Channel (R)" and not show_R:
                    continue

                track_sorted = track.sort_values('frame')
                frames = track_sorted['frame'].values
                x_coords = track_sorted['x'].values
                y_coords = track_sorted['y'].values

                if channel == "Right Channel (R)" and transform_R and tmat is not None:
                    x_coords, y_coords = self.transform_track_coordinates(x_coords, y_coords, tmat)

                if len(frames) <= lag_N:
                    continue

                pts = np.column_stack([x_coords, y_coords])

                for i in range(len(frames) - lag_N):
                    if frames[i + lag_N] - frames[i] == lag_N:
                        dx = (x_coords[i + lag_N] - x_coords[i]) * pixel_size
                        dy = (y_coords[i + lag_N] - y_coords[i]) * pixel_size
                        dist = np.sqrt(dx**2 + dy**2)

                        if entity_mode == "Map Single Jump Steps":
                            segments.append(np.array([pts[i], pts[i + lag_N]]))
                        else:
                            segments.append(pts[i:i + lag_N + 1])
                        segment_vals.append(dist)

            if segments:
                segment_vals = np.array(segment_vals)
                vmin = np.percentile(segment_vals, 2) if len(segment_vals) > 10 else segment_vals.min()
                vmax = np.percentile(segment_vals, 98) if len(segment_vals) > 10 else segment_vals.max()
                if vmin == vmax:
                    vmax = vmin + 1e-5

                norm = Normalize(vmin=vmin, vmax=vmax)
                lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.2, alpha=0.85)
                lc.set_array(segment_vals)
                ax.add_collection(lc)

                cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(f"Jump Distance $r$ ($\mu\mathrm{{m}}$, N={lag_N} frames)", fontsize=10)

        elif color_mode == "Color by Spot Intensity" or entity_mode == "Map All Spots":
            intensity_col = 'integrated_intensity' if 'integrated_intensity' in df.columns else (
                'intensity' if 'intensity' in df.columns else (
                    'spot_intensity' if 'spot_intensity' in df.columns else None
                )
            )

            spot_x_list = []
            spot_y_list = []
            spot_int_list = []
            spot_colors_list = []

            for channel, chan_df in df.groupby('channel'):
                if channel == "Left Channel (L)" and not show_L:
                    continue
                if channel == "Right Channel (R)" and not show_R:
                    continue

                if intensity_col and intensity_col in chan_df.columns:
                    valid_chan_df = chan_df[chan_df[intensity_col].notna()].copy()
                else:
                    valid_chan_df = chan_df.copy()

                if valid_chan_df.empty:
                    continue

                x_coords = valid_chan_df['x'].values
                y_coords = valid_chan_df['y'].values

                if channel == "Right Channel (R)" and transform_R and tmat is not None:
                    x_coords, y_coords = self.transform_track_coordinates(x_coords, y_coords, tmat)

                spot_x_list.append(x_coords)
                spot_y_list.append(y_coords)

                if intensity_col and intensity_col in valid_chan_df.columns:
                    spot_int_list.append(valid_chan_df[intensity_col].values)

                ch_color = (
                    '#00FFFF' if channel == "Left Channel (L)" else '#FF00FF'
                ) if img_data is not None else (
                    '#008080' if channel == "Left Channel (L)" else '#D81B60'
                )
                spot_colors_list.extend([ch_color] * len(x_coords))

            if spot_x_list:
                all_x = np.concatenate(spot_x_list)
                all_y = np.concatenate(spot_y_list)

                if color_mode == "Color by Spot Intensity" and spot_int_list:
                    all_intensities = np.concatenate(spot_int_list)
                    vmin = np.percentile(all_intensities, 2) if len(all_intensities) > 10 else all_intensities.min()
                    vmax = np.percentile(all_intensities, 98) if len(all_intensities) > 10 else all_intensities.max()
                    if vmin == vmax:
                        vmax = vmin + 1e-5

                    norm = Normalize(vmin=vmin, vmax=vmax)
                    sc = ax.scatter(
                        all_x, all_y, c=all_intensities, cmap=cmap, norm=norm,
                        s=16, alpha=0.85, edgecolors='none', zorder=3
                    )
                    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label("Spot Intensity (a.u.)", fontsize=10)
                else:
                    ax.scatter(
                        all_x, all_y, c=spot_colors_list,
                        s=16, alpha=0.75, edgecolors='none', zorder=3
                    )

        else:  # Channel Color (Default)
            for track_id, track in df.groupby('global_track_id'):
                channel = track['channel'].iloc[0]

                if channel == "Left Channel (L)" and not show_L:
                    continue
                if channel == "Right Channel (R)" and not show_R:
                    continue

                track_sorted = track.sort_values('frame')
                x_coords = track_sorted['x'].values
                y_coords = track_sorted['y'].values

                if channel == "Right Channel (R)" and transform_R and tmat is not None:
                    x_coords, y_coords = self.transform_track_coordinates(x_coords, y_coords, tmat)

                color = (
                    '#00FFFF' if channel == "Left Channel (L)" else '#FF00FF'
                ) if img_data is not None else (
                    '#008080' if channel == "Left Channel (L)" else '#D81B60'
                )

                if entity_mode == "Map Single Jump Steps":
                    for i in range(len(x_coords) - 1):
                        ax.plot(x_coords[i:i+2], y_coords[i:i+2], color=color, alpha=0.6, linewidth=0.8)
                else:
                    ax.plot(x_coords, y_coords, color=color, alpha=0.6, linewidth=0.8)

        legend_handles = []
        field_arg = None if selected_field in ("-- All Fields --", "-- Selected Fields --", "") else selected_field

        # Overlay Saved Radial Filter Circle if checkbox is selected
        if self.chk_spatial_show_radial_circle.isChecked() and hasattr(self.viewer, 'get_radial_filter_circle'):
            circle_info = self.viewer.get_radial_filter_circle(field_arg)
            if circle_info is not None:
                cx = float(circle_info.get('center_x', 0.0))
                cy = float(circle_info.get('center_y', 0.0))
                radius_px = float(circle_info.get('radius_px', 0.0))
                radius_pct = float(circle_info.get('radius_pct', 0.0))
                if radius_px > 0:
                    circle_color = '#FFD700' if img_data is not None else '#FF6F00'
                    circle_patch = Circle(
                        (cx, cy),
                        radius_px,
                        fill=False,
                        edgecolor=circle_color,
                        linestyle='--',
                        linewidth=1.8,
                        zorder=10,
                        label=f"Radial Filter ({radius_pct:.1f}%, r={radius_px:.1f} px)"
                    )
                    ax.add_patch(circle_patch)
                    ax.plot(cx, cy, marker='+', markersize=8, color=circle_color, markeredgewidth=1.5, zorder=10)
                    legend_handles.append(circle_patch)

        # Overlay TIFF Mask as Dotted Green Lines (transforming individual objects for L/R overlay)
        if self.chk_spatial_show_mask.isChecked() and hasattr(self.viewer, 'get_mask_for_field'):
            mask_info = self.viewer.get_mask_for_field(field_arg)
            if mask_info is not None and mask_info.get('object_masks'):
                object_masks = mask_info['object_masks']
                mask_channel = self.viewer.combo_mask_channel.currentText()
                viewing_raw_r_only = (bg_choice == "R_channel Average (R_avg.tif)" and not transform_R)

                if mask_channel == "Right Channel (R)" and transform_R and not viewing_raw_r_only and tmat is not None:
                    # Transform each individual object in the stack/mask from R -> L for the L/R overlay
                    display_objects = [
                        self._transform_single_object_mask(obj_m, tmat, direction="R_to_L")
                        for obj_m in object_masks
                    ]
                    transformed_tag = " [transformed R→L]"
                elif mask_channel == "Left Channel (L)" and viewing_raw_r_only and tmat is not None:
                    display_objects = [
                        self._transform_single_object_mask(obj_m, tmat, direction="L_to_R")
                        for obj_m in object_masks
                    ]
                    transformed_tag = " [transformed L→R]"
                else:
                    display_objects = object_masks
                    transformed_tag = ""

                mask_color = '#00FF00' if img_data is not None else '#009E00'
                plotted_objects = 0
                for obj_m in display_objects:
                    if np.any(obj_m):
                        ax.contour(
                            obj_m.astype(np.float64),
                            levels=[0.5],
                            colors=[mask_color],
                            linestyles=[':'],
                            linewidths=1.6,
                            zorder=9,
                        )
                        plotted_objects += 1

                if plotted_objects > 0:
                    ch_short = "R" if mask_channel == "Right Channel (R)" else "L"
                    legend_handles.append(
                        Line2D(
                            [0], [0],
                            color=mask_color,
                            linestyle=':',
                            linewidth=1.8,
                            label=f"Mask ({ch_short}, {plotted_objects} obj){transformed_tag}"
                        )
                    )

        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', fontsize=8)

        # Construct Channel Title Label
        if self.viewer.channel_selector.isVisible():
            sel_combo = self.viewer.channel_selector.currentText()
            if sel_combo == "Left Channel (L)":
                chan_title_str = "Left Channel (L)"
            elif sel_combo == "Right Channel (R)":
                chan_title_str = "Right Channel (R)"
            else:
                if show_L and show_R:
                    chan_title_str = "Both Channels (L + R Overlay)"
                elif show_L and not show_R:
                    chan_title_str = "Left Channel (L)"
                elif show_R and not show_L:
                    chan_title_str = "Right Channel (R)"
                else:
                    chan_title_str = "No Channels Visible"
        else:
            if show_L and not show_R:
                chan_title_str = "Left Channel (L)"
            elif show_R and not show_L:
                chan_title_str = "Right Channel (R)"
            elif show_L and show_R:
                chan_title_str = "Both Channels (L + R Overlay)"
            else:
                chan_title_str = "Single Channel (L)"

        field_suffix = (
            f" | Field: {selected_field}"
            if selected_field and selected_field != "-- All Fields --"
            else ""
        )
        tmat_info = f" [tmat: {os.path.basename(tmat_path)}]" if (transform_R and tmat_path) else ""

        ax.set_title(
            f"Spatial Map — {self.viewer.sample_selector.currentText()} [{chan_title_str}]{field_suffix}{tmat_info}",
            fontsize=11
        )
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(False)
        self.spatial_canvas.draw()