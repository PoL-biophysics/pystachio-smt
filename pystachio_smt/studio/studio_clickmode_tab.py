import sys
import os
import copy
from pathlib import Path
import numpy as np
import tifffile as tf
import cv2
import matplotlib
matplotlib.use('QtAgg')

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.patches import Rectangle

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, 
                             QLineEdit, QLabel, QRadioButton, QGroupBox, QFileDialog, 
                             QCheckBox, QMessageBox, QScrollArea, QFormLayout)
from PyQt6.QtCore import Qt

try:
    import images
    import spots
    import parameters
    import trajectories
    import postprocessing
    MODULES_LOADED = True
except ImportError:
    MODULES_LOADED = False

def show_popup(parent, title, text, critical=False):
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QMessageBox.Icon.Critical if critical else QMessageBox.Icon.Information)
    msg.exec()


# =========================================================================
# TAB 3: CLICK MODE (DYNAMIC SPLITTING & GLOBALLY LINKED IDs)
# =========================================================================
class ClickModeTab(QWidget):
    def __init__(self, parent_suite=None):
        super().__init__()
        self.parent_suite = parent_suite
        self.params = parameters.Parameters()
        self.active_frame = 0
        self.inset_bounds = None
        self.zoom_selector = None
        self.raw_frame_indices = {}
        self.dropped_raw_frames = 0
        
        # Persistent Storage for On-the-Fly Splitting
        self.raw_stack = None
        self.loaded_fname = None
        
        # --- ID Linking Ledger ---
        self.global_track_id = 0
        self.click_events = [] # Stores dicts: {"id": int, "ch1": [x,y] or None, "ch2": [x,y] or None}
        
        # Native Separated Channel State
        self.img_ch1 = None
        self.img_ch2 = None
        self.coords_ch1 = []
        self.coords_ch2 = []
        self.trajs_ch1 = []
        self.trajs_ch2 = []
        
        # Backward Compatibility Fallbacks for Main Suite Window Reset
        self.image_data = None 
        self.coords = []
        self.trajs = []
        self.all_spots = []

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # --- Matplotlib Canvas Section ---
        canvas_layout = QVBoxLayout()
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(14, 5))
        self.fig.subplots_adjust(bottom=0.15, wspace=0.3)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.toolbar = NavigationToolbar(self.canvas, self)

        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)
        
        slider_layout = QHBoxLayout()
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.update_frame)
        
        slider_layout.addWidget(self.frame_label)
        slider_layout.addWidget(self.frame_slider)
        canvas_layout.addLayout(slider_layout)
        
        main_layout.addLayout(canvas_layout, stretch=4)

        # --- Interactive Controls Panel ---
        controls_widget = QWidget()
        control_layout = QVBoxLayout(controls_widget)
        
        # Configuration Mode Selection
        mode_group = QGroupBox("Channel Mode")
        mode_lay = QVBoxLayout()
        self.rb_single = QRadioButton("Single Channel (Full Frame)")
        self.rb_dual = QRadioButton("Dual Channel (Split Side-by-Side)")
        self.rb_single.setChecked(True)
        self.rb_single.toggled.connect(self.on_mode_change)
        mode_lay.addWidget(self.rb_single)
        mode_lay.addWidget(self.rb_dual)
        mode_group.setLayout(mode_lay)
        control_layout.addWidget(mode_group)

        alex_group = QGroupBox("Alternating excitation (ALEX)")
        alex_layout = QVBoxLayout(alex_group)
        self.chk_alex = QCheckBox("Deinterleave alternating laser frames")
        self.chk_alex.setChecked(True)
        self.chk_alex.setEnabled(False)
        self.rb_first_ch1 = QRadioButton("First frame: channel 1 (left) excitation")
        self.rb_first_ch2 = QRadioButton("First frame: channel 2 (right) excitation")
        self.rb_first_ch1.setChecked(True)
        self.rb_first_ch1.setEnabled(False)
        self.rb_first_ch2.setEnabled(False)
        alex_layout.addWidget(self.chk_alex)
        alex_layout.addWidget(self.rb_first_ch1)
        alex_layout.addWidget(self.rb_first_ch2)
        control_layout.addWidget(alex_group)
        self.chk_alex.toggled.connect(self.on_excitation_change)
        self.rb_first_ch1.toggled.connect(self.on_excitation_change)

        # Propagation Parameter Configuration
        self.chk_propagate = QCheckBox("Propagate clicks to other channel")
        self.chk_propagate.setEnabled(False)
        control_layout.addWidget(self.chk_propagate)

        # Core Pipeline Buttons
        self.btn_pick_file = QPushButton("Load TIF Stack File")
        self.btn_pick_file.clicked.connect(self.pick_file)
        self.btn_clear = QPushButton("Clear Clicks")
        self.btn_clear.clicked.connect(self.clear_plot)
        self.btn_refresh = QPushButton("Refresh Plot")
        self.btn_refresh.clicked.connect(self.refresh_plot)
        
        control_layout.addWidget(self.btn_pick_file)
        control_layout.addWidget(self.btn_clear)
        control_layout.addWidget(self.btn_refresh)

        # Signal Filtering Customization
        ck_group = QGroupBox("Chung-Kennedy Filter?")
        ck_lay = QHBoxLayout()
        self.ck_yes, self.ck_no = QRadioButton("Yes"), QRadioButton("No")
        self.ck_yes.setChecked(True)
        ck_lay.addWidget(self.ck_yes); ck_lay.addWidget(self.ck_no)
        ck_group.setLayout(ck_lay)
        control_layout.addWidget(ck_group)

        stoic_group = QGroupBox("Estimate stoichiometry?")
        stoic_lay = QHBoxLayout()
        self.stoic_yes, self.stoic_no = QRadioButton("Yes"), QRadioButton("No")
        self.stoic_no.setChecked(True)
        stoic_lay.addWidget(self.stoic_yes); stoic_lay.addWidget(self.stoic_no)
        stoic_group.setLayout(stoic_lay)
        control_layout.addWidget(stoic_group)

        control_layout.addWidget(QLabel("Isingle (raw intensity units):"))
        self.isingle_box = QLineEdit("120")
        self.isingle_box.setToolTip("Raw intensity units; used for enabled Isingle guides and stoichiometry.")
        control_layout.addWidget(self.isingle_box)
        control_layout.addWidget(QLabel("Isingle guide baseline offset (raw units):"))
        self.isingle_offset_box = QLineEdit("0")
        self.isingle_offset_box.setToolTip("Guide positions = offset + k × Isingle. Changes guides only, not measured intensities.")
        control_layout.addWidget(self.isingle_offset_box)
        self.stoic_summary = QLabel("")
        self.stoic_summary.setWordWrap(True)
        control_layout.addWidget(self.stoic_summary)

        appearance = QGroupBox("Plot appearance")
        appearance_layout = QFormLayout(appearance)
        self.plot_title_box = QLineEdit("")
        appearance_layout.addRow("Title:", self.plot_title_box)
        self._default_x_label = "Frame number"
        self.x_label_box = QLineEdit(self._default_x_label)
        self.y_label_box = QLineEdit("Intensity (x10$^3$)")
        for box in (self.x_label_box, self.y_label_box):
            box.setToolTip("Type the label directly, with $...$ for maths; no Python quotes or r prefix.")
            box.editingFinished.connect(self.refresh_plot)
        appearance_layout.addRow("X-axis label:", self.x_label_box)
        appearance_layout.addRow("Y-axis label:", self.y_label_box)
        self.chk_main_guides = QCheckBox("Show Isingle guides in main plot")
        self.chk_legend = QCheckBox("Show legend")
        self.chk_legend.setChecked(True)
        self.chk_x_ticks = QCheckBox("Show x ticks")
        self.chk_y_ticks = QCheckBox("Show y ticks")
        self.chk_x_ticks.setChecked(True)
        self.chk_y_ticks.setChecked(True)
        for control in (self.chk_legend, self.chk_x_ticks, self.chk_y_ticks, self.chk_main_guides):
            appearance_layout.addRow(control)
        control_layout.addWidget(appearance)
        
        self.main_style_boxes = {}
        main_style_group = QGroupBox("Main plot style")
        main_style_layout = QFormLayout(main_style_group)
        self.inset_style_boxes = {}
        inset_group = QGroupBox("Zoom inset")
        inset_layout = QFormLayout(inset_group)
        self.btn_select_zoom = QPushButton("Select zoom region")
        self.btn_select_zoom.setCheckable(True)
        self.btn_select_zoom.toggled.connect(self._toggle_zoom_selection)
        self.btn_clear_zoom = QPushButton("Remove zoom inset")
        self.btn_clear_zoom.clicked.connect(self.clear_zoom_inset)
        inset_layout.addRow(self.btn_select_zoom)
        inset_layout.addRow(self.btn_clear_zoom)
        self.chk_inset_ck = QCheckBox("Show CK line over raw crosses")
        self.chk_inset_ck.setChecked(True)
        self.chk_inset_x_ticks = QCheckBox("Show inset x ticks")
        self.chk_inset_y_ticks = QCheckBox("Show inset y ticks")
        self.chk_inset_x_ticks.setChecked(True)
        self.chk_inset_y_ticks.setChecked(True)
        self.chk_inset_guides = QCheckBox("Show Isingle guides in inset")
        for control in (self.chk_inset_ck, self.chk_inset_x_ticks, self.chk_inset_y_ticks, self.chk_inset_guides):
            inset_layout.addRow(control)
            control.toggled.connect(self.refresh_plot)
        for boxes, layout, font_size in ((self.main_style_boxes, main_style_layout, "10"),
                                          (self.inset_style_boxes, inset_layout, "8")):
            for key, label, default in (("line_width", "Line width:", "1.8"),
                                         ("cross_size", "Cross size:", "4"),
                                         ("cross_width", "Cross stroke width:", "1"),
                                         ("font_size", "Font size:", font_size)):
                box = QLineEdit(default)
                boxes[key] = box
                layout.addRow(label, box)
                box.editingFinished.connect(self.refresh_plot)
        control_layout.addWidget(main_style_group)
        control_layout.addWidget(inset_group)

        # Storage & Export Selection
        export_group = QGroupBox("Export Selection")
        export_lay = QVBoxLayout()
        self.save_ch1 = QRadioButton("Ch 1 (Left/Single) Only")
        self.save_ch2 = QRadioButton("Ch 2 (Right) Only")
        self.save_both = QRadioButton("Save Both Channels")
        self.save_ch1.setChecked(True)
        export_lay.addWidget(self.save_ch1)
        export_lay.addWidget(self.save_ch2)
        export_lay.addWidget(self.save_both)
        export_group.setLayout(export_lay)
        control_layout.addWidget(export_group)

        control_layout.addWidget(QLabel("Output Base Name:"))
        self.outname_box = QLineEdit("clickmode_output")
        control_layout.addWidget(self.outname_box)
        self.btn_save = QPushButton("Save Trajectories")
        self.btn_save.clicked.connect(self.save_data)
        control_layout.addWidget(self.btn_save)
        
        control_layout.addStretch()
        control_layout.addWidget(QLabel("Plot filename:"))
        self.plot_filename_box = QLineEdit("clickmode_plot.png")
        control_layout.addWidget(self.plot_filename_box)
        self.btn_export_plot = QPushButton("Export current plot")
        self.btn_export_plot.setMinimumHeight(48)
        self.btn_export_plot.setStyleSheet("font-weight: bold; font-size: 15px; padding: 10px;")
        self.btn_export_plot.clicked.connect(self.export_current_plot)
        control_layout.addWidget(self.btn_export_plot)

        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setWidget(controls_widget)
        self.sidebar_scroll.setMinimumWidth(controls_widget.sizeHint().width() + 24)
        main_layout.addWidget(self.sidebar_scroll, stretch=1)
        for control in (self.ck_yes, self.stoic_yes, self.chk_legend, self.chk_x_ticks, self.chk_y_ticks, self.chk_main_guides):
            control.toggled.connect(self.refresh_plot)
        self.plot_title_box.editingFinished.connect(self.refresh_plot)
        self.isingle_box.editingFinished.connect(self.refresh_plot)
        self.isingle_offset_box.editingFinished.connect(self.refresh_plot)
        self.apply_axes_layout()

    def on_mode_change(self):
        """Triggers dynamic, on-the-fly channel segmenting or joining from stored stack memory."""
        self.chk_propagate.setEnabled(self.rb_dual.isChecked())
        self._update_excitation_controls()
        if self.rb_dual.isChecked():
            self.save_both.setChecked(True)
        else:
            self.save_ch1.setChecked(True)
            
        self.apply_axes_layout()
        
        if self.raw_stack is not None and self.loaded_fname is not None:
            self.process_image_channels()
        else:
            self.clear_plot()

    def apply_axes_layout(self):
        """Adjusts subplot visual sizing profiles without instantiating or destroying underlying Axes."""
        if self.rb_single.isChecked():
            self.ax2.set_visible(False)
            self.ax1.set_position([0.05, 0.15, 0.4, 0.75])
            self.ax3.set_position([0.55, 0.15, 0.4, 0.75])
        else:
            self.ax2.set_visible(True)
            self.ax1.set_position([0.05, 0.15, 0.28, 0.75])
            self.ax2.set_position([0.36, 0.15, 0.28, 0.75])
            self.ax3.set_position([0.69, 0.15, 0.28, 0.75])
        if self.inset_bounds is not None and self.chk_legend.isChecked():
            pos = self.ax3.get_position()
            self.ax3.set_position([pos.x0, 0.32, pos.width, 0.58])
        self.canvas.draw_idle()

    def create_pystachio_image(self, array_data, filename):
        """Constructs an isolated, native PySTACHIO ImageData structure map."""
        img = images.ImageData()
        img.read(filename, self._channel_params())
        img.pixel_data = array_data
        img.num_frames = array_data.shape[0]
        img.frame_size = [array_data.shape[2], array_data.shape[1]]
        img.has_mask, img.mask_data = True, np.ones((array_data.shape[1], array_data.shape[2]))
        return img

    def pick_file(self):
        """Ingests raw file path reference from file explorer dialog."""
        fname, _ = QFileDialog.getOpenFileName(self, "Open TIF Stack File", filter="TIFF Files (*.tif *.tiff)")
        if not fname or not MODULES_LOADED: return
        
        if self.parent_suite: 
            self.parent_suite.reset_entire_suite()
            
        self.loaded_fname = fname
        self.params.name = str(Path(fname).with_suffix(""))
        try:
            raw = tf.imread(fname)
            if raw.ndim < 3: 
                raw = raw[np.newaxis, :, :]
            self.raw_stack = raw
            
            self.process_image_channels()
            
            
        except Exception as e:
            show_popup(self, "Ingestion Error", f"Failed to open image file:\n{e}", critical=True)

    def _alex_enabled(self):
        return self.rb_dual.isChecked() and self.chk_alex.isChecked()

    def _update_excitation_controls(self):
        self.chk_alex.setEnabled(self.rb_dual.isChecked())
        for control in (self.rb_first_ch1, self.rb_first_ch2):
            control.setEnabled(self._alex_enabled())
        default_label = "ALEX cycle number" if self._alex_enabled() else "Frame number"
        if self.x_label_box.text() == self._default_x_label:
            self.x_label_box.setText(default_label)
        self._default_x_label = default_label

    def on_excitation_change(self, *_args):
        self._update_excitation_controls()
        if self.raw_stack is not None and self.loaded_fname is not None:
            self.process_image_channels()
        else:
            self.clear_plot()

    def _channel_params(self):
        params = copy.deepcopy(self.params)
        # Images supplied to spot measurement are already split/deinterleaved.
        params.ALEX = False
        params.colocalize = False
        if hasattr(params, 'frame_time'):
            params.frame_time = self.params.frame_time * (2 if self._alex_enabled() else 1)
        return params

    def process_image_channels(self):
        """Split emission sides and, for ALEX, retain each side's excitation phase."""
        if self.raw_stack is None or self.loaded_fname is None:
            return
        raw = self.raw_stack
        indices = np.arange(len(raw))
        dropped = 0
        if self.rb_single.isChecked():
            ch1, ch2 = raw, None
            mapping = {'ch1': indices}
        else:
            mid = raw.shape[2] // 2
            ch1, ch2 = raw[:, :, :mid], raw[:, :, mid:]
            mapping = {'ch1': indices, 'ch2': indices}
            if self._alex_enabled():
                pairs = len(raw) // 2
                dropped = len(raw) % 2
                first = 0 if self.rb_first_ch1.isChecked() else 1
                mapping = {'ch1': indices[first::2][:pairs],
                           'ch2': indices[1-first::2][:pairs]}
                ch1 = ch1[mapping['ch1']]
                ch2 = ch2[mapping['ch2']]
        # Reset the slider before drawing a shorter, deinterleaved stack.
        self.active_frame = 0
        self.img_ch1 = self.img_ch2 = None
        self.image_data = None
        self.clear_plot()
        self.raw_frame_indices = mapping
        self.dropped_raw_frames = dropped
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(0)
        self.frame_slider.setMaximum(max(0, len(ch1)-1))
        self.frame_slider.setEnabled(bool(len(ch1)))
        self.frame_slider.blockSignals(False)
        if not len(ch1):
            self.frame_label.setText("ALEX requires at least two raw frames")
            self.canvas.draw_idle()
            return
        self.img_ch1 = self.create_pystachio_image(np.ascontiguousarray(ch1), self.loaded_fname)
        if ch2 is not None:
            self.img_ch2 = self.create_pystachio_image(np.ascontiguousarray(ch2), self.loaded_fname)
        self.image_data = self.img_ch1
        self.update_frame()
        self.canvas.draw_idle()

    def update_frame(self, val=None):
        if val is not None: 
            self.active_frame = val
            
        if not self.img_ch1: 
            return
            
        self.active_frame = min(max(0, self.active_frame), self.img_ch1.num_frames - 1)
        if self._alex_enabled():
            left = self.raw_frame_indices['ch1'][self.active_frame]
            right = self.raw_frame_indices['ch2'][self.active_frame]
            suffix = " | final unpaired raw frame omitted" if self.dropped_raw_frames else ""
            self.frame_label.setText(f"ALEX cycle: {self.active_frame} / {self.frame_slider.maximum()} | Raw frames: Ch1={left}, Ch2={right}{suffix}")
        else:
            self.frame_label.setText(f"Frame: {self.active_frame} / {self.frame_slider.maximum()}")
        
        # Render Channel 1 Local Viewport
        self.ax1.clear()
        self.ax1.imshow(self.img_ch1[self.active_frame].as_image(), cmap='Greys_r')
        self.ax1.set_title("Ch1: left / Ch1 excitation" if self._alex_enabled() else "Channel 1 (Left)" if self.rb_dual.isChecked() else "Channel 1")
        
        # Render Channel 2 Local Viewport
        if self.rb_dual.isChecked() and self.img_ch2:
            self.ax2.clear()
            self.ax2.imshow(self.img_ch2[self.active_frame].as_image(), cmap='Greys_r')
            self.ax2.set_title("Ch2: right / Ch2 excitation" if self._alex_enabled() else "Channel 2 (Right)")

        # Map all global spots explicitly with their unique Global IDs
        for event in self.click_events:
            track_id = event["id"]
            if event["ch1"] is not None:
                cx, cy = event["ch1"]
                self.ax1.scatter(cx, cy, marker='x', color='red')
                self.ax1.text(cx + 2, cy + 2, str(track_id + 1), color='red', fontsize=8, fontweight='bold')
                
            if self.rb_dual.isChecked() and self.img_ch2 and event["ch2"] is not None:
                cx, cy = event["ch2"]
                self.ax2.scatter(cx, cy, marker='x', color='blue')
                self.ax2.text(cx + 2, cy + 2, str(track_id + 1), color='blue', fontsize=8, fontweight='bold')

        self.canvas.draw_idle()

    def onclick(self, event):
        if event.inaxes not in (self.ax1, self.ax2) or self.btn_select_zoom.isChecked():
            return
        if event.xdata is None or event.ydata is None or not self.img_ch1 or not MODULES_LOADED: return
        
        mode = getattr(self.toolbar, 'mode', '')
        if hasattr(mode, 'value'): mode = mode.value
        if mode != "": return

        do_propagate = self.chk_propagate.isChecked()
        reg_mat = getattr(self.parent_suite, 'registration_matrix', None)

        # Generate a globally unique ID for this explicit click action
        current_id = self.global_track_id
        self.global_track_id += 1        
        click_record = {"id": current_id, "frame": self.active_frame, "ch1": None, "ch2": None}

        # Path A: Click Action registered on Channel 1
        if event.inaxes == self.ax1:
            rx, ry = self.refine_point(event.xdata, event.ydata, self.img_ch1)
            click_record["ch1"] = [rx, ry]
            
            if self.rb_dual.isChecked() and do_propagate and self.img_ch2 is not None:
                if reg_mat is not None:
                    px, py = np.dot(reg_mat, [rx, ry, 1.0])[:2]
                else:
                    px, py = rx, ry
                rx2, ry2 = self.refine_point(px, py, self.img_ch2)
                click_record["ch2"] = [rx2, ry2]

        # Path B: Click Action registered on Channel 2
        elif self.rb_dual.isChecked() and event.inaxes == self.ax2 and self.img_ch2 is not None:
            rx2, ry2 = self.refine_point(event.xdata, event.ydata, self.img_ch2)
            click_record["ch2"] = [rx2, ry2]
            
            if do_propagate:
                if reg_mat is not None:
                    try:
                        inv_mat = cv2.invertAffineTransform(reg_mat[:2, :])
                        px, py = np.dot(inv_mat, [rx2, ry2, 1.0])
                    except Exception as matrix_err:
                        print(f"Matrix inversion failure: {matrix_err}. Defaulting to identity transfer.")
                        px, py = rx2, ry2
                else:
                    px, py = rx2, ry2
                rx, ry = self.refine_point(px, py, self.img_ch1)
                click_record["ch1"] = [rx, ry]
        else:
            return

        self.click_events.append(click_record)

        # Rebuild trajectories and stamp them with the Global Tracking ID
        ch1_events = [e for e in self.click_events if e["ch1"] is not None]
        self.coords_ch1 = [e["ch1"] for e in ch1_events]
        self.trajs_ch1 = self.extract_trajectories(ch1_events, self.img_ch1, "ch1")

        if self.rb_dual.isChecked():
            ch2_events = [e for e in self.click_events if e["ch2"] is not None]
            self.coords_ch2 = [e["ch2"] for e in ch2_events]
            self.trajs_ch2 = self.extract_trajectories(ch2_events, self.img_ch2, "ch2")
                    
        self.coords = self.coords_ch1
        self.trajs = self.trajs_ch1
        
        self.refresh_plot()

    def refine_point(self, x, y, target_image_object):
        """Utilizes PySTACHIO engine mechanics to find local sub-pixel spot centroids."""
        s = spots.Spots(frame=self.active_frame)
        s.set_positions(np.array([[x, y]]))
        s.num_spots = 1
        s.refine_centres(target_image_object[self.active_frame], self._channel_params())
        
        if s.positions is not None and len(s.positions) > 0:
            return float(s.positions[0][0]), float(s.positions[0][1])
        return float(x), float(y)

    def extract_trajectories(self, click_events, target_image_object, channel_key):
        """Assembles continuous multi-frame intensity data from an array of explicit click events."""
        if not click_events: return []
        
        all_trajs = []
        channel_params = self._channel_params()
        channel_params.num_frames = target_image_object.num_frames
        
        for event in click_events:
            origin_frame = event["frame"]
            coord = event[channel_key]
            
            # 1. Establish master properties for THIS SPOT on its specific origin frame
            master_spot = spots.Spots(frame=origin_frame)
            master_spot.set_positions(np.array([coord]))
            
            active_img = target_image_object[origin_frame]
            active_img_array = active_img.as_image()[:, :]
            
            # Calculate structure and noise metrics ONLY on the frame where it was clicked
            master_spot.refine_centres(active_img, channel_params)
            master_spot.get_spot_widths(active_img_array, channel_params)
            master_spot.get_precision(active_img_array, channel_params)
            
            spot_frames = []
            
            # 2. Extract intensities across all frames for this single spot
            for f in range(target_image_object.num_frames):
                s = spots.Spots(frame=f)
                
                # Lock positions to the refined master coordinates
                s.set_positions(np.copy(master_spot.positions))
                
                # Propagate the structural and noise metrics calculated from the origin frame
                # s.bg_intensity = np.copy(master_spot.bg_intensity)
                s.snr = np.copy(master_spot.snr)
                s.noise = np.copy(master_spot.noise)
                s.width = np.copy(master_spot.width) # s.get_spot_widths(target_image_object[origin_frame].as_image(), self.params)
                s.precision = np.copy(master_spot.precision)
                
                # ONLY calculate the intensity for the current frame
                img_array = target_image_object[f].as_image()[:, :]
                s.get_spot_intensities(img_array, channel_params)
                
                spot_frames.append(s)
                
            # Build the trajectory for this individual spot and apply its Global ID
            traj_list = trajectories.build_trajectories(spot_frames, channel_params)
            if traj_list:
                traj = traj_list[0]
                traj.id = event["id"]
                all_trajs.append(traj)   
        return all_trajs
    
    def _plot_settings(self):
        value = self.isingle_box.text().strip()
        isingle = None
        if value:
            try:
                isingle = float(value)
            except ValueError:
                raise ValueError("Isingle must be a positive number, or blank to hide guides.") from None
            if not np.isfinite(isingle) or isingle <= 0:
                raise ValueError("Isingle must be finite and greater than zero.")
        if (self.stoic_yes.isChecked() or self.chk_main_guides.isChecked() or self.chk_inset_guides.isChecked()) and isingle is None:
            raise ValueError("Enter Isingle for the enabled guides or stoichiometry estimate.")
        try:
            offset = float(self.isingle_offset_box.text().strip())
        except ValueError:
            raise ValueError("Isingle guide offset must be a finite number.") from None
        if not np.isfinite(offset):
            raise ValueError("Isingle guide offset must be a finite number.")
        return dict(filtered=self.ck_yes.isChecked(), isingle=isingle, guide_offset=offset,
                    main_guides=self.chk_main_guides.isChecked(),
                    stoic=self.stoic_yes.isChecked(), title=self.plot_title_box.text(),
                    x_label=self.x_label_box.text(), y_label=self.y_label_box.text(),
                    legend=self.chk_legend.isChecked(), x_ticks=self.chk_x_ticks.isChecked(),
                    y_ticks=self.chk_y_ticks.isChecked(),
                    main_style=self._read_plot_style(self.main_style_boxes, "Main plot"),
                    inset_style=self._read_plot_style(self.inset_style_boxes, "Inset"),
                    inset_ck=self.chk_inset_ck.isChecked(),
                    inset_x_ticks=self.chk_inset_x_ticks.isChecked(),
                    inset_y_ticks=self.chk_inset_y_ticks.isChecked(),
                    inset_guides=self.chk_inset_guides.isChecked())

    def _draw_trajectory_plot(self, ax, channel_groups, settings):
        # Prepare everything before clearing the existing graph, so bad input
        # or a filtering error leaves the previous plot intact.
        records = []
        raw_max = None
        for channel, trajs in channel_groups:
            for traj in trajs:
                raw = np.asarray(traj.intensity, dtype=float).ravel()[:self.img_ch1.num_frames]
                if not raw.size:
                    continue
                finite = raw[np.isfinite(raw)]
                if finite.size:
                    raw_max = float(finite.max()) if raw_max is None else max(raw_max, float(finite.max()))
                filtered = None
                if settings['filtered'] or (self.inset_bounds is not None and settings['inset_ck']):
                    # The backend leaves its final sample uncomputed; retain
                    # the existing exclusion of that sample in the line.
                    filtered = np.asarray(postprocessing.chung_kennedy_filter(
                        raw, self.params.chung_kennedy_window, 1)[0], dtype=float).ravel()[:-1]
                records.append((channel, traj.id, raw, filtered))
        levels = np.array([], dtype=float)
        if (settings['main_guides'] or (self.inset_bounds is not None and settings['inset_guides'])) and settings['isingle'] is not None and raw_max is not None:
            ratio = (raw_max - settings['guide_offset']) / settings['isingle']
            if not np.isfinite(ratio) or ratio >= 10000:
                raise ValueError("Isingle would create over 10,000 guide lines; enter a larger value.")
            count = max(1, int(np.floor(ratio)) + 1)
            levels = (settings['guide_offset'] + np.arange(1, count + 1, dtype=float)
                      * settings['isingle']) / 1000.0

        if ax is self.ax3:
            self._disconnect_zoom_selector()
        for child in list(ax.child_axes):
            child.remove()
        ax.clear()
        main_style = settings['main_style']
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        seen = set()
        summaries = []
        for channel, track_id, raw, filtered in records:
            color = colors[int(track_id) % len(colors)]
            label = f"Clicked trajectory {int(track_id) + 1}"
            legend_label = label if track_id not in seen else '_nolegend_'
            seen.add(track_id)
            style = '-'
            if settings['filtered'] and filtered is not None:
                ax.plot(np.arange(len(raw)), raw / 1000.0, linestyle='None', marker='x',
                        markersize=main_style['cross_size'], markeredgewidth=main_style['cross_width'], color=color, alpha=0.5, zorder=2, label='_nolegend_')
                ax.plot(np.arange(len(filtered)), filtered / 1000.0, linestyle=style,
                        color=color, linewidth=main_style['line_width'], zorder=3, label=legend_label)
            else:
                ax.plot(np.arange(len(raw)), raw / 1000.0, linestyle=style,
                        color=color, linewidth=main_style['line_width'], label=legend_label)
            if settings['stoic']:
                summaries.append(f"{label} ({channel}): {np.mean(raw[:3]) / settings['isingle']:.1f}")
        for level in levels if settings['main_guides'] else []:
            ax.axhline(level, color='black', linestyle=':', linewidth=0.8,
                       zorder=1, label='_nolegend_')
        ax.set_title(settings['title'], fontsize=main_style['font_size'])
        ax.set_xlabel(settings['x_label'], fontsize=main_style['font_size'])
        ax.set_ylabel(settings['y_label'], fontsize=main_style['font_size'])
        ax.tick_params(labelsize=main_style['font_size'])
        ax.xaxis.get_offset_text().set_fontsize(main_style['font_size'])
        ax.yaxis.get_offset_text().set_fontsize(main_style['font_size'])
        ax.tick_params(axis='x', which='both', bottom=settings['x_ticks'], top=False,
                       labelbottom=settings['x_ticks'], labeltop=False)
        ax.tick_params(axis='y', which='both', left=settings['y_ticks'], right=False,
                       labelleft=settings['y_ticks'], labelright=False)
        ax.xaxis.get_offset_text().set_visible(settings['x_ticks'])
        ax.yaxis.get_offset_text().set_visible(settings['y_ticks'])
        if self.inset_bounds is not None and records:
            self._draw_zoom_inset(ax, records, levels, settings)
        if settings['legend'] and records:
            if self.inset_bounds is not None:
                # Outside the plotting axes: never obscures the inset.
                ax.legend(fontsize=main_style['font_size'], loc='upper left',
                          bbox_to_anchor=(0, -0.24), borderaxespad=0)
            else:
                ax.legend(fontsize=main_style['font_size'], loc='upper right')
        if ax is self.ax3:
            self._install_zoom_selector()
        return summaries

    def _displayed_channels(self):
        groups = [('ch1', self.trajs_ch1)]
        if self.rb_dual.isChecked():
            groups.append(('ch2', self.trajs_ch2))
        return groups

    def refresh_plot(self, *_args):
        if not self.img_ch1 or not MODULES_LOADED:
            return
        try:
            settings = self._plot_settings()
            summaries = self._draw_trajectory_plot(self.ax3, self._displayed_channels(), settings)
        except Exception as exc:
            show_popup(self, "Plot settings", str(exc), critical=True)
            return
        self.params.num_frames = self.img_ch1.num_frames
        self.params.chung_kennedy = settings['filtered']
        if settings['isingle'] is not None:
            self.params.Isingle = settings['isingle']
        self.stoic_summary.setText("Estimated stoichiometry:\n" + "\n".join(summaries) if summaries else "")
        self.apply_axes_layout()
        self.update_frame()
        self.canvas.draw_idle()

    def export_current_plot(self):
        if not self.img_ch1 or not any(trajs for _, trajs in self._displayed_channels()):
            show_popup(self, "No plot", "Click a spot to plot a trajectory first.")
            return
        name = self.plot_filename_box.text().strip()
        if not name:
            show_popup(self, "Plot filename", "Enter a filename for the plot.", critical=True)
            return
        path = Path(name).expanduser()
        if not path.suffix:
            path = path.with_suffix('.png')
        if not path.is_absolute():
            path = (Path(self.loaded_fname).parent if self.loaded_fname else Path.cwd()) / path
        if path.exists() and QMessageBox.question(
            self, "Replace plot?", f"Replace {path.name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        figure = None
        try:
            settings = self._plot_settings()
            figure, ax = plt.subplots(figsize=(7, 5))
            self._draw_trajectory_plot(ax, self._displayed_channels(), settings)
            # Match the displayed view, including toolbar zoom/pan.
            ax.set_xlim(self.ax3.get_xlim())
            ax.set_ylim(self.ax3.get_ylim())
            figure.tight_layout()
            figure.savefig(path, dpi=300, bbox_inches='tight')
        except Exception as exc:
            show_popup(self, "Plot export error", str(exc), critical=True)
            return
        finally:
            if figure is not None:
                plt.close(figure)
        show_popup(self, "Plot exported", f"Saved plot to:\n{path}")

    def clear_plot(self):
        self._disconnect_zoom_selector()
        self.inset_bounds = None
        self.btn_select_zoom.setChecked(False)
        for child in list(self.ax3.child_axes):
            child.remove()
        self.global_track_id = 0
        self.click_events = []
        self.coords_ch1, self.coords_ch2, self.coords = [], [], []
        self.trajs_ch1, self.trajs_ch2, self.trajs = [], [], []
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.stoic_summary.clear()
        self.update_frame()
        self.canvas.draw_idle()

    def save_data(self):
        if not self.params.name or not MODULES_LOADED:
            return
        base_dir = Path(self.loaded_fname or self.params.name).parent
        base_name = base_dir / self.outname_box.text().strip()
        do_ch1 = self.save_ch1.isChecked() or self.save_both.isChecked()
        do_ch2 = self.save_ch2.isChecked() or self.save_both.isChecked()
        groups = []
        if do_ch1:
            groups.append(('ch1', self.trajs_ch1))
        if do_ch2 and self.rb_dual.isChecked():
            groups.append(('ch2', self.trajs_ch2))
        figure = None
        try:
            settings = self._plot_settings()
            figure, ax = plt.subplots(figsize=(7, 5))
            self._draw_trajectory_plot(ax, groups, settings)
            for channel, trajs in groups:
                if trajs:
                    trajectories.write_trajectories(trajs, f"{base_name}_{channel}.tsv")
            figure.tight_layout()
            figure.savefig(f"{base_name}_plot.png", dpi=300, bbox_inches='tight')
        except Exception as exc:
            show_popup(self, "Export Error", str(exc), critical=True)
            return
        finally:
            if figure is not None:
                plt.close(figure)
        show_popup(self, "Export Success", f"Saved trajectories and plot for selected channels to:\n{base_dir}")

    @staticmethod
    def _read_plot_style(boxes, label):
        values = {}
        for key, box in boxes.items():
            try:
                value = float(box.text())
            except ValueError:
                raise ValueError(f"{label} {key.replace('_', ' ')} must be a positive number.") from None
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{label} {key.replace('_', ' ')} must be finite and positive.")
            values[key] = value
        return values

    @staticmethod
    def _nice_end_ticks(low, high):
        # Use the outermost nice locator ticks *inside* the selected range.
        ticks = MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10]).tick_values(low, high)
        ticks = ticks[(ticks >= low) & (ticks <= high)]
        if len(ticks) >= 2:
            return [float(ticks[0]), float(ticks[-1])]
        return [low, high]

    def _draw_zoom_inset(self, ax, records, levels, settings):
        xmin, xmax, ymin, ymax = self.inset_bounds
        style = settings['inset_style']
        inset = ax.inset_axes([0.51, 0.56, 0.46, 0.40], zorder=6)
        inset.set_facecolor('white')
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for channel, track_id, raw, filtered in records:
            color = colors[int(track_id) % len(colors)]
            # Draw the original full-frame filter, then crop the view. Filtering
            # only the selected fragment would change the boundary estimates.
            inset.plot(np.arange(len(raw)), raw / 1000.0, linestyle='None', marker='x',
                       markersize=style['cross_size'], markeredgewidth=style['cross_width'],
                       color=color, alpha=0.5, zorder=2)
            if settings['inset_ck'] and filtered is not None:
                inset.plot(np.arange(len(filtered)), filtered / 1000.0,
                           linestyle='-', color=color,
                           linewidth=style['line_width'], zorder=3)
        if settings['inset_guides']:
            for level in levels:
                if ymin <= level <= ymax:
                    inset.axhline(level, color='black', linestyle=':', linewidth=0.8, zorder=1)
        inset.set_xlim(xmin, xmax)
        inset.set_ylim(ymin, ymax)
        inset.set_xticks(self._nice_end_ticks(xmin, xmax) if settings['inset_x_ticks'] else [])
        inset.set_yticks(self._nice_end_ticks(ymin, ymax) if settings['inset_y_ticks'] else [])
        inset.xaxis.set_minor_locator(NullLocator())
        inset.yaxis.set_minor_locator(NullLocator())
        inset.tick_params(labelsize=style['font_size'])
        inset.xaxis.get_offset_text().set_fontsize(style['font_size'])
        inset.yaxis.get_offset_text().set_fontsize(style['font_size'])
        ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                               fill=False, edgecolor='0.35', linewidth=0.8, zorder=4))

    def _disconnect_zoom_selector(self):
        if self.zoom_selector is not None:
            self.zoom_selector.set_active(False)
            self.zoom_selector.disconnect_events()
            self.zoom_selector = None

    def _install_zoom_selector(self):
        self._disconnect_zoom_selector()
        self.zoom_selector = RectangleSelector(
            self.ax3, self._on_zoom_selected, button=[1], minspanx=5, minspany=5,
            spancoords='pixels', useblit=False, interactive=False,
            props=dict(facecolor='tab:blue', edgecolor='tab:blue', alpha=0.15),
        )
        self.zoom_selector.set_active(self.btn_select_zoom.isChecked())

    def _toggle_zoom_selection(self, enabled):
        if enabled:
            mode = getattr(self.toolbar.mode, 'value', self.toolbar.mode)
            if mode == 'pan/zoom':
                self.toolbar.pan()
            elif mode == 'zoom rect':
                self.toolbar.zoom()
            self._install_zoom_selector()
        elif self.zoom_selector is not None:
            self.zoom_selector.set_active(False)

    def _on_zoom_selected(self, press, release):
        if not any(trajs for _, trajs in self._displayed_channels()):
            return
        values = (press.xdata, release.xdata, press.ydata, release.ydata)
        if any(value is None for value in values) or not np.isfinite(values).all():
            return
        xmin, xmax = sorted(values[:2])
        ymin, ymax = sorted(values[2:])
        if xmin == xmax or ymin == ymax:
            return
        limits = (self.ax3.get_xlim(), self.ax3.get_ylim())
        self.inset_bounds = (xmin, xmax, ymin, ymax)
        self.btn_select_zoom.setChecked(False)
        self.refresh_plot()
        self.ax3.set_xlim(limits[0])
        self.ax3.set_ylim(limits[1])
        self.canvas.draw_idle()

    def clear_zoom_inset(self):
        self.inset_bounds = None
        self.btn_select_zoom.setChecked(False)
        self.refresh_plot()
