#!/usr/bin/env simnibs_python
"""Multipolar exhaustive-search GUI tab."""

import json
import os

from PyQt5 import QtCore, QtWidgets

from tit.config_io import write_config_json
from tit.gui.confirmation_dialog import ConfirmationDialog
from tit.gui.ex_search_tab import ExSearchEngine, ExSearchTab, ExSearchThread
from tit.gui.utils import confirm_overwrite
from tit.opt.config import MExConfig
from tit.opt.mex.engine import safe_mex_run_name
from tit.opt.mex.logic import MEX_BUCKET_KEYS, count_multipolar_combinations


MEX_BUCKET_LABELS = {
    "e1_plus": "E1+",
    "e1_minus": "E1-",
    "e2_plus": "E2+",
    "e2_minus": "E2-",
    "e3_plus": "E3+",
    "e3_minus": "E3-",
    "e4_plus": "E4+",
    "e4_minus": "E4-",
}

MAX_UNSYMMETRIC_BUCKET_CANDIDATES = 1_000_000


class MExSearchTab(ExSearchTab):
    """Multipolar exhaustive search over four bipolar electrode pairs."""

    METRIC_LABELS = [
        ("Recursive TI", "recursive_ti"),
        ("Botzanowski magnitude AM", "botzanowski_magnitude_am"),
        ("Botzanowski directional AM", "botzanowski_directional_am"),
        ("Botzanowski directional AM avg", "botzanowski_directional_am_ti_avg"),
        ("Grossman extension directional AM", "grossman_ext_directional_am"),
        ("Grossman extension directional AM avg", "grossman_ext_directional_am_ti_avg"),
    ]

    def setup_ui(self):
        super().setup_ui()
        self.run_btn.setText("Run m-Ex-Search")
        self.stop_btn.setText("Stop m-Ex-Search")
        self.rb_all_combinations.hide()
        self.rb_all_combinations.setEnabled(False)
        self.rb_bucketed.setText("Bucketed Mode")
        self.rb_bucketed.setChecked(True)
        self._configure_current_panel()

    def _configure_current_panel(self):
        current_group = None
        for group in self.findChildren(QtWidgets.QGroupBox):
            if group.title() == "Current Configuration":
                current_group = group
                break
        if current_group is None:
            return
        current_group.setTitle("mTI Configuration")

        for label in current_group.findChildren(QtWidgets.QLabel):
            text = label.text()
            if text.startswith("Total Current"):
                label.setText("Pair Current (mA):")
                label.setFixedWidth(140)
            elif text.startswith("Current Step") or text.startswith("Channel Limit"):
                label.hide()
        self.total_current_spinbox.setValue(5.0)
        self.total_current_spinbox.setToolTip(
            "Fixed current applied to each of the four bipolar electrode pairs"
        )
        self.current_step_spinbox.hide()
        self.channel_limit_spinbox.hide()

        self.metric_combo = QtWidgets.QComboBox()
        for label, value in self.METRIC_LABELS:
            self.metric_combo.addItem(label, value)
        layout = current_group.layout()
        metric_layout = QtWidgets.QHBoxLayout()
        metric_label = QtWidgets.QLabel("mTI Metric:")
        metric_label.setFixedWidth(140)
        metric_layout.addWidget(metric_label)
        metric_layout.addWidget(self.metric_combo)
        metric_layout.addStretch()
        layout.insertLayout(1, metric_layout)

    def _build_bucketed_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        self.mex_bucket_inputs = {}
        for key in MEX_BUCKET_KEYS:
            field = QtWidgets.QLineEdit()
            field.hide()
            self.mex_bucket_inputs[key] = field

        self.e1_plus_input = self.mex_bucket_inputs["e1_plus"]
        self.e1_minus_input = self.mex_bucket_inputs["e1_minus"]
        self.e2_plus_input = self.mex_bucket_inputs["e2_plus"]
        self.e2_minus_input = self.mex_bucket_inputs["e2_minus"]

        self.bucket_summary_label = QtWidgets.QLabel("No m-ex-search bucket file loaded")
        self.bucket_summary_label.setWordWrap(True)
        self.bucket_summary_label.setStyleSheet("color: #666;")
        layout.addWidget(self.bucket_summary_label)

        self.symmetric_bucket_checkbox = QtWidgets.QCheckBox(
            "Force left/right symmetry"
        )
        self.symmetric_bucket_checkbox.setToolTip(
            "Only evaluate candidates satisfying the selected left/right symmetry relationship."
        )
        layout.addWidget(self.symmetric_bucket_checkbox)

        symmetry_layout = QtWidgets.QHBoxLayout()
        symmetry_label = QtWidgets.QLabel("Symmetry:")
        self.symmetry_pairing_combo = QtWidgets.QComboBox()
        self.symmetry_pairing_combo.addItem("Within each pair", "within_pairs")
        self.symmetry_pairing_combo.addItem("E1/E3 and E2/E4", "cross_pairs")
        self.symmetry_pairing_combo.setToolTip(
            "Choose which m-ex-search pairs should mirror each other when symmetry is enabled."
        )
        symmetry_layout.addWidget(symmetry_label)
        symmetry_layout.addWidget(self.symmetry_pairing_combo)
        symmetry_layout.addStretch()
        layout.addLayout(symmetry_layout)

        preset_layout = QtWidgets.QHBoxLayout()
        self.quadrant_buckets_btn = QtWidgets.QPushButton("Use Quadrants")
        self.quadrant_buckets_btn.hide()
        self.quadrant_buckets_btn.setEnabled(False)
        self.quadrant_buckets_btn.setToolTip(
            "Quadrant presets are currently only defined for regular Ex-Search."
        )
        self.load_buckets_btn = QtWidgets.QPushButton("Load Buckets")
        self.load_buckets_btn.setToolTip(
            "Load a JSON file with E1+/E1-/.../E4- bucket definitions"
        )
        self.load_buckets_btn.clicked.connect(self.load_buckets_from_file)
        self.save_buckets_btn = QtWidgets.QPushButton("Save Buckets")
        self.save_buckets_btn.hide()
        self.save_buckets_btn.setEnabled(False)
        preset_layout.addWidget(self.quadrant_buckets_btn)
        preset_layout.addWidget(self.load_buckets_btn)
        preset_layout.addWidget(self.save_buckets_btn)
        preset_layout.addStretch()
        layout.addLayout(preset_layout)
        layout.addStretch()

        self.electrode_stack.addWidget(panel)

    def _build_all_combinations_panel(self):
        panel = QtWidgets.QWidget()
        self.all_electrodes_input = QtWidgets.QLineEdit()
        layout = QtWidgets.QFormLayout(panel)
        layout.addRow("All electrodes:", self.all_electrodes_input)
        self.electrode_stack.addWidget(panel)

    def _current_bucket_inputs(self):
        return {
            key: self.parse_electrode_input(field.text()) or []
            for key, field in self.mex_bucket_inputs.items()
        }

    def _set_bucket_inputs(self, buckets, preset_label=""):
        for key, field in self.mex_bucket_inputs.items():
            field.setText(", ".join(buckets.get(key, [])))
        self.bucket_preset_label = preset_label
        counts = ", ".join(
            f"{MEX_BUCKET_LABELS[key]}: {len(buckets.get(key, []))}"
            for key in MEX_BUCKET_KEYS
        )
        label = preset_label or "Loaded buckets"
        self.bucket_summary_label.setText(f"{label}\n{counts}")
        self.rb_bucketed.setChecked(True)
        self._on_electrode_mode_changed()

    def load_buckets_from_file(self):
        start_dir = self.pm.config_dir() if self.pm.project_dir else os.getcwd()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load m-Ex-Search Buckets",
            start_dir,
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            buckets = {key: list(data.get(key, [])) for key in MEX_BUCKET_KEYS}
        except (OSError, json.JSONDecodeError, TypeError) as e:
            QtWidgets.QMessageBox.critical(
                self, "Load Buckets Failed", f"Could not load bucket file:\n{e}"
            )
            return
        if not all(buckets.values()):
            QtWidgets.QMessageBox.warning(
                self,
                "Incomplete Buckets",
                "The file must define non-empty E1+/E1-/.../E4- buckets.",
            )
            return
        self._set_bucket_inputs(buckets, preset_label=f"File: {os.path.basename(path)}")
        self.update_status(f"Loaded bucket file: {os.path.basename(path)}")

    def save_buckets_to_file(self):
        buckets = self._current_bucket_inputs()
        if not all(buckets.values()):
            QtWidgets.QMessageBox.warning(
                self,
                "Incomplete Buckets",
                "Enter valid electrodes in all eight bucket fields before saving.",
            )
            return
        start_dir = self.pm.config_dir() if self.pm.project_dir else os.getcwd()
        default_path = os.path.join(start_dir, "m_ex_search_buckets.json")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save m-Ex-Search Buckets",
            default_path,
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(buckets, f, indent=2)
        except OSError as e:
            QtWidgets.QMessageBox.critical(
                self, "Save Buckets Failed", f"Could not save bucket file:\n{e}"
            )
            return
        self.update_status(f"Saved bucket file: {os.path.basename(path)}")

    def validate_inputs(self):
        if self.subject_combo.currentText() == "":
            self.update_status("Please select a subject", error=True)
            return False
        selected_items = self.leadfield_list.selectedItems()
        if not selected_items or not selected_items[0].data(QtCore.Qt.UserRole):
            self.update_status("Please select a leadfield for simulation", error=True)
            return False
        if not self.roi_list.selectedItems():
            self.update_status("Please select at least one ROI", error=True)
            return False

        buckets = self._current_bucket_inputs()
        if not all(buckets.values()):
            self.update_status(
                "Please load an m-ex-search bucket file with all eight buckets",
                error=True,
            )
            return False
        if not self.symmetric_bucket_checkbox.isChecked():
            n_upper = self._bucket_cartesian_upper_bound(buckets)
            if n_upper > MAX_UNSYMMETRIC_BUCKET_CANDIDATES:
                self.update_status(
                    "This bucket file is too large without symmetry. "
                    "Enable left/right symmetry or load smaller buckets.",
                    error=True,
                )
                return False
        if self.symmetric_bucket_checkbox.isChecked():
            try:
                mirror_map = self._build_current_symmetry_mirror_map()
                n_symmetric = count_multipolar_combinations(
                    buckets,
                    all_combinations=False,
                    symmetry_mirror_map=mirror_map,
                    symmetry_pairing=self.symmetry_pairing_combo.currentData(),
                )
            except (OSError, ValueError) as e:
                self.update_status(str(e), error=True)
                return False
            if n_symmetric == 0:
                self.update_status(
                    "No complete mirrored four-pair candidates were found.",
                    error=True,
                )
                return False
        return True

    @staticmethod
    def _bucket_cartesian_upper_bound(buckets):
        total = 1
        for values in buckets.values():
            total *= len(values)
        return total

    def run_optimization(self):
        if not self.validate_inputs():
            return

        subject_id = self.subject_combo.currentText()
        project_dir = self.pm.project_dir
        mex_search_dir = self.pm.m_ex_search(subject_id)
        buckets = self._current_bucket_inputs()
        self.use_all_combinations = False
        self.use_symmetric_bucket = self.symmetric_bucket_checkbox.isChecked()
        self.mex_buckets = buckets

        if self.use_symmetric_bucket:
            symmetry_mirror_map = self._build_current_symmetry_mirror_map()
            n_combos = count_multipolar_combinations(
                buckets,
                all_combinations=False,
                symmetry_mirror_map=symmetry_mirror_map,
                symmetry_pairing=self.symmetry_pairing_combo.currentData(),
            )
            n_combo_label = f"{n_combos:,}"
        else:
            symmetry_mirror_map = None
            n_combo_label = (
                f"up to {self._bucket_cartesian_upper_bound(buckets):,}"
            )

        selected_rois = self.roi_list.selectedItems()
        roi_names = [
            ExSearchEngine.display_roi_name(self._roi_name_from_item(item))
            for item in selected_rois
        ]
        metric_label = self.metric_combo.currentText()
        details = (
            f"Subject: {subject_id}\n"
            f"Mode: Bucketed Mode"
            f"{' (left/right symmetric)' if self.use_symmetric_bucket else ''}\n"
            f"Symmetry Pairing: {self.symmetry_pairing_combo.currentText()}\n"
            f"mTI Metric: {metric_label}\n"
            f"Current per Pair: {self.total_current_spinbox.value():.1f} mA\n"
            f"Search Space: {n_combo_label} four-pair candidates\n"
            f"ROIs: {', '.join(roi_names)}\n"
            f"Total ROIs: {len(roi_names)}"
        )
        if not ConfirmationDialog.confirm(
            self,
            title="Confirm m-Ex-Search Optimization",
            message="Are you sure you want to start the m-ex-search optimization?",
            details=details,
        ):
            return

        os.makedirs(mex_search_dir, exist_ok=True)
        selected_roi_names = [self._roi_name_from_item(item) for item in selected_rois]

        env = os.environ.copy()
        env["SUBJECTS_DIR"] = project_dir

        self.disable_controls()
        self.update_status(f"Running m-ex-search for subject {subject_id}...")
        self.roi_processing_queue = selected_roi_names.copy()
        self.current_roi_index = 0
        self._exsearch_had_errors = False
        self.log_exsearch_start(subject_id, len(selected_roi_names))
        self.run_roi_pipeline(subject_id, project_dir, mex_search_dir, env)

    def _build_ex_config(self, subject_id, roi_name, leadfield_hdf, eeg_net):
        electrodes = MExConfig.BucketElectrodes(**self.mex_buckets)
        return MExConfig(
            subject_id=subject_id,
            leadfield_hdf=leadfield_hdf,
            roi_name=roi_name,
            electrodes=electrodes,
            current_mA=self.total_current_spinbox.value(),
            mti_metric=self.metric_combo.currentData(),
            roi_radius=self.roi_radius_spinbox.value(),
            run_name=safe_mex_run_name(
                roi_name,
                eeg_net,
                self.metric_combo.currentData(),
            ),
            symmetric_bucket=bool(self.use_symmetric_bucket),
            symmetry_eeg_csv=(
                self._current_symmetry_eeg_csv()
                if self.use_symmetric_bucket
                else None
            ),
            symmetry_pairing=self.symmetry_pairing_combo.currentData(),
        )

    @staticmethod
    def _write_ex_config(config):
        return write_config_json(config, prefix="m_ex_config")

    def run_roi_pipeline(self, subject_id, project_dir, ex_search_dir, env):
        if self.current_roi_index >= len(self.roi_processing_queue):
            self.pipeline_completed()
            return

        current_roi = self.roi_processing_queue[self.current_roi_index]
        roi_name = ExSearchEngine.display_roi_name(current_roi)
        self.log_roi_start(
            self.current_roi_index, len(self.roi_processing_queue), roi_name
        )

        selected_items = self.leadfield_list.selectedItems()
        if not selected_items or not selected_items[0].data(QtCore.Qt.UserRole):
            self.update_status("No leadfield selected", error=True)
            self.enable_controls()
            return

        leadfield_data = selected_items[0].data(QtCore.Qt.UserRole)
        selected_net_name = leadfield_data["net_name"]
        selected_hdf5_path = leadfield_data["hdf5_path"]

        roi_dir = self.pm.rois(subject_id)
        roi_file = os.path.join(roi_dir, current_roi)
        coords = ExSearchEngine.get_roi_coordinates(subject_id, current_roi)
        if not os.path.isfile(roi_file):
            self.update_output(f"Error: ROI file not found: {current_roi}", "error")
            self.current_roi_index += 1
            self.run_roi_pipeline(subject_id, project_dir, ex_search_dir, env)
            return

        ex_config = self._build_ex_config(
            subject_id,
            current_roi,
            selected_hdf5_path,
            selected_net_name,
        )

        env = os.environ.copy()
        env["PROJECT_DIR"] = project_dir
        env["SUBJECT_NAME"] = subject_id
        env["SELECTED_EEG_NET"] = selected_net_name
        env["ROI_NAME"] = roi_name
        env["ROI_DIR"] = roi_dir

        if self.current_roi_index == 0:
            log_file = self.create_log_file_env("m_ex_search", subject_id)
            if log_file:
                env["TI_LOG_FILE"] = log_file
                self._shared_log_file = log_file
            self.log_pipeline_configuration(
                subject_id, project_dir, selected_net_name, selected_hdf5_path, env
            )
        elif self._shared_log_file:
            env["TI_LOG_FILE"] = self._shared_log_file

        if coords:
            self.log_roi_configuration(
                current_roi, roi_name, coords[0], coords[1], coords[2], env
            )

        output_dir_name = safe_mex_run_name(
            current_roi, selected_net_name, self.metric_combo.currentData()
        )
        roi_output_dir = os.path.join(ex_search_dir, output_dir_name)
        if os.path.exists(roi_output_dir) and os.listdir(roi_output_dir):
            if not confirm_overwrite(
                self, roi_output_dir, f"ROI m-ex-search directory '{output_dir_name}'"
            ):
                self.current_roi_index += 1
                self.run_roi_pipeline(subject_id, project_dir, ex_search_dir, env)
                return
            import shutil

            try:
                shutil.rmtree(roi_output_dir)
                self.update_output(f"Removed existing directory: {output_dir_name}")
            except OSError as e:
                self.update_output(
                    f"Error removing existing directory: {str(e)}", "error"
                )

        self.log_step_start("mTI search")
        config_path = self._write_ex_config(ex_config)
        cmd = ["simnibs_python", "-m", "tit.opt.mex", config_path]

        if self.debug_mode:
            self.update_output(f"[DEBUG] Config file: {config_path}", "debug")
            self.update_output(f"[DEBUG] Command: {' '.join(cmd)}", "debug")

        self.optimization_process = ExSearchThread(cmd, env)
        self.optimization_process.output_signal.connect(self.update_output)
        self.optimization_process.error_signal.connect(
            lambda msg: self.handle_process_error(msg)
        )
        self.optimization_process.process_finished.connect(
            lambda ok, rc: self.ti_simulation_completed(
                subject_id, project_dir, ex_search_dir, env, ok, rc
            )
        )
        self.optimization_process.start()

    def pipeline_completed(self):
        self.log_pipeline_completion()
        if self._exsearch_had_errors:
            self.enable_controls()
            self.update_status("m-ex-search stopped after an error", error=True)
            return
        subject_id = self.subject_combo.currentText()
        total_rois = len(self.roi_processing_queue)
        mex_search_dir = self.pm.m_ex_search(subject_id)
        self.log_exsearch_complete(subject_id, total_rois, mex_search_dir)
        self.enable_controls()
        self.update_status("m-ex-search optimization completed successfully")
        self.ex_search_completed.emit()

    def ti_simulation_completed(
        self, subject_id, project_dir, ex_search_dir, env, success=True, returncode=0
    ):
        if not success:
            self._exsearch_had_errors = True
            self.log_step_complete("mTI search", success=False)
            current_roi = self.roi_processing_queue[self.current_roi_index]
            self.update_output(
                f"m-ex-search failed for ROI {current_roi} "
                f"(exit code {returncode}).",
                "error",
            )
            self.pipeline_completed()
            return

        self.log_step_complete("mTI search", success=True)
        self.current_roi_completed(subject_id, project_dir, ex_search_dir, env)

    def disable_controls(self):
        super().disable_controls()
        self.quadrant_buckets_btn.setEnabled(False)
        self.metric_combo.setEnabled(False)
        self.symmetry_pairing_combo.setEnabled(False)

    def enable_controls(self):
        super().enable_controls()
        self.quadrant_buckets_btn.setEnabled(False)
        self.quadrant_buckets_btn.hide()
        self.save_buckets_btn.setEnabled(False)
        self.save_buckets_btn.hide()
        self.load_buckets_btn.setEnabled(True)
        self.metric_combo.setEnabled(True)
        self.symmetry_pairing_combo.setEnabled(True)
        self.rb_all_combinations.hide()
