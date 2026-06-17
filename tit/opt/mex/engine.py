"""Multipolar exhaustive-search engine."""

import logging
import os
import signal
import time

import numpy as np
from simnibs.utils import TI_utils as TI

from tit.calc import compute_mti_metric_field
from tit.opt.ex.engine import ExSearchEngine

from .logic import count_multipolar_combinations, generate_multipolar_combinations


class MExSearchEngine(ExSearchEngine):
    """Exhaustive search engine for four-pair multipolar mTI montages."""

    def __init__(
        self,
        leadfield_hdf: str,
        roi_file: str,
        roi_name: str,
        logger: logging.Logger,
        mti_metric: str,
    ):
        super().__init__(leadfield_hdf, roi_file, roi_name, logger)
        self.mti_metric = str(mti_metric)

    def compute_mti_field(
        self,
        electrodes: tuple[str, str, str, str, str, str, str, str],
        current_mA: float,
    ) -> dict[str, float]:
        """Compute one four-pair mTI candidate and return ROI metrics."""
        fields = []
        for idx in range(0, 8, 2):
            fields.append(
                TI.get_field(
                    [
                        electrodes[idx],
                        electrodes[idx + 1],
                        current_mA / 1000.0,
                    ],
                    self.leadfield,
                    self.idx_lf,
                )
            )

        metric_full = compute_mti_metric_field(fields, self.mti_metric)
        field_roi = metric_full[self.roi_indices]
        field_gm = metric_full[self.gm_indices]

        n_elements = int(len(field_roi))
        if n_elements == 0:
            roi_max = roi_mean = gm_mean = focality = 0.0
        else:
            roi_max = float(np.max(field_roi))
            roi_mean = float(np.average(field_roi, weights=self.roi_volumes))
            if len(field_gm) > 0:
                gm_mean = float(np.average(field_gm, weights=self.gm_volumes))
                focality = roi_mean / gm_mean if gm_mean > 0 else 0.0
            else:
                gm_mean = focality = 0.0

        return {
            f"{self.roi_name}_TImax_ROI": roi_max,
            f"{self.roi_name}_TImean_ROI": roi_mean,
            f"{self.roi_name}_TImean_GM": gm_mean,
            f"{self.roi_name}_Focality": focality,
            f"{self.roi_name}_n_elements": n_elements,
            "current_ch1_mA": current_mA,
            "current_ch2_mA": current_mA,
            "current_ch3_mA": current_mA,
            "current_ch4_mA": current_mA,
            "mti_metric": self.mti_metric,
        }

    def run(
        self,
        buckets_or_pool,
        all_combinations: bool,
        output_dir: str,
        current_mA: float,
        symmetry_mirror_map: dict[str, str] | None = None,
        symmetry_pairing: str = "within_pairs",
    ) -> dict[str, dict[str, float]]:
        """Run the full multipolar search loop."""
        stop = False

        def _on_signal(sig, frame):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)

        total = count_multipolar_combinations(
            buckets_or_pool,
            all_combinations=all_combinations,
            symmetry_mirror_map=symmetry_mirror_map,
            symmetry_pairing=symmetry_pairing,
        )
        self.logger.info("%s", "\n" + "=" * 60)
        mode = "All Combinations" if all_combinations else "Bucketed"
        if symmetry_mirror_map is not None:
            mode += f", left/right symmetric ({symmetry_pairing})"
        self.logger.info("Multipolar Exhaustive Search (%s)", mode)
        self.logger.info("mTI metric: %s", self.mti_metric)
        self.logger.info("Current per pair: %.3f mA", current_mA)
        self.logger.info("Total combinations: %d", total)
        self.logger.info("%s", "=" * 60 + "\n")

        results: dict[str, dict[str, float]] = {}
        start_time = time.time()

        for i, electrodes in enumerate(
            generate_multipolar_combinations(
                buckets_or_pool,
                all_combinations=all_combinations,
                symmetry_mirror_map=symmetry_mirror_map,
                symmetry_pairing=symmetry_pairing,
            ),
            1,
        ):
            if stop:
                self.logger.warning("Interrupted")
                break

            pair_names = [
                f"{electrodes[idx]}_{electrodes[idx + 1]}"
                for idx in range(0, 8, 2)
            ]
            name = "_and_".join(pair_names) + f"_I-{current_mA:.1f}mA"
            key = f"TI_field_{name}.msh"

            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0

            self.logger.info("[%d/%d] %s", i, total, name)
            if total:
                self.logger.info(
                    "  %.1f%% | %.2f/s | ETA %.1fmin",
                    100 * i / total,
                    rate,
                    eta / 60,
                )

            sim_start = time.time()
            data = self.compute_mti_field(electrodes, current_mA)
            results[key] = data
            self.logger.info(
                "  %.2fs | Max=%.4f Mean=%.4f Foc=%.4f",
                time.time() - sim_start,
                data[f"{self.roi_name}_TImax_ROI"],
                data[f"{self.roi_name}_TImean_ROI"],
                data[f"{self.roi_name}_Focality"],
            )

        if results:
            elapsed = time.time() - start_time
            self.logger.info(
                "Done: %d/%d in %.1fmin (%.2fs each)",
                len(results),
                total,
                elapsed / 60,
                elapsed / len(results),
            )
            self.logger.info("Output: %s", output_dir)

        return results


def safe_mex_run_name(roi_name: str, eeg_net: str, metric: str) -> str:
    """Return a filesystem-safe m-ex-search run folder name."""
    base = ExSearchEngine.safe_run_name(roi_name, eeg_net)
    metric = os.path.basename(str(metric)).replace(".", "_")
    return f"{base}_{metric}"
