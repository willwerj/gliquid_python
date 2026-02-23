"""
Lightweight production runner for pre-trained L0/L1 models.

Features:
- Loads pre-trained models/artifacts from an exported bundle
- Predicts from single-row DataFrames
- Preserves L1 antisymmetric constraint when mirror row is provided
- Includes compact SHAP force-plot figure generation (core behavior)

No imports from local project modules are used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from scripts.shap_compat import apply_patches as apply_shap_patches
apply_shap_patches()


class ProductionModelRunner:
    """Standalone model runner for production-style single-sample inference."""

    def __init__(self, bundle_dir: str):
        self.bundle_dir = Path(bundle_dir)
        self.model_dir = self.bundle_dir / "model"
        self.data_dir = self.bundle_dir / "data"

        self.models: Dict[str, object] = {}
        self.feature_names: Dict[str, List[str]] = {}
        self.target_transformers: Dict[str, object] = {}
        self.explainers: Dict[str, object] = {}

        self.df_symm: Optional[pd.DataFrame] = None
        self.df_anti: Optional[pd.DataFrame] = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model artifacts + prediction datasets from bundle."""
        required = [
            self.model_dir / "L0_a_model.joblib",
            self.model_dir / "L0_b_model.joblib",
            self.model_dir / "L1_a_model.joblib",
            self.model_dir / "feature_names_symm.joblib",
            self.model_dir / "feature_names_anti.joblib",
            self.data_dir / "prediction_dataset_symmetric.xlsx",
            self.data_dir / "prediction_dataset_antisymmetric.xlsx",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing required bundle files:\n" + "\n".join(missing))

        self.models = {
            "L0_a": joblib.load(self.model_dir / "L0_a_model.joblib"),
            "L0_b": joblib.load(self.model_dir / "L0_b_model.joblib"),
            "L1_a": joblib.load(self.model_dir / "L1_a_model.joblib"),
        }

        self.feature_names = {
            "symmetric": list(joblib.load(self.model_dir / "feature_names_symm.joblib")),
            "antisymmetric": list(joblib.load(self.model_dir / "feature_names_anti.joblib")),
        }

        tf_path = self.model_dir / "target_transformers.joblib"
        self.target_transformers = joblib.load(tf_path) if tf_path.exists() else {}

        self.df_symm = pd.read_excel(self.data_dir / "prediction_dataset_symmetric.xlsx")
        self.df_anti = pd.read_excel(self.data_dir / "prediction_dataset_antisymmetric.xlsx")

    @staticmethod
    def _mirror_system(system_name: str) -> Optional[str]:
        for sep in ("-", "_"):
            if sep in system_name:
                parts = system_name.split(sep)
                if len(parts) == 2:
                    return f"{parts[1]}{sep}{parts[0]}"
        return None

    @staticmethod
    def _extract_scalar(x: object) -> float:
        arr = np.asarray(x)
        if arr.ndim == 0:
            return float(arr)
        return float(arr.ravel()[0])

    def _prepare_row(self, row_df: pd.DataFrame, mode: str) -> pd.DataFrame:
        """Align row dataframe to required feature order."""
        if len(row_df) != 1:
            raise ValueError("Input must be a single-row DataFrame.")

        if mode not in ("symmetric", "antisymmetric"):
            raise ValueError("mode must be 'symmetric' or 'antisymmetric'.")

        req_features = self.feature_names[mode]
        missing = [c for c in req_features if c not in row_df.columns]
        if missing:
            raise KeyError(
                f"Missing {mode} features: {missing[:10]}"
                + (" ..." if len(missing) > 10 else "")
            )

        return row_df[req_features].copy()

    def _predict_single_target(self, target: str, X_row: pd.DataFrame) -> float:
        pred = self.models[target].predict(X_row.values)
        pred = np.asarray(pred).reshape(-1)

        if target in self.target_transformers:
            pred = self.target_transformers[target].inverse_transform(pred.reshape(-1, 1)).ravel()

        return float(pred[0])

    def get_rows_for_system(
        self, system_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Fetch one symmetric row, one antisymmetric row, and optional antisymmetric mirror row."""
        symm_row = self.df_symm[self.df_symm["system"] == system_name].head(1).copy()
        anti_row = self.df_anti[self.df_anti["system"] == system_name].head(1).copy()

        if symm_row.empty:
            raise ValueError(f"System '{system_name}' not found in symmetric dataset.")
        if anti_row.empty:
            raise ValueError(f"System '{system_name}' not found in antisymmetric dataset.")

        mirror_name = self._mirror_system(system_name)
        anti_mirror_row = None
        if mirror_name is not None:
            mirror = self.df_anti[self.df_anti["system"] == mirror_name].head(1).copy()
            if not mirror.empty:
                anti_mirror_row = mirror

        return symm_row, anti_row, anti_mirror_row

    def predict_from_dataframes(
        self,
        symm_row_df: pd.DataFrame,
        anti_row_df: pd.DataFrame,
        anti_mirror_row_df: Optional[pd.DataFrame] = None,
    ) -> List[float]:
        """
        Predict [L0_a, L0_b, L1_a] from single-row DataFrames.

        If anti_mirror_row_df is provided, enforce antisymmetry via:
            L1_a = 0.5 * (pred(A-B) - pred(B-A))
        """
        X_symm = self._prepare_row(symm_row_df, mode="symmetric")
        X_anti = self._prepare_row(anti_row_df, mode="antisymmetric")

        l0_a = self._predict_single_target("L0_a", X_symm)
        l0_b = self._predict_single_target("L0_b", X_symm)
        l1_raw = self._predict_single_target("L1_a", X_anti)

        if anti_mirror_row_df is not None:
            X_anti_mirror = self._prepare_row(anti_mirror_row_df, mode="antisymmetric")
            l1_raw_mirror = self._predict_single_target("L1_a", X_anti_mirror)
            l1_a = 0.5 * (l1_raw - l1_raw_mirror)
        else:
            l1_a = l1_raw

        return [l0_a, l0_b, l1_a]

    def predict_system(self, system_name: str) -> List[float]:
        """Predict [L0_a, L0_b, L1_a] using bundle datasets for a named system."""
        symm_row, anti_row, anti_mirror_row = self.get_rows_for_system(system_name)
        return self.predict_from_dataframes(symm_row, anti_row, anti_mirror_row)

    @staticmethod
    def apply_feature_updates(row_df: pd.DataFrame, updates: Dict[str, float]) -> pd.DataFrame:
        """Return a modified copy of row_df with selected feature updates."""
        if len(row_df) != 1:
            raise ValueError("row_df must be single-row.")
        out = row_df.copy()
        for key, value in updates.items():
            if key not in out.columns:
                raise KeyError(f"Feature '{key}' not found in input row.")
            out.loc[out.index[0], key] = value
        return out

    def _get_tree_model(self, model: object) -> object:
        """Extract final tree model for SHAP TreeExplainer."""
        if hasattr(model, "steps") and len(model.steps) > 0:
            return model.steps[-1][1]
        return model

    def _transform_row_for_explainer(self, model: object, features_df: pd.DataFrame) -> np.ndarray:
        """Apply all preprocessing steps before final estimator."""
        X = features_df.values
        if hasattr(model, "steps") and len(model.steps) > 1:
            for _, transformer in model.steps[:-1]:
                if transformer is not None:
                    X = transformer.transform(X)
        return X

    def _ensure_explainer(self, target: str) -> None:
        if target in self.explainers:
            return
        tree_model = self._get_tree_model(self.models[target])
        self.explainers[target] = shap.TreeExplainer(tree_model)

    def get_shap_explanation(
        self, target: str, features_df: pd.DataFrame
    ) -> shap.Explanation:
        """Compute a SHAP Explanation for *target* in original parameter space.

        If a ``StandardScaler`` was used on the target during training the
        SHAP values and base value live in z-score space.  This helper
        inverse-transforms them so the explanation is directly interpretable
        in physical units:

        * ``values``      → multiplied by σ  (scaler.scale_)
        * ``base_values``  → μ + σ · E[z]   (scaler.mean_ + scaler.scale_ · base)

        Parameters
        ----------
        target : str
            Model key, e.g. ``"L0_a"``, ``"L0_b"``, ``"L1_a"``.
        features_df : pd.DataFrame
            Single-row DataFrame already aligned to the correct feature order
            (output of :meth:`_prepare_row`).

        Returns
        -------
        shap.Explanation
            One-dimensional Explanation with values in original target space.
        """
        self._ensure_explainer(target)
        model = self.models[target]
        explainer = self.explainers[target]

        # Determine mode from target name
        mode = "antisymmetric" if target == "L1_a" else "symmetric"
        feature_names = self.feature_names[mode]

        # Preprocess features through pipeline (minus final estimator)
        X = self._transform_row_for_explainer(model, features_df)

        # Raw SHAP values (z-score space if scaler exists)
        sv = explainer.shap_values(X)
        base = float(explainer.expected_value)

        values = np.array(sv[0], dtype=float)

        # Inverse-transform if a StandardScaler was applied to this target
        if target in self.target_transformers:
            scaler = self.target_transformers[target]
            sigma = float(scaler.scale_[0])
            mu = float(scaler.mean_[0])
            values = values * sigma
            base = mu + sigma * base

        return shap.Explanation(
            values=values,
            base_values=base,
            data=features_df.iloc[0].values,
            feature_names=feature_names,
        )

    def create_compact_prediction_figure(self, system_name, output_file,
                                         max_display_features=6):
        """
        Create a compact 300x480px figure with vertical stack of waterfall plots.
        
        Generates a publication-ready compact figure showing SHAP waterfall plots for
        all three parameters (L0_a, L0_b, L1_a) in a vertical stack. Each subplot is
        100px tall by 480px wide, totaling 300x480px. Features are labeled in order
        of importance up to the maximum that can be reasonably displayed.
        
        Parameters
        ----------
        system_name : str
            Name of the system to analyze
        output_dir : str
            Directory to save the figure
        max_display_features : int, optional
            Maximum number of features to label. If None, automatically determines
            based on space available. Features are labeled in order of absolute
            SHAP value (most important first).
            
        Returns
        -------
        str
            Path to saved figure file
        """
        print("\n" + "="*80)
        print(f"CREATING COMPACT PREDICTION FIGURE: {system_name}")
        print("="*80)

        # Fetch feature rows using the existing workflow
        symm_row, anti_row, _ = self.get_rows_for_system(system_name)
        l0_features_df = self._prepare_row(symm_row, mode="symmetric")
        l1_features_df = self._prepare_row(anti_row, mode="antisymmetric")

        # Ensure all explainers are ready
        for param in ["L0_a", "L0_b", "L1_a"]:
            self._ensure_explainer(param)
        
        # Create figure with 3 subplots vertically stacked
        # Figure size in inches - will be saved as SVG (scalable)
        # Using nominal size that gives good proportions
        fig, axes = plt.subplots(3, 1, figsize=(4.8, 3.0), gridspec_kw={'hspace': 0.23, 'left': 0.305, 'right': 0.955, 'top': 0.95, 'bottom': 0})
        
        params = ['L0_a', 'L0_b', 'L1_a']
        
        for idx, (param, ax) in enumerate(zip(params, axes)):
            print(f"Creating waterfall plot for {param}...")
            
            # Determine features based on parameter
            if param in ['L0_a', 'L0_b']:
                features_df = l0_features_df
            else:  # L1_a
                features_df = l1_features_df

            # Get SHAP explanation in original parameter space
            shap_explanation = self.get_shap_explanation(param, features_df)
            
            # Create waterfall plot on this axis
            plt.sca(ax)  # Set current axis
            shap.plots.waterfall(
                shap_explanation, 
                max_display=max_display_features,
                show=False
            )
            # Get y-axis tick labels and modify them
            y_labels = ax.get_yticklabels()[:max_display_features]
            ax.set_yticks(ax.get_yticks()[:max_display_features])
            new_labels = []
            # print("Original y_labels:", [label.get_text() for label in y_labels])
            
            for label in y_labels:
                label_text = label.get_text()
                if ' = ' in label_text:
                    parts = label_text.split(' = ')
                    feature_name = parts[1]
                    value_str = parts[0]
                    value = float(value_str.replace('−', '-'))
                    formatted_value = f"{value:.2g}"
                    feature_name = feature_name.replace('metastable', 'ms')
                    feature_name = feature_name.replace('valence', 'val')
                    feature_name = feature_name.replace('enthalpy', 'enth')
                    feature_name = feature_name.replace('change', 'Δ')
                    new_labels.append(f"{feature_name} = {formatted_value}")
                else:
                    new_labels.append(label_text)
            
            ax.set_yticklabels(new_labels)
            ax.tick_params(axis='y', labelsize=11, labelcolor='black')
            ax.set_xlabel('')  # Remove x-axis label

            # SHAP value formatting
            for text in ax.texts:
                text.set_fontsize(10)
                text_content = text.get_text()
                if param == 'L0_b':
                    value = float(text_content.replace('−', '-'))
                    text.set_text(f"{value:.2f}")
                elif param in ['L0_a', 'L1_a']:
                    value = float(text_content.replace('−', '-'))
                    text.set_text(f"{int(round(value))}")

            xlim = ax.get_xlim()
            x_min, x_max = xlim
            x_range = x_max - x_min
            ideal_step = x_range / 4  # 4 intervals for 5 ticks

            # Determine appropriate step with at most 2 significant figures
            if ideal_step == 0:
                step = 1
            else:
                # Find the order of magnitude
                magnitude = 10 ** int(np.floor(np.log10(abs(ideal_step))))
                
                # Normalize to 1-10 range
                normalized = ideal_step / magnitude
                
                # Pick step with 2 significant figures
                if normalized <= 1.0:
                    step_normalized = 1.0
                elif normalized <= 1.5:
                    step_normalized = 1.5
                elif normalized <= 2.0:
                    step_normalized = 2.0
                elif normalized <= 2.5:
                    step_normalized = 2.5
                elif normalized <= 3.0:
                    step_normalized = 3.0
                elif normalized <= 4.0:
                    step_normalized = 4.0
                elif normalized <= 5.0:
                    step_normalized = 5.0
                elif normalized <= 7.5:
                    step_normalized = 7.5
                else:
                    step_normalized = 10.0
                
                step = step_normalized * magnitude

            # Find starting tick (should be multiple of step and <= x_min)
            start_tick = int(np.floor(x_min / step)) * step

            # If range crosses zero, ensure 0 is included
            if x_min <= 0 <= x_max:
                # Adjust start_tick to ensure 0 is one of the 5 ticks
                ticks_before_zero = 2  # Use 2 ticks before zero for balance
                start_tick = -ticks_before_zero * step

            # Generate 5 evenly spaced ticks
            tick_values = [start_tick + i * step for i in range(5)]

            # Filter to only include ticks within the visible range
            tick_values = [t for t in tick_values if x_min <= t <= x_max]

            # Ensure we have at least 3 ticks for readability
            if len(tick_values) < 3:
                # Fall back to simple approach
                if x_min <= 0 <= x_max:
                    tick_values = [int(x_min), 0, int(x_max)]
                else:
                    tick_values = [int(x_min), int((x_min + x_max) / 2), int(x_max)]
            
            
            # print("tick_values:", tick_values)
            x_min = x_max - x_range * 1.08
            ax.set_xlim(x_min, x_max)
            ax.set_xticks(tick_values)
            ax.set_xticklabels([str(int(t)) for t in tick_values], fontsize=9)
            
            # Remove grid lines
            ax.grid(False)
            
            # Add parameter label in top left
            ax.text(0.01, 1, param, transform=ax.transAxes,
                   fontsize=11, fontweight='bold',
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
            # Add figure title in top left corner
            fig.suptitle('SHAP Analysis', fontsize=14, fontweight='bold', 
                         x=0.025, y=1, ha='left', va='top')
        
        if output_file is None:
            plt.show()
            return None
        else:
            plt.savefig(output_file, format='svg', bbox_inches='tight', dpi=400)
            plt.close()
        
            print(f"[OK] Compact prediction figure saved: {output_file}")
            print(f"{'='*80}\n")
            
            return output_file

    def create_beeswarm_figures(
        self,
        output_dir: Optional[str] = None,
        max_display: int = 6,
        sample_fraction: float = 1.0,
        publication: bool = False,
    ) -> List[Optional[str]]:
        """Create beeswarm SHAP plots for L0_a, L0_b, and L1_a.

        Parameters
        ----------
        output_dir : str or None
            Directory to save SVG files.  If ``None``, plots are shown
            interactively with ``plt.show()`` instead.
        max_display : int
            Maximum number of features to display (default 6).
        sample_fraction : float
            Fraction of the dataset to use (0–1).  Values < 1 speed up
            computation for quick visual checks.
        publication : bool
            If ``True``, strip **all** text (axis labels, tick labels, feature
            names, colorbar labels) so only points and lines remain, then save
            as a high-resolution SVG.

        Returns
        -------
        list of str or None
            Paths to saved files, or ``None`` entries when shown interactively.
        """
        print("\n" + "=" * 80)
        print("CREATING BEESWARM FIGURES")
        print("=" * 80)

        results: List[Optional[str]] = []

        for target in ["L0_a", "L0_b", "L1_a"]:
            print(f"Computing beeswarm for {target} "
                  f"(sample_fraction={sample_fraction:.0%})...")

            # --- select the right dataset & feature set -----------------------
            mode = "antisymmetric" if target == "L1_a" else "symmetric"
            df = self.df_anti if target == "L1_a" else self.df_symm
            req_features = self.feature_names[mode]

            # --- optional subsampling -----------------------------------------
            if sample_fraction < 1.0:
                df = df.sample(frac=sample_fraction, random_state=42)

            features_df = df[req_features].copy()

            # --- batch SHAP in original parameter space -----------------------
            self._ensure_explainer(target)
            model = self.models[target]
            explainer = self.explainers[target]

            X = self._transform_row_for_explainer(model, features_df)
            sv = explainer.shap_values(X)
            base = float(explainer.expected_value)
            values = np.array(sv, dtype=float)

            if target in self.target_transformers:
                scaler = self.target_transformers[target]
                sigma = float(scaler.scale_[0])
                mu = float(scaler.mean_[0])
                values = values * sigma
                base = mu + sigma * base

            explanation = shap.Explanation(
                values=values,
                base_values=np.full(len(features_df), base),
                data=features_df.values,
                feature_names=req_features,
            )

            # --- plot ---------------------------------------------------------
            fig, ax = plt.subplots(figsize=(8, 5))
            plt.sca(ax)
            shap.plots.beeswarm(explanation, max_display=max_display, show=False)

            if publication:
                # Strip ALL text; keep only points, lines, and the zero vline.
                ax = plt.gca()
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_title("")
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                # ax.tick_params(
                #     axis="both", which="both", length=0,
                #     labelbottom=False, labelleft=False,
                # )

                # Clear colorbar / any other axes
                for other_ax in fig.get_axes():
                    if other_ax is not ax:
                        other_ax.set_ylabel("")
                        other_ax.set_xlabel("")
                        other_ax.set_title("")
                        other_ax.set_xticklabels([])
                        other_ax.set_yticklabels([])
                        # other_ax.tick_params(
                        #     axis="both", which="both", length=0,
                        #     labelbottom=False, labelleft=False,
                        # )

                # Remove any standalone figure-level text
                for txt in fig.texts:
                    txt.set_visible(False)

            # --- save or show -------------------------------------------------
            if output_dir is not None:
                out_path = Path(output_dir) / f"{target}_beeswarm.svg"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(out_path), format="svg",
                            bbox_inches="tight", dpi=400)
                plt.close()
                print(f"  [OK] Saved: {out_path}")
                results.append(str(out_path))
            else:
                plt.show()
                results.append(None)

        print("=" * 80 + "\n")
        return results


if __name__ == "__main__":
    # Minimal smoke-test style usage for the selected model bundle.
    scripts_dir = Path(__file__).resolve().parent
    bundle = scripts_dir / "bundle_20260217_135723"

    runner = ProductionModelRunner(str(bundle))

    # Example: pick first available system
    example_system = str(runner.df_symm["system"].iloc[1])
    preds = runner.predict_system(example_system)
    print(f"System: {example_system}")
    print(f"Predictions [L0_a, L0_b, L1_a]: {preds}")

    fig_file = runner.create_compact_prediction_figure(
        system_name=example_system,
        output_file=str(bundle / "shap_output" / f"{example_system}_compact_prediction.svg"),
        max_display_features=6,
    )
    print(f"Saved SHAP compact prediction figure: {fig_file}")

    # Beeswarm test – 20% sample, default settings, saved to file
    runner.create_beeswarm_figures(
        output_dir=str(bundle / "shap_output/pub"),
        max_display=6,
        sample_fraction=1,
        publication=True,
    )
