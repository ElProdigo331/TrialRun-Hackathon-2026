"""
Experiment History Management System
Provides persistent memory for all model training runs, tracking configurations,
metrics, and performance to enable data-driven model selection.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

HISTORY_FILE = "outputs/experiment_history.json"

def ensure_output_dir():
    """Ensure outputs directory exists."""
    os.makedirs("outputs", exist_ok=True)

def load_history() -> Dict:
    """Load experiment history from file."""
    ensure_output_dir()
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return create_empty_history()
    return create_empty_history()

def create_empty_history() -> Dict:
    """Create empty history structure."""
    return {
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "total_experiments": 0,
        "experiments": [],
        "best_ever": {
            "val_r2": None,
            "val_rmse": None,
            "cv_r2": None
        },
        "model_type_stats": {},
        "config_rankings": []
    }

def save_history(history: Dict):
    """Save experiment history to file."""
    ensure_output_dir()
    history["last_updated"] = datetime.now().isoformat()
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)

def record_experiment(
    model_type: str,
    config: Dict,
    metrics: Dict,
    feature_count: int,
    top_features: Optional[List] = None,
    experiment_name: Optional[str] = None
) -> Dict:
    """
    Record a new experiment to history.
    
    Args:
        model_type: Type of model (Ridge, RandomForest, etc.)
        config: Full configuration dict (normalize, sand_map, hyperparameters)
        metrics: Performance metrics (train_r2, val_r2, cv_r2_mean, rmse, etc.)
        feature_count: Number of features used
        top_features: Top important features (if available)
        experiment_name: Optional custom experiment name
    
    Returns:
        Dict with experiment record and comparison to previous best
    """
    history = load_history()
    
    experiment_id = history["total_experiments"] + 1
    timestamp = datetime.now().isoformat()
    
    if not experiment_name:
        experiment_name = f"{model_type}_{experiment_id}"
    
    experiment = {
        "id": experiment_id,
        "name": experiment_name,
        "timestamp": timestamp,
        "model_type": model_type,
        "config": config,
        "metrics": metrics,
        "feature_count": feature_count,
        "top_features": top_features or []
    }
    
    comparison = compare_to_best(history, metrics)
    experiment["comparison"] = comparison
    
    history["experiments"].append(experiment)
    history["total_experiments"] = experiment_id
    
    update_best_ever(history, experiment)
    update_model_type_stats(history)
    update_config_rankings(history)
    
    save_history(history)
    
    return {
        "experiment": experiment,
        "comparison": comparison,
        "rank": get_experiment_rank(history, experiment_id)
    }

def compare_to_best(history: Dict, new_metrics: Dict) -> Dict:
    """Compare new metrics to historical best."""
    comparison = {
        "is_new_best_val_r2": False,
        "is_new_best_rmse": False,
        "val_r2_improvement": None,
        "rmse_improvement": None,
        "vs_previous": None
    }
    
    best = history.get("best_ever", {})
    
    new_val_r2 = new_metrics.get("val_r2")
    if new_val_r2 is not None:
        if best.get("val_r2") is None or new_val_r2 > best["val_r2"]["value"]:
            comparison["is_new_best_val_r2"] = True
        if best.get("val_r2"):
            comparison["val_r2_improvement"] = new_val_r2 - best["val_r2"]["value"]
    
    new_rmse = new_metrics.get("val_rmse")
    if new_rmse is not None:
        if best.get("val_rmse") is None or new_rmse < best["val_rmse"]["value"]:
            comparison["is_new_best_rmse"] = True
        if best.get("val_rmse"):
            comparison["rmse_improvement"] = best["val_rmse"]["value"] - new_rmse
    
    experiments = history.get("experiments", [])
    if experiments:
        prev = experiments[-1]
        prev_val_r2 = prev["metrics"].get("val_r2")
        if prev_val_r2 is not None and new_val_r2 is not None:
            diff = new_val_r2 - prev_val_r2
            comparison["vs_previous"] = {
                "val_r2_diff": diff,
                "better": diff > 0,
                "previous_id": prev["id"]
            }
    
    return comparison

def update_best_ever(history: Dict, experiment: Dict):
    """Update best-ever records if new experiment is better."""
    metrics = experiment["metrics"]
    
    val_r2 = metrics.get("val_r2")
    if val_r2 is not None:
        current_best = history["best_ever"].get("val_r2")
        if current_best is None or val_r2 > current_best["value"]:
            history["best_ever"]["val_r2"] = {
                "value": val_r2,
                "experiment_id": experiment["id"],
                "config": experiment["config"],
                "model_type": experiment["model_type"],
                "timestamp": experiment["timestamp"]
            }
    
    val_rmse = metrics.get("val_rmse")
    if val_rmse is not None:
        current_best = history["best_ever"].get("val_rmse")
        if current_best is None or val_rmse < current_best["value"]:
            history["best_ever"]["val_rmse"] = {
                "value": val_rmse,
                "experiment_id": experiment["id"],
                "config": experiment["config"],
                "model_type": experiment["model_type"],
                "timestamp": experiment["timestamp"]
            }
    
    cv_r2 = metrics.get("cv_r2_mean")
    if cv_r2 is not None:
        current_best = history["best_ever"].get("cv_r2")
        if current_best is None or cv_r2 > current_best["value"]:
            history["best_ever"]["cv_r2"] = {
                "value": cv_r2,
                "experiment_id": experiment["id"],
                "config": experiment["config"],
                "model_type": experiment["model_type"],
                "timestamp": experiment["timestamp"]
            }

def update_model_type_stats(history: Dict):
    """Update aggregate statistics by model type."""
    experiments = history.get("experiments", [])
    if not experiments:
        return
    
    model_stats = {}
    for exp in experiments:
        model_type = exp["model_type"]
        if model_type not in model_stats:
            model_stats[model_type] = {
                "count": 0,
                "val_r2_values": [],
                "val_rmse_values": [],
                "best_val_r2": None,
                "best_config": None
            }
        
        stats = model_stats[model_type]
        stats["count"] += 1
        
        val_r2 = exp["metrics"].get("val_r2")
        if val_r2 is not None:
            stats["val_r2_values"].append(val_r2)
            if stats["best_val_r2"] is None or val_r2 > stats["best_val_r2"]:
                stats["best_val_r2"] = val_r2
                stats["best_config"] = exp["config"]
        
        val_rmse = exp["metrics"].get("val_rmse")
        if val_rmse is not None:
            stats["val_rmse_values"].append(val_rmse)
    
    for model_type, stats in model_stats.items():
        if stats["val_r2_values"]:
            stats["avg_val_r2"] = sum(stats["val_r2_values"]) / len(stats["val_r2_values"])
            stats["max_val_r2"] = max(stats["val_r2_values"])
            stats["min_val_r2"] = min(stats["val_r2_values"])
        if stats["val_rmse_values"]:
            stats["avg_val_rmse"] = sum(stats["val_rmse_values"]) / len(stats["val_rmse_values"])
            stats["min_val_rmse"] = min(stats["val_rmse_values"])
        del stats["val_r2_values"]
        del stats["val_rmse_values"]
    
    history["model_type_stats"] = model_stats

def update_config_rankings(history: Dict):
    """Update rankings of all configurations by val_r2."""
    experiments = history.get("experiments", [])
    if not experiments:
        return
    
    valid_experiments = [e for e in experiments if e["metrics"].get("val_r2") is not None]
    
    sorted_experiments = sorted(
        valid_experiments,
        key=lambda x: x["metrics"]["val_r2"],
        reverse=True
    )
    
    rankings = []
    for rank, exp in enumerate(sorted_experiments[:20], 1):
        rankings.append({
            "rank": rank,
            "experiment_id": exp["id"],
            "name": exp["name"],
            "model_type": exp["model_type"],
            "val_r2": exp["metrics"]["val_r2"],
            "val_rmse": exp["metrics"].get("val_rmse"),
            "config_summary": summarize_config(exp["config"]),
            "timestamp": exp["timestamp"]
        })
    
    history["config_rankings"] = rankings

def summarize_config(config: Dict) -> str:
    """Create a short summary of configuration."""
    parts = []
    if config.get("normalize") is not None:
        parts.append(f"norm={'Y' if config['normalize'] else 'N'}")
    if config.get("sand_map"):
        parts.append(f"sand={config['sand_map']}")
    if config.get("params"):
        params = config["params"]
        if "alpha" in params:
            parts.append(f"α={params['alpha']}")
        if "n_estimators" in params:
            parts.append(f"n={params['n_estimators']}")
        if "max_depth" in params:
            parts.append(f"d={params['max_depth']}")
        if "learning_rate" in params:
            parts.append(f"lr={params['learning_rate']}")
    return ", ".join(parts)

def get_experiment_rank(history: Dict, experiment_id: int) -> int:
    """Get the rank of an experiment in the history."""
    rankings = history.get("config_rankings", [])
    for r in rankings:
        if r["experiment_id"] == experiment_id:
            return r["rank"]
    return len(history.get("experiments", [])) + 1

def get_recommendations() -> Dict:
    """Get data-driven recommendations based on experiment history."""
    history = load_history()
    
    if not history.get("experiments"):
        return {
            "has_data": False,
            "message": "No experiments recorded yet. Run some model training to build history."
        }
    
    recommendations = {
        "has_data": True,
        "total_experiments": history["total_experiments"],
        "best_model_type": None,
        "best_config": None,
        "suggestions": []
    }
    
    model_stats = history.get("model_type_stats", {})
    if model_stats:
        best_model = max(model_stats.items(), key=lambda x: x[1].get("max_val_r2", -999))
        recommendations["best_model_type"] = {
            "name": best_model[0],
            "max_val_r2": best_model[1].get("max_val_r2"),
            "avg_val_r2": best_model[1].get("avg_val_r2"),
            "experiments_run": best_model[1].get("count")
        }
    
    if history["best_ever"].get("val_r2"):
        best = history["best_ever"]["val_r2"]
        recommendations["best_config"] = {
            "model_type": best["model_type"],
            "val_r2": best["value"],
            "config": best["config"]
        }
    
    if model_stats:
        for model_type, stats in model_stats.items():
            if stats.get("count", 0) < 3:
                recommendations["suggestions"].append(
                    f"Try more {model_type} configurations (only {stats['count']} tested)"
                )
    
    rankings = history.get("config_rankings", [])
    if rankings:
        top_models = set(r["model_type"] for r in rankings[:5])
        if len(top_models) == 1:
            recommendations["suggestions"].append(
                f"Top 5 results all use {list(top_models)[0]} - consider exploring this model type more"
            )
    
    return recommendations

def get_history_dataframe() -> pd.DataFrame:
    """Get experiment history as a pandas DataFrame for easy analysis."""
    history = load_history()
    experiments = history.get("experiments", [])
    
    if not experiments:
        return pd.DataFrame()
    
    records = []
    for exp in experiments:
        record = {
            "ID": exp["id"],
            "Name": exp["name"],
            "Timestamp": exp["timestamp"],
            "Model": exp["model_type"],
            "Val R²": exp["metrics"].get("val_r2"),
            "Train R²": exp["metrics"].get("train_r2"),
            "CV R²": exp["metrics"].get("cv_r2_mean"),
            "Val RMSE": exp["metrics"].get("val_rmse"),
            "Features": exp["feature_count"],
            "Config": summarize_config(exp["config"])
        }
        records.append(record)
    
    return pd.DataFrame(records)

def clear_history():
    """Clear all experiment history (use with caution)."""
    history = create_empty_history()
    save_history(history)
    return history
