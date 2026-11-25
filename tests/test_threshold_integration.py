import numpy as np
import pandas as pd
import pytest

from churn_predict_ml.experiment import evaluate_model, run_experiment
from churn_predict_ml.threshold import (
    compare_thresholds,
    predict_with_threshold,
    select_best_threshold,
)


def test_selected_threshold_uses_validation_metrics() -> None:
    """Verifica a escolha do threshold usando métricas de validação."""
    y_validation = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.10, 0.40, 0.35, 0.80])

    results = compare_thresholds(
        y_validation,
        probabilities,
        thresholds=[0.30, 0.50],
    )

    selected_threshold = select_best_threshold(results)

    assert selected_threshold == pytest.approx(0.30)


def test_optimized_threshold_changes_classification() -> None:
    """Verifica se thresholds diferentes alteram as previsões."""
    probabilities = np.array([0.20, 0.35, 0.45, 0.80])

    default_predictions = predict_with_threshold(probabilities, 0.50)
    optimized_predictions = predict_with_threshold(probabilities, 0.30)

    np.testing.assert_array_equal(default_predictions, [0, 0, 0, 1])
    np.testing.assert_array_equal(optimized_predictions, [0, 1, 1, 1])


def test_roc_auc_is_independent_of_threshold() -> None:
    """Verifica que o ROC-AUC não muda ao alterar o threshold."""
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.10, 0.40, 0.35, 0.80])

    default_predictions = predict_with_threshold(probabilities, 0.50)
    optimized_predictions = predict_with_threshold(probabilities, 0.30)

    default_metrics = evaluate_model(
        y_true,
        default_predictions,
        probabilities,
    )

    optimized_metrics = evaluate_model(
        y_true,
        optimized_predictions,
        probabilities,
    )

    assert default_metrics["roc_auc"] == pytest.approx(optimized_metrics["roc_auc"])


def test_run_experiment_with_synthetic_dataset(
    tmp_path,
    capsys,
) -> None:
    """Verifica a execução completa usando um dataset sintético."""
    rows = 100

    df = pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(rows)],
            "tenure": np.arange(1, rows + 1),
            "MonthlyCharges": np.tile([40.0, 80.0], rows // 2),
            "TotalCharges": np.arange(1, rows + 1) * 50.0,
            "Contract": np.tile(
                ["Month-to-month", "One year"],
                rows // 2,
            ),
            "Churn": np.tile(["No", "Yes"], rows // 2),
        }
    )

    dataset_path = tmp_path / "synthetic_churn.csv"
    df.to_csv(dataset_path, index=False)

    run_experiment(dataset_path)

    output = capsys.readouterr().out

    assert "Total de registros: 100" in output
    assert "Registros de desenvolvimento: 80" in output
    assert "Registros de treinamento interno: 64" in output
    assert "Registros de validação: 16" in output
    assert "Registros de teste: 20" in output
    assert "Selected Threshold:" in output
    assert "Model Evaluation:" in output
    assert "Confusion Matrix" in output
