from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest


def test_dashboard_lookup_and_prediction():
    result = {
        "churn_probability": 0.7,
        "churn_prediction": 1,
        "risk_level": "Alto",
        "model_version": "3",
    }
    response = Mock(ok=True, status_code=200)
    response.json.return_value = result
    with patch("requests.request", return_value=response) as request:
        app = AppTest.from_file("../streamlit_app.py", default_timeout=15).run()
        assert not app.exception
        app.text_input[0].set_value("TEST-1")
        next(
            button for button in app.button if button.label == "Consultar"
        ).click().run()
        assert not app.exception
        assert app.metric[0].value == "70.0%"
        next(
            button for button in app.button if button.label == "Calcular previsão"
        ).click().run()
        assert not app.exception
        assert any(call.args[0] == "POST" for call in request.call_args_list)
