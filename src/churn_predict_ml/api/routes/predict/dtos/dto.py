from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

YesNo = Literal["Yes", "No"]

InternetDependentService = Literal[
    "Yes",
    "No",
    "No internet service",
]


class PredictRequest(BaseModel):
    """Representa os dados de um cliente para previsão de churn.

    Contém as 19 features utilizadas pelo pipeline de
    Machine Learning, com validações de tipos, categorias
    e consistência entre serviços contratados.
    """

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetDependentService
    OnlineBackup: InternetDependentService
    DeviceProtection: InternetDependentService
    TechSupport: InternetDependentService
    StreamingTV: InternetDependentService
    StreamingMovies: InternetDependentService
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(ge=0, allow_inf_nan=False)
    TotalCharges: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_service_consistency(self) -> PredictRequest:
        """Valida a consistência entre serviços contratados.

        Returns:
            Instância validada da requisição.

        Raises:
            ValueError: Se os serviços forem incompatíveis
                com a disponibilidade de telefone ou internet.
        """
        if self.PhoneService == "No" and self.MultipleLines != "No phone service":
            raise ValueError(
                "MultipleLines must be 'No phone service' when PhoneService is 'No'."
            )

        if self.PhoneService == "Yes" and self.MultipleLines == "No phone service":
            raise ValueError(
                "MultipleLines cannot be 'No phone service' when PhoneService is 'Yes'."
            )

        internet_services = (
            self.OnlineSecurity,
            self.OnlineBackup,
            self.DeviceProtection,
            self.TechSupport,
            self.StreamingTV,
            self.StreamingMovies,
        )

        if self.InternetService == "No":
            if any(value != "No internet service" for value in internet_services):
                raise ValueError(
                    "Internet-dependent services must be "
                    "'No internet service' when InternetService is 'No'."
                )
        elif any(value == "No internet service" for value in internet_services):
            raise ValueError(
                "Internet-dependent services cannot be "
                "'No internet service' when internet is available."
            )

        return self


class PredictResponse(BaseModel):
    """Representa o resultado da previsão de churn.

    Attributes:
        churn_probability: Probabilidade prevista de cancelamento.
        churn_prediction: Classificação binária do cliente.
    """

    model_config = ConfigDict(extra="forbid")

    churn_probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    churn_prediction: Literal[0, 1]
