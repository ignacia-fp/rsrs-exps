#[derive(Clone, strum_macros::AsRefStr, strum_macros::EnumString, Debug, serde::Deserialize)]
    pub enum StructuredOperatorType {
    BaseStructuredOperator,
    BasicStructuredOperator,
    BemppClLaplaceSingleLayer,
    BemppClHelmholtzSingleLayer,
    KiFMMLaplaceOperator,
    KiFMMHelmholtzOperator,
}
