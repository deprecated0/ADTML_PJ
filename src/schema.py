import great_expectations as gx
from great_expectations.core import ExpectationSuite

SUITE_NAME = "credit_card_suite"
TARGET = "default.payment.next.month"


def build_suite() -> ExpectationSuite:
    suite = gx.ExpectationSuite(name=SUITE_NAME)

    # Обязательные колонки
    required_columns = [
        "LIMIT_BAL", "AGE", "SEX", "EDUCATION", "MARRIAGE", TARGET
    ]
    for col in required_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(column=col)
        )

    # Отсутствие null в ключевых полях
    not_null_columns = ["LIMIT_BAL", "AGE", TARGET]
    for col in not_null_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
        )

    # Возраст больше 18, т.к. до 18 кредиты не могут выдавать
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="AGE",
            min_value=18,
            max_value=95
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column=TARGET,
            value_set=[0, 1]
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="LIMIT_BAL",
            min_value=0,
            max_value=2_000_000
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="SEX",
            value_set=[1, 2]
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="EDUCATION",
            min_value=0,
            max_value=6
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="MARRIAGE",
            min_value=0,
            max_value=3
        )
    )

    return suite


def run_validation(df):
    context = gx.get_context()
    suite = build_suite()

    datasource = context.data_sources.add_pandas(name="credit_datasource")
    data_asset = datasource.add_dataframe_asset(name="credit_df")

    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        name="credit_batch"
    )
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    validation_result = batch.validate(suite)

    if not validation_result.success:
        failed_expectations = []
        for result in validation_result.results:
            if not result.success:
                exp_type = result.expectation_config.type
                column = result.expectation_config.kwargs.get('column', 'unknown')
                failed_expectations.append(f"{exp_type} on column '{column}'")

        raise ValueError(
            f"Great Expectations validation failed:\n"
            f"Failed expectations: {', '.join(failed_expectations)}\n"
            f"Success rate: {validation_result.statistics['success_percent']:.1f}%"
        )

    print(f"Great Expectations validation passed "
          f"(success rate: {validation_result.statistics['success_percent']:.1f}%)")
    return validation_result