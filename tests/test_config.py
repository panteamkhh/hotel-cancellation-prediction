from src import config


def test_month_order_has_twelve_calendar_months():
    assert len(config.MONTH_ORDER) == 12
    assert config.MONTH_ORDER[0] == "January"
    assert config.MONTH_ORDER[-1] == "December"


def test_target_and_random_state_are_set():
    assert config.TARGET == "is_canceled"
    assert isinstance(config.RANDOM_STATE, int)


def test_paths_are_absolute_and_under_project_root():
    for path in (
        config.RAW_DATA_PATH,
        config.CLEAN_DATA_PATH,
        config.FEATURES_DATA_PATH,
        config.BEST_MODEL_PATH,
        config.MODEL_METRICS_PATH,
    ):
        assert path.is_absolute()
        assert config.ROOT_DIR in path.parents


def test_generated_directories_exist():
    for directory in (
        config.PROCESSED_DATA_DIR,
        config.MODELS_DIR,
        config.FIGURES_DIR,
        config.OUTPUT_DIR,
    ):
        assert directory.is_dir()
