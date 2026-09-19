import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src import theme  # noqa: E402


def test_apply_theme_sets_whitegrid_style():
    theme.apply_theme()
    assert matplotlib.rcParams["axes.facecolor"] != "none"


def test_save_figure_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(theme, "FIGURES_DIR", tmp_path)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    output = theme.save_figure(fig, "chart.png")
    plt.close(fig)

    assert output.exists()
    assert output.name == "chart.png"
    assert output.parent == tmp_path
