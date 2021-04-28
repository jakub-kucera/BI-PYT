import pytest
import matplotlib.pyplot as plt

from ocr.plotter import Plotter

TEST_DEFAULT_DATASET = "dataset/"
TEST_DEFAULT_DATASET_ADVANCED = "test_datasets/written_hiragana_dataset/"


@pytest.mark.parametrize("dataset_directory", [
    TEST_DEFAULT_DATASET,
    TEST_DEFAULT_DATASET_ADVANCED,
    "testik",
    "asfsdafasfd",
])
def test_plotter_create(dataset_directory, mocker):
    mocker.patch('matplotlib.pyplot.title')
    plotter = Plotter(dataset_directory)
    plt.title.assert_called_once_with(dataset_directory)
    assert plotter.records == []


@pytest.mark.parametrize("dataset_directory, records", [
    (TEST_DEFAULT_DATASET, []),
    (TEST_DEFAULT_DATASET_ADVANCED, [1]),
    ("testik", [1, 2, 3]),
    ("asfsdafasfd", [x for x in range(0, 20, 2)]),
])
def test_plotter_add(dataset_directory, records, mocker):
    plotter = Plotter(dataset_directory)
    assert plotter.records == []

    inserted_records = []
    for record in records:
        inserted_records += [record]
        plotter.add_record(record)
        assert inserted_records == plotter.records


@pytest.mark.parametrize("dataset_directory, records", [
    (TEST_DEFAULT_DATASET, []),
    (TEST_DEFAULT_DATASET_ADVANCED, [1]),
    ("testik", [1, 2, 3]),
    ("asfsdafasfd", [x for x in range(0, 20, 2)]),
])
def test_plotter_show(dataset_directory, records, mocker):
    mocker.patch('matplotlib.pyplot.show')
    plotter = Plotter(dataset_directory)
    map(plotter.add_record, records)
    plotter.show()
    plt.show.assert_called()
