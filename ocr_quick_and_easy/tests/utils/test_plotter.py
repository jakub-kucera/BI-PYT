from typing import Mapping, Any, Optional, List

import pytest
import matplotlib.pyplot as plt
from PIL import Image

from ocr.utils.plotter import Plotter

TEST_DEFAULT_DATASET = "dataset/"
TEST_DEFAULT_DATASET_ADVANCED = "test_datasets/written_hiragana_dataset/"


@pytest.mark.parametrize("dataset_directory, records, cutoff, section_width", [
    (
        TEST_DEFAULT_DATASET,
        [1, 2],
        127,
        None,
    ),
    (
        TEST_DEFAULT_DATASET,
        [],
        127,
        100,
    ),
    (
        TEST_DEFAULT_DATASET_ADVANCED,
        list(range(0, 20, 2)),
        1,
        2,
    ),
    (
        "testing name",
        list(range(50, 100, -1)),
        200,
        None,
    ),
])
def test_plotter_create(dataset_directory: str,
                        records: List[int],
                        cutoff: int,
                        section_width: Optional[int], mocker):
    # mocker.patch('matplotlib.pyplot.grid')
    mocker.patch('ocr.utils.plotter.Plotter._Plotter__set_section_width')

    plotter = Plotter(dataset_directory=dataset_directory, records=records,
                      cutoff=cutoff, section_width=section_width)

    plotter._Plotter__set_section_width.assert_called_once_with(section_width=section_width)
    assert len(plotter.records) == len(records)


@pytest.mark.parametrize("section_width", [
    10, None, 1000, -200,
])
def test_plotter__set_section_width(section_width: Optional[int], mocker):
    mocker.patch('matplotlib.pyplot.grid')

    plotter = Plotter(dataset_directory="demo_dir", records=[1, 2, 3],
                      cutoff=5, section_width=section_width)

    if section_width is not None:
        plt.grid.assert_called_once()


@pytest.mark.parametrize("dataset_directory, records, processed_records, cutoff", [
    (
            TEST_DEFAULT_DATASET,
            [1, 2],
            [127, 127],
            127,
    ),
    (
            TEST_DEFAULT_DATASET,
            [0, 255, 3, 1, 128, 600],
            [127, 255, 127, 127, 128, 600],
            127,
    ),
])
def test_plotter_add(dataset_directory: str, records: List[int], processed_records: List[int], cutoff: int):
    plotter_no_records = Plotter(dataset_directory=dataset_directory, cutoff=cutoff)
    assert plotter_no_records.records == []

    for record in records:
        plotter_no_records.add_record(record)
        assert processed_records[0:len(plotter_no_records.records)] == plotter_no_records.records

    plotter_full = Plotter(dataset_directory=dataset_directory, cutoff=cutoff, records=records)
    assert plotter_full.records == processed_records


@pytest.mark.parametrize("dataset_directory, show_plot, records", [
    (TEST_DEFAULT_DATASET, True, []),
    (TEST_DEFAULT_DATASET_ADVANCED, False, [1]),
    ("testik", False, [1, 2, 3]),
    ("asfsdafasfd", True, list(range(0, 20, 2))),
])
def test_plotter_show(dataset_directory, show_plot, records, mocker):
    mocker.patch('matplotlib.pyplot.plot')
    mocker.patch('matplotlib.pyplot.savefig')
    mocker.patch('ocr.utils.plotter.Plotter.write_counter')
    mocker.patch('PIL.Image.open')
    plotter = Plotter(dataset_directory)
    map(plotter.add_record, records)
    plotter.show(show_plot=show_plot)
    plt.plot.assert_called_once()
    plt.savefig.assert_called_once()
    plotter.write_counter.assert_called_once()

    if show_plot:
        Image.open.assert_called_once()