from plotlib import DimensionsSingle, FigSize, Margins


def test_dimensions_single():
    DimensionsSingle(Margins(1, 1, 1, 1), FigSize(6, 4))

    print(DimensionsSingle.from_no_height(Margins(1, 1, 1, 1), 6, 2))
    print("testestsetses")
