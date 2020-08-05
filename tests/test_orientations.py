from pytest import raises

from haiopy import Orientations


def test_orientations_init():
    """Test to init Orientations without optional parameters."""
    orient = Orientations()
    assert isinstance(orient, Orientations)


def test_orientations_init_view_up():
    """Test to init Orientations with view and up vector(s)."""
    # test with single view and up vectors
    views = [1, 0, 0]
    ups = [0, 1, 0]
    Orientations(views, ups)
    # test with multiple view and up vectors
    views = [[1, 0, 0], [0, 0, 1]]
    ups = [[0, 1, 0], [0, 1, 0]]
    Orientations(views, ups)
    # mal-formed lists
    views = [[1, 0, 0], [0, 0]]
    ups = [[0, 1, 0], [0, 0, 0]]
    with raises(ValueError):
        Orientations(views, ups)
    # any of views and ups has zero-length
    views = [[1, 0, 0], [0, 0, 1]]
    ups = [[0, 1, 0], [0, 0, 0]]
    with raises(ValueError):
        Orientations(views, ups)
    # view and up counts not matching
    views = [[1, 0, 0], [0, 0, 1]]
    ups = [[0, 1, 0]]
    with raises(ValueError):
        Orientations(views, ups)
    # views and ups must be either empty or 3D
    views = [0, 1]
    ups = [0, 1]
    with raises(ValueError):
        Orientations(views, ups)
    # view and up vectors must be orthogonal
    views = [1.0, 0.5, 0.1]
    ups = [0, 0, 1]
    with raises(ValueError):
        Orientations(views, ups)
