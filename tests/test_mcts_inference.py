import pytest
from mcts_inference.mcts import Constraint as Constraint


def test_constraint_initialization() -> None:
    pass


def test_constraint_bool() -> None:
    constraint = Constraint(time=True, d_time=1, verbose=True)

    result = bool(constraint)

    assert (result is True)


# def test_empty_slap():
#     assert slap_many(LikeState.empty, '') is LikeState.empty


# def test_single_slaps():
#     assert slap_many(LikeState.empty, 'l') is LikeState.liked
#     assert slap_many(LikeState.empty, 'd') is LikeState.disliked


# @pytest.mark.parametrize("test_input,expected", [
#     ('ll', LikeState.empty),
#     ('dd', LikeState.empty),
#     ('ld', LikeState.disliked),
#     ('dl', LikeState.liked),
#     ('ldd', LikeState.empty),
#     ('lldd', LikeState.empty),
#     ('ddl', LikeState.liked),
# ])
# def test_multi_slaps(test_input, expected):
#     assert slap_many(LikeState.empty, test_input) is expected


@pytest.mark.skip(reason="example skip")
def test_skip() -> None:
    assert (True is False)


@pytest.mark.xfail
def test_divide_by_zero() -> None:
    assert 1 / 0 == 1


# def test_invalid_slap():
#     with pytest.raises(ValueError):
#         slap_many(LikeState.empty, 'x')


# @pytest.mark.xfail
# def test_db_slap(db_conn):
#     db_conn.read_slaps()
#     assert ...


# def test_print(capture_stdout):
#     print("hello")
#     assert capture_stdout["stdout"] == "hello\n"

# # def test_many_slaps():
# #     assert slap_many(LikeState.empty, 'll') is LikeState.empty
# #     assert slap_many(LikeState.empty, 'dd') is LikeState.empty
# #     assert slap_many(LikeState.empty, 'ld') is LikeState.disliked
# #     assert slap_many(LikeState.empty, 'dl') is LikeState.liked
# #     assert slap_many(LikeState.empty, 'ldd') is LikeState.empty
# #     assert slap_many(LikeState.empty, 'lldd') is LikeState.empty
# #     assert slap_many(LikeState.empty, 'ddl') is LikeState.liked
