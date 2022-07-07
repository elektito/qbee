from qbee.utils import parse_data, Empty


def test_simple():
    items = parse_data(' foo, bar, 1')
    assert items == ['foo', 'bar', '1']


def test_inner_space():
    items = parse_data(' foo, b  a  r  , 1')
    assert items == ['foo', 'b  a  r', '1']


def test_inner_space_last_item():
    items = parse_data(' foo, b  a  r  ')
    assert items == ['foo', 'b  a  r']


def test_quoted():
    items = parse_data(' foo, "foo,bar " ,1')
    assert items == ['foo', 'foo,bar ', '1']


def test_quoted_at_the_end():
    items = parse_data(' foo, "spam"')
    assert items == ['foo', 'spam']


def test_quoted_at_the_beginning():
    items = parse_data(' "foo", 10')
    assert items == ['foo', '10']


def test_partially_quoted():
    items = parse_data('foo, "spam,  bar , 100')
    assert items == ['foo', 'spam,  bar , 100']


def test_quote_in_the_middle():
    items = parse_data('foo, a"b,c"d, 90')
    assert items == ['foo', 'a"b', 'c"d', '90']


def test_empty_items():
    items = parse_data('foo,, ,   ,1')
    assert items == ['foo', Empty.value, Empty.value, Empty.value, '1']


def test_empty_quoted():
    items = parse_data('foo, ""   , 100')
    assert items == ['foo', '', '100']


def test_empty():
    items = parse_data('')
    assert items == [Empty.value]


def test_empty_with_spaces():
    items = parse_data('  \t ')
    assert items == [Empty.value]


def test_syntax_error_multi_quoted():
    items = parse_data('foo, "foo" "bar", 2')
    assert items is None
