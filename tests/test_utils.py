from qbee import utils


def test_split_camel1():
    parts = utils.split_camel('ExitSubStmt')
    assert parts == ['Exit', 'Sub', 'Stmt']


def test_split_camel2():
    parts = utils.split_camel('exit')
    assert parts == ['exit']


def test_split_camel3():
    parts = utils.split_camel('')
    assert parts == []

def test_split_camel4():
    parts = utils.split_camel('ABC')
    assert parts == ['A', 'B', 'C']
