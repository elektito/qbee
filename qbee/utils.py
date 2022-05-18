import sys


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def parse_data(s):
    BEFORE_ITEM = 1
    READING_UNQUOTED = 2
    READING_QUOTED = 3
    AFTER_ITEM = 4

    whitespace = ' \t'
    state = BEFORE_ITEM
    items = []
    item = ''
    for c in s:
        if state == BEFORE_ITEM:
            if c in whitespace:
                pass
            elif c == ',':
                items.append('')
            elif c == '"':
                state = READING_QUOTED
            else:
                item += c
                state = READING_UNQUOTED
        elif state == READING_UNQUOTED:
            if c == ',':
                items.append(item.strip())
                item = ''
                state = BEFORE_ITEM
            else:
                item += c
        elif state == READING_QUOTED:
            if c == '"':
                items.append(item)
                item = ''
                state = AFTER_ITEM
            else:
                item += c
        elif state == AFTER_ITEM:
            if c in whitespace:
                pass
            elif c == ',':
                state = BEFORE_ITEM
            else:
                return None

    if state == READING_UNQUOTED:
        items.append(item.strip())
    elif state in [BEFORE_ITEM, READING_QUOTED]:
        items.append(item)

    return items
