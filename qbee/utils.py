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


def display_with_context(text, loc_start, loc_end=None, msg='Error'):
    n_context_lines = 3

    prev_lines = []
    next_lines = []
    target_lines = []
    i = 0
    target_col = None
    while i < len(text):
        try:
            nl = text.index('\n', i)
        except ValueError:
            text += '\n'
            nl = len(text) - 1

        line = text[i:nl]
        if ((loc_end is not None and
             (i <= loc_start <= nl or
              (loc_start >=i and
               loc_end <= nl))) or
            (loc_end is None and i <= loc_start <= nl)):
            target_lines.append(line)
            target_col = loc_start - i + 1
        elif i < loc_start:
            prev_lines.append(line)
        else:
            next_lines.append(line)

        if len(next_lines) == n_context_lines:
            break

        i = nl + 1

    for line in prev_lines[-n_context_lines:]:
        eprint(' || ', line)
    for line in target_lines:
        eprint(' >> ', line)
    if len(target_lines) == 1:
        eprint(' :: ' + ' ' * target_col + '^')
        eprint(' :: ' + ' ' * target_col + msg)
    for line in next_lines:
        eprint(' || ', line)


def split_camel(name):
    """Split a camel/pascal case name into words. For exaple ExitSubStmt
is split into ['Exit', 'Sub', 'Stmt']."""

    parts = []
    cur = ''
    for c in name:
        if c.isupper() and cur:
            parts.append(cur)
            cur = ''
        cur += c
    if cur:
        parts.append(cur)
    return parts
