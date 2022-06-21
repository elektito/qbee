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
    assert loc_start is not None

    n_context_lines = 3

    prev_lines = []
    next_lines = []
    target_lines = []
    i = 0
    target_col = None
    target_line = None
    cur_line = 0
    while i < len(text):
        cur_line += 1
        try:
            nl = text.index('\n', i)
        except ValueError:
            text += '\n'
            nl = len(text) - 1

        line = text[i:nl]
        if ((loc_end is not None and
             (i <= loc_start <= nl or
              (loc_start >= i and loc_end <= nl))) or
            (loc_end is None and i <= loc_start <= nl)
        ):
            target_lines.append(line)
            target_col = loc_start - i + 1
            target_line = cur_line
        elif i < loc_start:
            prev_lines.append(line)
        else:
            next_lines.append(line)

        if len(next_lines) == n_context_lines:
            break

        i = nl + 1

    line_no_width = len(str(target_line + n_context_lines))

    prev_lines = prev_lines[-n_context_lines:]
    for i, line in enumerate(prev_lines):
        line_no = target_line - (len(prev_lines) - i)
        eprint(f' {line_no: >{line_no_width}} || ', line)

    for line in target_lines:
        eprint(f' {target_line: >{line_no_width}} >> ', line)
    if len(target_lines) == 1:
        eprint(' ' * line_no_width + '  :: ' + ' ' * target_col + '^')
        eprint(' ' * line_no_width + '  :: ' + ' ' * target_col + msg)
        eprint(' ' * line_no_width + '  :: ')


    for i, line in enumerate(next_lines):
        eprint(f' {target_line + i + 1: >{line_no_width}} || ', line)


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


def convert_index_to_line_col(text, offset):
    """
Convert an index into a text to a (line, column) pair. The returned
values are 1-based.
    """

    line = 1
    col = 0
    for idx, char in enumerate(text):
        if idx == offset:
            break

        if char == '\n':
            line += 1
            col = 1
            continue

        col += 1
    else:
        return None, None

    return line, col
