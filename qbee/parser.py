from pyparsing import ParseException, ParseSyntaxException
from .grammar import line as line_rule
from .program import Program
from .stmt import Block
from .exceptions import SyntaxError


def parse_string(input_string):
    block_start_types = tuple(Block.known_blocks.keys())
    block_end_types = tuple(b.end_stmt
                            for b in Block.known_blocks.values())

    entered_blocks = []
    cur_block_body = []

    line_loc = 0
    for line in input_string.split('\n'):
        try:
            line_node = line_rule.parse_string(line, parse_all=True)
        except (ParseException, ParseSyntaxException) as e:
            raise SyntaxError(loc=line_loc + e.loc)
        except SyntaxError as e:
            raise SyntaxError(loc=line_loc + e.loc_start,
                              msg=e.msg)

        # the line is always a list of one containing the actual Line
        # node
        line_node = line_node[0]

        for stmt in line_node.nodes:
            update_node_loc(stmt, line_loc)

            if isinstance(stmt, block_start_types):
                entered_blocks.append((stmt, cur_block_body))
                cur_block_body = []
            elif isinstance(stmt, block_end_types):
                block_end = stmt
                if not entered_blocks:
                    expected_start = Block.start_stmt_from_end(
                        block_end)
                    raise SyntaxError(
                        loc=line_loc,
                        msg=(f'{block_end.node_name()} without '
                             f'{expected_start.node_name()}'))
                block_start, prev_body = entered_blocks.pop()
                block = Block.create(block_start,
                                     block_end,
                                     cur_block_body)
                prev_body.append(block)
                cur_block_body = prev_body
            else:
                cur_block_body.append(stmt)

        line_loc += len(line) + 1

    if entered_blocks:
        block, _ = entered_blocks[-1]
        raise SyntaxError(
            loc=block.loc_start,
            msg=f'{block.node_name()} block not closed')

    return Program(cur_block_body)


def update_node_loc(node, offset):
    """
Since we parse each line separately, the location values set by the
parser are based on the beginning of the line. This function can then
be used to add the offset of the beginning of the line to a node and
all its children.

    """
    if node.loc_start is not None:
        node.loc_start += offset
    if node.loc_end is not None:
        node.loc_end += offset
    for child in node.children:
        update_node_loc(child, offset)
