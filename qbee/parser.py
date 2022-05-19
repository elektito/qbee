from pyparsing import ParseException
from .grammar import line as line_rule
from .program import Program
from .stmt import Block


def parse_string(input_string):
    block_start_types = tuple(Block.known_blocks.keys())
    block_end_types = tuple(b.end_stmt
                            for b in Block.known_blocks.values())

    entered_blocks = []
    cur_block_body = []

    for line in input_string.split('\n'):
        try:
            line_node = line_rule.parse_string(line, parse_all=True)
        except ParseException as e:
            print(e.explain())
            exit(1)
            #raise SyntaxError(*syntax_error_args)

        for stmt in line_node[0].nodes:
            if isinstance(stmt, block_start_types):
                entered_blocks.append((stmt, cur_block_body))
                cur_block_body = []
            elif isinstance(stmt, block_end_types):
                block_start, prev_body = entered_blocks.pop()
                block_end = stmt
                block = Block.create(block_start, block_end, cur_block_body)
                prev_body.append(block)
                cur_block_body = prev_body
            else:
                cur_block_body.append(stmt)

    if entered_blocks:
        raise SytnaxError(*some_args,
                          f'Block {start_block_name} not closed')

    return Program(cur_block_body)
