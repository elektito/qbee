import sys
import struct
import gzip
import pickle
from dataclasses import dataclass, fields
from enum import Enum
from qbee.stmt import Stmt, Block, SubBlock, FunctionBlock
from qbee.node import Node
from qbee.utils import convert_index_to_line_col


class RoutineType(Enum):
    SUB = 1
    FUNCTION = 2


class DebugRecord:
    pass


@dataclass
class DebugNodeRecord(DebugRecord):
    node: Node

    start_offset: int
    end_offset: int

    source_start_offset: int
    source_end_offset: int

    source_start_line: int
    source_start_col: int
    source_end_line: int
    source_end_col: int


@dataclass
class DebugRoutineRecord(DebugNodeRecord):
    name: str
    type: RoutineType
    local_consts: dict[str, tuple]


class DebugInfo:
    def __init__(self, source_code, empty_blocks, compilation,
                 global_consts):
        self.source_code = source_code
        self.empty_blocks = empty_blocks
        self.main_routine = compilation.routines['_main']
        self.user_types = compilation.user_types
        self.global_vars = compilation.global_vars
        self.global_consts = global_consts
        self.routines = {}
        self.blocks = []
        self.stmts = []
        self.other_nodes = []

    def add_node(self, node, start_offset, end_offset):
        start_line, start_col = None, None
        if getattr(node, 'loc_start', None) is not None:
            start_line, start_col = convert_index_to_line_col(
                self.source_code, node.loc_start,
            )

        end_line, end_col = None, None
        if getattr(node, 'loc_end'):
            end_line, end_col = convert_index_to_line_col(
                self.source_code, node.loc_end,
            )

        if isinstance(node, (SubBlock, FunctionBlock)):
            if isinstance(node, SubBlock):
                routine_type=RoutineType.SUB
            else:
                routine_type=RoutineType.FUNCTION

            self.routines[node.name] = DebugRoutineRecord(
                type=routine_type,
                name=node.name,
                start_offset=start_offset,
                end_offset=end_offset,
                source_start_offset=node.loc_start,
                source_end_offset=node.loc_end,
                source_start_line=start_line,
                source_start_col=start_col,
                source_end_line=end_line,
                source_end_col=end_col,
                local_consts={
                    name: (const.type, const.eval())
                    for name, const in node.routine.local_consts.items()
                },
                node=node,
            )

        # routines are also blocks, which we need to keep track of
        # separately. that's why the following is an "if" and not an
        # "elif".

        if isinstance(node, Block):
            self.blocks.append((node, start_offset, end_offset))
        elif isinstance(node, Stmt):
            self.stmts.append(DebugNodeRecord(
                start_offset=start_offset,
                end_offset=end_offset,
                source_start_offset=node.loc_start,
                source_end_offset=node.loc_end,
                source_start_line=start_line,
                source_start_col=start_col,
                source_end_line=end_line,
                source_end_col=end_col,
                node=node,
            ))
        elif isinstance(node, Node):
            self.other_nodes.append(DebugNodeRecord(
                start_offset=start_offset,
                end_offset=end_offset,
                source_start_offset=node.loc_start,
                source_end_offset=node.loc_end,
                source_start_line=start_line,
                source_start_col=start_col,
                source_end_line=end_line,
                source_end_col=end_col,
                node=node,
            ))
        else:
            assert False, 'Not a sub-class of Node'

    def finalize(self):
        def add_node_record(node, start_offset, end_offset):
            start_line, start_col = convert_index_to_line_col(
                self.source_code, node.loc_start,
            )
            end_line, end_col = convert_index_to_line_col(
                self.source_code, node.loc_end,
            )
            self.stmts.append((DebugNodeRecord(
                start_offset=start_offset,
                end_offset=end_offset,
                source_start_offset=node.loc_start,
                source_end_offset=node.loc_end,
                source_start_line=start_line,
                source_start_col=start_col,
                source_end_line=end_line,
                source_end_col=end_col,
                node=node,
            )))

        def get_children(block, block_start, block_end):
            results = []
            for stmt in self.stmts:
                if block_start <= stmt.start_offset and \
                   block_end >= stmt.end_offset:
                    # don't add empty statement, otherwise an empty
                    # statement (like a const) _before_ the block
                    # could be considered as part of the block
                    if stmt.end_offset - stmt.start_offset > 0:
                        results.append(stmt)
            return results

        # add the "start" and "end" statements of each block to the
        # list of statements.
        for block, start_offset, end_offset in self.blocks:
            children = get_children(block, start_offset, end_offset)

            if children:
                children.sort(key=lambda r: r.start_offset)

                # anything between start of block and start of first
                # child statement inside it, is part of the block
                # "start statement"
                first_child = children[0]
                add_node_record(block.start_stmt,
                                start_offset,
                                first_child.start_offset)

                # and anything between the end of the last child
                # statement and the end of the block is part of the
                # "end statement" of the block.
                last_child = children[-1]
                add_node_record(block.end_stmt,
                                last_child.end_offset,
                                end_offset)
            else:
                # there should have been an empty block marker inside.
                for addr in self.empty_blocks:
                    if start_offset <= addr < end_offset:
                        add_node_record(block.start_stmt,
                                        start_offset,
                                        addr)
                        add_node_record(block.end_stmt,
                                        addr,
                                        end_offset)

        self.stmts.sort(key=lambda r: r.start_offset)

        # we don't need these anymore
        del self.blocks
        del self.empty_blocks

    def find_stmt(self, addr, cpu):
        # if we're at the beginning of the module code, there should
        # be a call instruction telling us where the actual beginning
        # of the module code is. we'll use that to find the first
        # statement.
        if addr == 0:
            instr, operands, _ = cpu.get_instruction_at(0)
            if instr.op != 'call':
                return None
            return self.find_stmt(operands[0], cpu)

        # if the address falls within the range of some statement,
        # return the one with the smallest range, excluding any
        # blocks.
        matching = []
        for stmt in self.stmts:
            if stmt.start_offset <= addr < stmt.end_offset:
                matching.append(stmt)

        blocks = [stmt for stmt in matching if isinstance(stmt, Block)]
        non_blocks = [
            stmt for stmt in matching
            if not isinstance(stmt, Block)]

        non_blocks.sort(key=lambda r: r.end_offset - r.start_offset)
        if non_blocks:
            return non_blocks[0]

        # if not however...

        if blocks:
            blocks.sort(key=lambda r: r.end_offset - r.start_offset)

            # if it falls between the start of a block, and the start
            # of the first statement inside that block, it's
            # considered part of the "state statement" of that block
            first_inner_stmt = first_stmt_in_block(block)
            if first_inner_stmt:
                first_inner_offset = first_inner_stmt.start_offset
            else:
                first_inner_offset = block.end_stmt.start_offset
            if block.start_offset <= addr < first_inner_offset:
                return block.start_stmt

            # if it falls between the end of the last statement inside
            # a block and the end of the block, it's considered part
            # of the "end statement" of that block.
            last_inner_stmt = last_stmt_in_block(block)
            if last_inner_stmt:
                last_inner_offset = last_inner_stmt.end_offset
            else:
                last_inner_offset = block.start_stmt.end_offset
            if last_inner_offset <= addr < block.end_offset:
                return block.end_stmt

        return None

    def serialize(self):
        return gzip.compress(pickle.dumps(self))

    @staticmethod
    def deserialize(data):
        return pickle.loads(gzip.decompress(data))


class DebugInfoCollector:
    def __init__(self, source_code, compilation):
        self._source_code = source_code
        self._compilation = compilation
        self._global_consts = compilation.global_consts
        self._stack = []
        self._nodes = []
        self._empty_blocks = []

    def start_node(self, node, code_offset):
        self._stack.append((node, code_offset))

    def end_node(self, node, code_offset):
        start_node, start_offset = self._stack.pop()
        assert start_node == node, 'Incorrect debug info'
        self._nodes.append((node, start_offset, code_offset))

    def mark_empty_block(self, code_offset):
        self._empty_blocks.append(code_offset)

    def get_debug_info(self):
        global_consts = {
            name: (const.type, const.eval())
            for name, const in self._global_consts.items()
        }
        dbg_info = DebugInfo(self._source_code,
                             self._empty_blocks,
                             self._compilation,
                             global_consts)

        for node, start_offset, end_offset in self._nodes:
            dbg_info.add_node(node, start_offset, end_offset)

        dbg_info.finalize()

        return dbg_info
