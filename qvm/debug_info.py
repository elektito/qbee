import sys
import struct
import gzip
import pickle
from dataclasses import dataclass, fields
from enum import Enum
from qbee.stmt import Stmt, SubBlock, FunctionBlock
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


class DebugInfo:
    def __init__(self, source_code):
        self.source_code = source_code
        self.routines = {}
        self.stmts = []
        self.other_nodes = []

    def add_node(self, node, start_offset, end_offset):
        start_line, start_col = None, None
        if getattr(node, 'loc_start'):
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
                node=node,
            )
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

        self.stmts.sort(key=lambda r: r.start_offset)

    def serialize(self):
        return gzip.compress(pickle.dumps(self))

    @staticmethod
    def deserialize(data):
        return pickle.loads(gzip.decompress(data))


class DebugInfoCollector:
    def __init__(self, source_code):
        self._source_code = source_code
        self._stack = []
        self._nodes = []

    def start_node(self, node, code_offset):
        self._stack.append((node, code_offset))

    def end_node(self, node, code_offset):
        start_node, start_offset = self._stack.pop()
        assert start_node == node, 'Incorrect debug info'
        self._nodes.append((node, start_offset, code_offset))

    def get_debug_info(self):
        dbg_info = DebugInfo(self._source_code)

        for node, start_offset, end_offset in self._nodes:
            dbg_info.add_node(node, start_offset, end_offset)

        return dbg_info
