"""
Assembler Phase I for DM2022W assembly language.

This assembler is for fully resolved instructions,
which may be the output of assembler_phase1.py, which
transforms instructions with symbolic addresses into
instructions with fully resolved (PC-relative) addresses.

Assembly instruction format with all options is 

label: instruction

Labels are resolved (translated into addresses) in
phase I.  In fully resolved assembly code, they serve
only as documentation.

Both parts are optional:  A label may appear without 
an instruction, and an instruction may appear without 
a label. 

A label is just an alphabetic string, eg.,
  myDogBoo but not Betcha_5_Dollars

An instruction has the following form: 

  opcode/predicate  target,src1,src2[disp]

Opcode is required, and should be one of the DM2022W 
instruction codes (ADD, MOVE, etc); case-insensitive

/predicate is optional.  If present, it should be some 
combination of N,Z,P, e.g., /NP would be "execute if 
not zero".  If /predicate is not given, it is interpreted 
as /ALWAYS, which is an alias for /NZP. 

target is a register number (r0,r1, ... r15) or one of the
register aliases ZERO, PC, SP, etc. 

src1 and src2 are likewise register specifiers. 

[disp] is optional.  If present, it is a 12 bit 
signed integer displacement.  If absent, it is 
treated as [0]. 

DATA is a pseudo-operation:
   myvar:  DATA   18
indicates that the integer value 18
should be stored at this location, rather than
a DM2022W instruction.

"""
from instr_format import Instruction, OpCode, CondFlag, NAMED_REGS
import argparse

from typing import Union, List, Dict
from enum import Enum, auto

import sys
import re

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Configuration constants
ERROR_LIMIT = 5    # Abandon assembly if we exceed this

# Exceptions raised by this module
class SyntaxError(Exception):
    pass

###
# The whole instruction line is encoded as a single
# regex with capture names for the parts we might
# refer to. Error messages will be crappy (we'll only
# know that the pattern didn't match, and not why), but 
# we get a very simple match/process cycle.  By creating
# a dict containing the captured fields, we can determine
# which optional parts are present (e.g., there could be
# label without an instruction or an instruction without
# a label).
###


# To simplify client code, we'd like to return a dict with
# the right fields even if the line is syntactically incorrect. 
DICT_NO_MATCH = { 'label': None, 'opcode': None, 'predicate': None,
                      'target': None, 'src1': None, 'src2': None,
                      'offset': None, 'comment': None }


###
# Although the DM2022W instruction set is very simple, a source
# line can still come in several forms.  Each form (even comments)
# can start with a label.
###

class AsmSrcKind(Enum):
    """Distinguish which kind of assembly language instruction
    we have matched.  Each element of the enum corresponds to
    one of the regular expressions below.
    """
    # Blank or just a comment, optionally
    # with a label
    COMMENT = auto()
    # Fully specified  (all addresses resolved)
    FULL = auto()
    # A data location, not an instruction
    DATA = auto()
    # An instruction that refers to a memory
    # location in place of its source and offset
    # parts.
    MEMOP = auto()
    # Jump instructions refer to memory locations
    # like MEMOP but implicitly target register 15
    JUMP = auto()


# Lines that contain only a comment (and possibly a label).
# This includes blank lines and labels on a line by themselves.
#
ASM_COMMENT_PAT = re.compile(r"""
   # Optional label 
   \s*  
   (
     (?P<label> [a-zA-Z]\w*):
   )?
   \s*
   # Optional comment follows # or ; 
   (
     (?P<comment>[\#;].*)
   )?       
   \s*$             
   """, re.VERBOSE)

# Instructions with fully specified fields. We can generate
# code directly from these.
ASM_FULL_PAT = re.compile(r"""
   # Optional label 
   \s* 
   (
     (?P<label> [a-zA-Z]\w*):
   )?
   \s*
    # The instruction proper 
    (?P<opcode>    [a-zA-Z]+)           # Opcode
    (/ (?P<predicate> [A-Z]+) )?   # Predicate (optional)
    \s+
    (?P<target>    r[0-9]+),            # Target register
    (?P<src1>      r[0-9]+),            # Source register 1
    (?P<src2>      r[0-9]+)             # Source register 2
    (\[ (?P<offset>[-]?[0-9]+) \])?     # Offset (optional)
   # Optional comment follows # or ; 
   (
     \s*
     (?P<comment>[\#;].*)
   )?       
   \s*$             
   """, re.VERBOSE)

# Instructions that refer to a memory location.  src1, src2, and offset
# are replaced by the memory reference.
ASM_MEMOP_PAT = re.compile(r"""
   # Optional label 
   \s* 
   (
     (?P<label> [a-zA-Z]\w*):
   )?
   \s*
    # The instruction proper 
    (?P<opcode>    [a-zA-Z]+)           # Opcode
    (/ (?P<predicate> [A-Z]+) )?   # Predicate (optional)
    \s+
    (?P<target>    r[0-9]+),            # Target register
    (?P<labelref>  [a-zA-Z]\w*)         # reference to label
   # Optional comment follows # or ; 
   (
     \s*
     (?P<comment>[\#;].*)
   )?       
   \s*$             
   """, re.VERBOSE)

# The JUMP pseudo-operation implicitly targets register 15.
ASM_JUMP_PAT = re.compile(r"""
   # Optional label 
   \s* 
   (
     (?P<label> [a-zA-Z]\w*):
   )?
   \s*
    # The instruction proper 
    (?P<opcode>    JUMP)           # Opcode
    (/ (?P<predicate> [A-Z]+) )?   # Predicate (optional)
    \s+
    (?P<labelref>  [a-zA-Z]\w*)         # reference to label
   # Optional comment follows # or ; 
   (
     \s*
     (?P<comment>[\#;].*)
   )?       
   \s*$             
   """, re.VERBOSE)


# Defaults for values that ASM_FULL_PAT makes optional
INSTR_DEFAULTS = [ ('predicate', 'ALWAYS'), ('offset', '0') ]

# A data word in memory; not a DM2022W instruction
#
ASM_DATA_PAT = re.compile(r""" 
   \s* 
   # Optional label 
   (
     (?P<label> [a-zA-Z]\w*):
   )?
   # The instruction proper  
   \s*
    (?P<opcode>    DATA)           # Opcode
   # Optional data value
   \s*
   (?P<value>  (0x[a-fA-F0-9]+)
             | ([0-9]+))?
    # Optional comment follows # or ; 
   (
     \s*
     (?P<comment>[\#;].*)
   )?       
   \s*$             
   """, re.VERBOSE)


PATTERNS = [(ASM_FULL_PAT, AsmSrcKind.FULL),
            (ASM_DATA_PAT, AsmSrcKind.DATA),
            (ASM_COMMENT_PAT, AsmSrcKind.COMMENT),
            (ASM_MEMOP_PAT, AsmSrcKind.MEMOP),
            (ASM_JUMP_PAT, AsmSrcKind.JUMP)
            ]

def parse_line(line: str) -> dict:
    """Parse one line of assembly code.
    Returns a dict containing the matched fields,
    some of which may be empty.  Raises SyntaxError
    if the line does not match assembly language
    syntax. Sets the 'kind' field to indicate
    which of the patterns was matched.
    """
    log.debug("\nParsing assembler line: '{}'".format(line))
    # Try each kind of pattern
    for pattern, kind in PATTERNS:
        match = pattern.fullmatch(line)
        if match:
            fields = match.groupdict()
            fields["kind"] = kind
            log.debug("Extracted fields {}".format(fields))
            return fields
    raise SyntaxError("Assembler syntax error in {}".format(line))


def fill_defaults(fields: dict) -> None:
    """Fill in default values for optional fields of instruction"""
    for key, value in INSTR_DEFAULTS:
        if fields[key] == None:
            fields[key] = value


def value_parse(int_literal: str) -> int:
    """Parse an integer literal that could look like
    42 or like 0x2a
    """
    if int_literal.startswith("0x"):
        return int(int_literal, 16)
    else:
        return int(int_literal, 10)


def resolve(lines: List[str]) -> Dict[str, int]:
    """
    Build table associating labels in the source code
    with addresses.
    """
    labels = { }
    address = 0
    for lnum in range(len(lines)):
        line = lines[lnum].rstrip()
        log.debug("Resolving line {}: {}".format(lnum, line))
        try:
            fields = parse_line(line)
            if fields["label"] is not None:
                label = fields["label"]
                labels[label] = address
                log.debug(f"Resolved label {label} -> {address}")
            if fields["kind"] != AsmSrcKind.COMMENT:
                address += 1
        except  Exception:
            # Just ignore errors here; they will be handled in
            # transform
            pass
    return labels

def fix_optional_fields(fields: Dict[str, str]):
    """Fill in values of optional fields label,
    predicate, and comment, adding the punctuation
    they require.
    """
    if fields["label"] is None:
        fields["label"] = "     "
    else:
        fields["label"] = fields["label"] + ":"
    if fields["predicate"] is None:
        fields["predicate"] = ""
    else:
        fields["predicate"] = "/" + fields["predicate"]
    if fields["comment"] is None:
        fields["comment"] = ""
    return


def transform(lines: List[str]) -> List[str]:
    """
    Transform some assembly language lines, leaving others
    unchanged.
    Initial version:  No changes to any source line.

    Planned version:
       again:   STORE r1,x
                SUB   r1,r0,r0[1]
                JUMP/P  again
                HALT r0,r0,r0
       x:       DATA 0
    should become
       again:   STORE r1,r0,r15[4]   #x
                SUB   r1,r0,r0[1]
                ADD/P r15,r0,r15[-2]
                HALT r0,r0,r0
       x:       DATA 0
     """
    labels = resolve(lines)
    transformed = [ ]
    error_count = 0
    address = 0
    for lnum in range(len(lines)):
        line = lines[lnum].rstrip()
        log.debug("Processing line {}: {}".format(lnum, line))
        try: 
            fields = parse_line(line)
            if fields["kind"] == AsmSrcKind.FULL:
                log.debug("Passing through FULL instruction")
                transformed.append(line)
            elif fields["kind"] == AsmSrcKind.MEMOP:
                ref = fields["labelref"]
                mem_addr = labels[ref]
                pc_relative = mem_addr - address
                fix_optional_fields(fields)
                f = fields
                full = (f"{f['label']}   {f['opcode']}{f['predicate']} " +
                    f" {f['target']},r0,r15[{pc_relative}] #{ref} " +
                    f" {f['comment']}")
                transformed.append(full)
            elif fields["kind"] == AsmSrcKind.JUMP:
                ref = fields["labelref"]
                mem_addr = labels[ref]
                pc_relative = mem_addr - address
                fix_optional_fields(fields)
                f = fields
                full = (f"{f['label']}   ADD{f['predicate']} " +
                    f" r15,r0,r15[{pc_relative}] #{ref} " +
                    f" {f['comment']}")
                transformed.append(full)
            elif fields["kind"] == AsmSrcKind.DATA:
                log.debug("Passing through DATA instruction")
                transformed.append(line)
            elif fields["kind"] == AsmSrcKind.COMMENT:
                log.debug("Passing comment through without change")
                transformed.append(line)
            else:
                log.warning(f" -xxx- No pattern matched -xxx- : {line}")
                transformed.append(line)
            if fields["kind"] != AsmSrcKind.COMMENT:
                address += 1
        except SyntaxError as e:
            error_count += 1
            print("Syntax error in line {}: {}".format(lnum, line), file=sys.stderr)
        except KeyError as e:
            error_count += 1
            print("Unknown word in line {}: {}".format(lnum, e), file=sys.stderr)
        except Exception as e:
            error_count += 1
            print("Exception encountered in line {}: {}".format(lnum, e), file=sys.stderr)
        if error_count > ERROR_LIMIT:
            print("Too many errors; abandoning", file=sys.stderr)
            sys.exit(1)
    return transformed

def cli() -> object:
    """Get arguments from command line"""
    parser = argparse.ArgumentParser(description="Duck Machine Assembler (phase 1)")
    parser.add_argument("sourcefile", type=argparse.FileType('r'),
                            nargs="?", default=sys.stdin,
                            help="Duck Machine assembly code file")
    parser.add_argument("objfile", type=argparse.FileType('w'),
                            nargs="?", default=sys.stdout, 
                            help="Transformed assembly language file")
    args = parser.parse_args()
    return args


def main():
    """"Pre-process duck machine assembly language"""
    args = cli()
    lines = args.sourcefile.readlines()
    transformed = transform(lines)
    log.debug(f"Transformed: \n{transformed}")
    for line in transformed:
        print(line,file=args.objfile)

if __name__ == "__main__":
    main()

