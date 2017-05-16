import os
import re
import string
import timeit
import token
from token import *

import glob2

__all__ = [x for x in dir(token) if not x.startswith("_")]
__all__ += ["COMMENT", "tokenize", "generate_tokens", "NL", "untokenize"]
del x
del token

COMMENT = N_TOKENS
tok_name[COMMENT] = 'COMMENT'
NL = N_TOKENS + 1
tok_name[NL] = 'NL'
N_TOKENS += 2


def group(*choices): return '(' + '|'.join(choices) + ')'


def any(*choices): return group(*choices) + '*'


def maybe(*choices): return group(*choices) + '?'


Whitespace = r'[ \f\t]*'
Comment = r'#[^\r\n]*'
Ignore = Whitespace + any(r'\\\r?\n' + Whitespace) + any(Comment)
Name = r'[a-zA-Z_]\w*'

Hexnumber = r'0[xX][\da-fA-F]+[lL]?'
Octnumber = r'(0[oO][0-7]+)|(0[0-7]*)[lL]?'
Binnumber = r'0[bB][01]+[lL]?'
Decnumber = r'[1-9]\d*[lL]?'
Intnumber = group(Hexnumber, Binnumber, Octnumber, Decnumber)
Exponent = r'[eE][-+]?\d+'
Pointfloat = group(r'\d+\.\d*', r'\.\d+') + maybe(Exponent)
Expfloat = r'\d+' + Exponent
Floatnumber = group(Pointfloat, Expfloat)
Imagnumber = group(r'\d+[jJ]', Floatnumber + r'[jJ]')
Number = group(Imagnumber, Floatnumber, Intnumber)

# Tail end of ' string.
Single = r"[^'\\]*(?:\\.[^'\\]*)*'"
# Tail end of " string.
Double = r'[^"\\]*(?:\\.[^"\\]*)*"'
# Tail end of ''' string.
Single3 = r"[^'\\]*(?:(?:\\.|'(?!''))[^'\\]*)*'''"
# Tail end of """ string.
Double3 = r'[^"\\]*(?:(?:\\.|"(?!""))[^"\\]*)*"""'
Triple = group("[uUbB]?[rR]?'''", '[uUbB]?[rR]?"""')
# Single-line ' or " string.
Bracket = '[][(){}]'
String = group(r"[uUbB]?[rR]?'[^\n'\\]*(?:\\.[^\n'\\]*)*'",
               r'[uUbB]?[rR]?"[^\n"\\]*(?:\\.[^\n"\\]*)*"')

# Because of leftmost-then-longest match semantics, be sure to put the
# longest operators first (e.g., if = came before ==, == would get
# recognized as two instances of =).
Operator = group(r"\*\*=?", r">>=?", r"<<=?", r"<>", r"!=",
                 r"//=?",
                 r"[+\-*/&|^=<>]=?",
                 r"~")

Special = group(r'\r?\n', r'[:;.,`@]')
Funny = group(Operator, Bracket, Special)

PlainToken = group(Number, Funny, String, Name)
Token = Ignore + PlainToken

# First (or only) line of ' or " string.
# ContStr = group(r"[uUbB]?[rR]?'[^\n'\\]*(?:\\.[^\n'\\]*)*" +
#                 group("'", r'\\\r?\n'),
#                 r'[uUbB]?[rR]?"[^\n"\\]*(?:\\.[^\n"\\]*)*' +
#                 group('"', r'\\\r?\n'))
ContStr = group(r"(\")", r"(\')", r"%\w+%?", r"(%)", r"(r''')")

PseudoExtras = group(r'\\\r?\n|\Z', Comment, Triple)
PseudoToken = Whitespace + group(PseudoExtras, Number, Funny, ContStr, Name)

tokenprog, pseudoprog, single3prog, double3prog = map(
    re.compile, (Token, PseudoToken, Single3, Double3))
endprogs = {"'": re.compile(Single), '"': re.compile(Double),
              '"""': double3prog,
              'r"""': double3prog,
            "u'''": single3prog, 'u"""': double3prog,
            "ur'''": single3prog, 'ur"""': double3prog,
            "R'''": single3prog, 'R"""': double3prog,
            "U'''": single3prog, 'U"""': double3prog,
            "uR'''": single3prog, 'uR"""': double3prog,
            "Ur'''": single3prog, 'Ur"""': double3prog,
            "UR'''": single3prog, 'UR"""': double3prog,
            "b'''": single3prog, 'b"""': double3prog,
            "br'''": single3prog, 'br"""': double3prog,
            "B'''": single3prog, 'B"""': double3prog,
            "bR'''": single3prog, 'bR"""': double3prog,
            "Br'''": single3prog, 'Br"""': double3prog,
            "BR'''": single3prog, 'BR"""': double3prog,
            'r': None, 'R': None, 'u': None, 'U': None,
            'b': None, 'B': None}
#"r'''", "'''", "r'''": single3prog, "'''": single3prog,
triple_quoted = {}
for t in ( '"""', 'r"""', "R'''", 'R"""',
          "u'''", 'u"""', "U'''", 'U"""',
          "ur'''", 'ur"""', "Ur'''", 'Ur"""',
          "uR'''", 'uR"""', "UR'''", 'UR"""',
          "b'''", 'b"""', "B'''", 'B"""',
          "br'''", 'br"""', "Br'''", 'Br"""',
          "bR'''", 'bR"""', "BR'''", 'BR"""'):
    triple_quoted[t] = t
single_quoted = {}
for t in ("'", '"',
          "r'", 'r"', "R'", 'R"',
          "u'", 'u"', "U'", 'U"',
          "ur'", 'ur"', "Ur'", 'Ur"',
          "uR'", 'uR"', "UR'", 'UR"',
          "b'", 'b"', "B'", 'B"',
          "br'", 'br"', "Br'", 'Br"',
          "bR'", 'bR"', "BR'", 'BR"'):
    single_quoted[t] = t

tabsize = 8
filename = 'tokresults1.txt'

class TokenError(Exception):
    pass


class StopTokenizing(Exception):
    pass


def file_splitting(data, percent):
    line_num = int(round(percent * len(data)))
    split = data[:]
    return split[line_num:], split[:line_num]


def printtoken(type, token, srow_scol, erow_ecol, line):
    tok = " "
    srow, scol = srow_scol
    erow, ecol = erow_ecol

    with open(filename, "a+") as outfile:
        if tok_name[type] == "NEWLINE" or tok_name[type] == "NL" or tok_name[type] == "DEDENT" or tok_name[type] == "ENDMARKER" or tok_name[type] == "INDENT" or tok_name[type] == "COMMENT":
            pass
        elif tok_name[type] == "STRING" or tok_name[type] == "NAME" or tok_name[type] == "NUMBER":
            outfile.write(token.lower() + '\n')


def tokenize(readline, tokeneater=printtoken):
    try:
        tokenize_loop(readline, tokeneater)
    except StopTokenizing:
        pass


def tokenize_loop(readline, tokeneater):
    for token_info in generate_tokens(readline):
        tokeneater(*token_info)


def generate_tokens(readline):
    lnum = parenlev = continued = 0
    namechars, numchars = string.ascii_letters + '_', '0123456789'
    contstr, needcont = '', 0
    contline = None
    indents = [0]

    while 1:  # loop over lines in stream
        try:
            line = readline()
        except StopIteration:
            line = ''
        lnum += 1
        pos, max = 0, len(line)

        if contstr:  # continued string
            if not line:
                raise TokenError, ("EOF in multi-line string", strstart)
            endmatch = endprog.match(line)
            if endmatch:
                pos = end = endmatch.end(0)
                if contstr.startswith('"""') or contstr.startswith("'''"):
                    pass
                else:
                    yield (STRING, contstr + line[:end], strstart, (lnum, end), contline + line)
                contstr, needcont = '', 0
                contline = None
            elif needcont and line[-2:] != '\\\n' and line[-3:] != '\\\r\n':
                yield (ERRORTOKEN, contstr + line,
                       strstart, (lnum, len(line)), contline)
                contstr = ''
                contline = None
                continue
            else:
                contstr = contstr + line
                contline = contline + line
                continue

        if parenlev == 0 and not continued:  # new statement
            if not line: break
            column = 0
            while pos < max:  # measure leading whitespace
                if line[pos] == ' ':
                    column += 1
                elif line[pos] == '\t':
                    column = (column // tabsize + 1) * tabsize
                elif line[pos] == '\f':
                    column = 0
                else:
                    break
                pos += 1
            if pos == max:
                break

            if line[pos] in '#\r\n':  # skip comments or blank lines
                if line[pos] == '#':
                    comment_token = line[pos:].rstrip('\r\n')
                    nl_pos = pos + len(comment_token)
                    yield (COMMENT, comment_token,
                           (lnum, pos), (lnum, pos + len(comment_token)), line)
                    yield (NL, line[nl_pos:],
                           (lnum, nl_pos), (lnum, len(line)), line)
                else:
                    yield ((NL, COMMENT)[line[pos] == '#'], line[pos:],
                           (lnum, pos), (lnum, len(line)), line)
                continue

            if column > indents[-1]:  # count indents or dedents
                indents.append(column)
                yield (INDENT, line[:pos], (lnum, 0), (lnum, pos), line)
            while column < indents[-1]:
                if column not in indents:
                    raise IndentationError(
                        "unindent does not match any outer indentation level",
                        ("<tokenize>", lnum, pos, line))
                indents = indents[:-1]
                yield (DEDENT, '', (lnum, pos), (lnum, pos), line)

        else:  # continued statement
            if not line:
                raise TokenError, ("EOF in multi-line statement", (lnum, 0))
            continued = 0

        while pos < max:
            pseudomatch = pseudoprog.match(line, pos)
            if pseudomatch:  # scan for tokens
                start, end = pseudomatch.span(1)
                spos, epos, pos = (lnum, start), (lnum, end), end
                if start == end:
                    continue
                token, initial = line[start:end], line[start]

                if initial in numchars or \
                        (initial == '.' and token != '.'):  # ordinary number
                    yield (NUMBER, "NUM", spos, epos, line)
                elif initial in '\r\n':
                    yield (NL if parenlev > 0 else NEWLINE,
                           token, spos, epos, line)
                elif initial == '#':
                    assert not token.endswith("\n")
                    yield (COMMENT, token, spos, epos, line)
                elif token in triple_quoted:
                    endprog = endprogs[token]
                    endmatch = endprog.match(line, pos)
                    if endmatch:  # all on one line
                        pos = endmatch.end(0)
                        token = line[start:pos]
                        yield (STRING, token, spos, (lnum, pos), line)
                    else:
                        strstart = (lnum, start)  # multiple lines
                        contstr = line[start:]
                        contline = line
                        break
                elif initial in single_quoted or \
                                token[:2] in single_quoted or \
                                token[:3] in single_quoted:
                    if token[-1] == '\n':  # continued string
                        strstart = (lnum, start)
                        endprog = (endprogs[initial] or endprogs[token[1]] or
                                   endprogs[token[2]])
                        contstr, needcont = line[start:], 1
                        contline = line
                        break
                    else:  # ordinary string
                        yield (STRING, token, spos, epos, line)
                elif initial in namechars:  # ordinary name
                    yield (NAME, token, spos, epos, line)
                elif initial == '\\':  # continued stmt
                    continued = 1
                else:
                    if initial in '([{':
                        parenlev += 1
                    elif initial in ')]}':
                        parenlev -= 1
                    yield (STRING, token, spos, epos, line)
            else:
                yield (ERRORTOKEN, line[pos],
                       (lnum, pos), (lnum, pos + 1), line)
                pos += 1

    for indent in indents[1:]:  # pop remaining indent levels
        yield (DEDENT, '', (lnum, 0), (lnum, 0), '')
    yield (ENDMARKER, '', (lnum, 0), (lnum, 0), '')


if __name__ == '__main__':  # testing
    #filename = 'tok_files.txt'
    # read_files = glob2.glob("../misc/ansible/*.py")
    #read_files = glob2.glob("../PythonCodeDownloader/files/**/*.py")
    read_files = glob2.glob("../misc/sample.py")
    print "Number of files to tokenize : %d \n" % (len(read_files))
    i = 0
    start = timeit.default_timer()
    for file_ in read_files:
        print "Tokenizing file : %s" % file_
        try:
            tokenize(open(file_).readline)
        except:
            i += 1
            pass

    #print "\n Number of files untokenized : %d " % i
    print "\n\n *** Splitting into train-test-valid files ***"

    lines = []

    with open(filename) as f_in:
        lines = list(line for line in (l.strip() for l in f_in) if line)

    train_data, test_valid_data = file_splitting(lines, 0.4)
    test_data, valid_data = file_splitting(test_valid_data, 0.5)

    with open('../data/py_files1/train.txt', 'w+') as train_file:
        for item in train_data:
            train_file.write(item + '\n')

    with open('../data/py_files1/test.txt', 'w+') as test_file:
        for item in test_data:
            test_file.write(item + '\n')

    with open('../data/py_files1/valid.txt', 'w+') as valid_file:
        for item in valid_data:
            valid_file.write(item + '\n')

    print "\nTrain.txt file size : %d bytes" % (os.path.getsize('../data/py_files1/train.txt'))
    print "\nTest.txt file size  : %d bytes" % (os.path.getsize('../data/py_files1/test.txt'))
    print "\nValid.txt file size : %d bytes" % (os.path.getsize('../data/py_files1/valid.txt'))

    stop = timeit.default_timer()
    print "\n\n ** Elapsed time : %8.2f seconds **" % (stop - start)
