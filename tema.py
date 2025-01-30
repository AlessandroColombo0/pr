from pygments.style import Style
from pygments.token import (
    Text, Name, Error, Other, String, Number, Keyword, Generic, Literal,
    Comment, Operator, Whitespace, Punctuation)




import colorama
from contextlib import contextmanager

RESET = "\033[0m"

from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import PythonLexer as PyLexer, Python3Lexer as Py3Lexer
import sys 

class TemaAlessandro(Style):
    BASE01 = '#4f7782'  # noqa
    BASE0 = '#dadada'  # noqa
    BASE1 = '#dadada'  # noqa
    YELLOW = '#b58900'  # noqa
    ORANGE = '#cb4b16'  # noqa
    RED = '#dc322f'  # noqa
    class_color = "#E9A6FF"
    MAGENTA = '#d33682'  # noqa
    blue = '#51a6e0'  # noqa
    light_blue = "#c3e3de"
    light_magenta = "#a98fc2"
    
    styles = {
        Text: BASE0,
        # Whitespace:             BASE03,
        Error: RED,
        Other: BASE0,
        
        Name: BASE1,
        Name.Attribute: BASE0,
        Name.Builtin: blue,
        Name.Builtin.Pseudo: blue,
        Name.Class: blue,
        Name.Constant: YELLOW,
        Name.Decorator: ORANGE,
        Name.Entity: ORANGE,
        Name.Exception: ORANGE,
        Name.Function: blue,
        Name.Property: blue,
        Name.Label: BASE0,
        Name.Namespace: YELLOW,
        Name.Other: BASE0,
        Name.Tag: YELLOW,
        Name.Variable: ORANGE,
        Name.Variable.Class: blue,
        Name.Variable.Global: blue,
        Name.Variable.Instance: blue,
        
        String: light_blue,
        String.Backtick: light_blue,
        String.Char: MAGENTA,
        String.Doc: light_blue,
        String.Double: light_blue,
        String.Escape: ORANGE,
        String.Heredoc: light_blue,
        String.Interpol: ORANGE,
        String.Other: MAGENTA,
        String.Regex: light_blue,
        String.Single: light_blue,
        String.Symbol: MAGENTA,
        
        Number: BASE01,
        Number.Float: BASE01,
        Number.Hex: BASE01,
        Number.Integer: BASE01,
        Number.Integer.Long: BASE01,
        Number.Oct: BASE01,
        
        Keyword: class_color,
        Keyword.Constant: class_color,
        Keyword.Declaration: class_color,
        Keyword.Namespace: class_color,
        Keyword.Pseudo: class_color,
        Keyword.Reserved: class_color,
        Keyword.Type: class_color,
        
        Generic: BASE0,
        Generic.Deleted: BASE0,
        Generic.Emph: BASE0,
        Generic.Error: BASE0,
        Generic.Heading: BASE0,
        Generic.Inserted: BASE0,
        Generic.Output: BASE0,
        Generic.Prompt: BASE0,
        Generic.Strong: BASE0,
        Generic.Subheading: BASE0,
        Generic.Traceback: BASE0,
        
        Literal: BASE0,
        Literal.Date: BASE0,
        
        Comment: BASE01,
        Comment.Multiline: BASE01,
        Comment.Preproc: BASE01,
        Comment.Single: BASE01,
        Comment.Special: BASE01,
        
        Operator: "#d17caa",  # < > -
        Operator.Word: BASE0,  # "in"
        Punctuation: "#a63b7b",  # , [ { : ...
    }


def bindStaticVariable(name, value):
    def decorator(fn):
        setattr(fn, name, value)
        return fn
    
    return decorator

# @bindStaticVariable('formatter', Terminal256Formatter(style=SolarizedDark))
@bindStaticVariable('formatter', Terminal256Formatter(style=TemaAlessandro))
@bindStaticVariable(
    'lexer', PyLexer(ensurenl=False) if False else Py3Lexer(ensurenl=False))
def colorize(s):
    self = colorize
    return highlight(s, self.lexer, self.formatter)

@contextmanager
def supportTerminalColorsInWindows():
    # Filter and replace ANSI escape sequences on Windows with equivalent Win32
    # API calls. This code does nothing on non-Windows systems.
    colorama.init()
    yield
    colorama.deinit()

def stderrPrint(*args):
    print(*args, file=sys.stderr)

def colorizedStderrPrint(s):
    colored = colorize(s)
    with supportTerminalColorsInWindows():
        stderrPrint(colored)
