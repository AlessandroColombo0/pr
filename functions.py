from typing import Any
import re
import executing
import functools
from textwrap import dedent
import sys
import numpy as np 
import pandas as pd
import pprint
from .tema import *
import ast
import json

from os.path import basename
import inspect
import time

def rgb2console_color(rgb, background=False):
    if not background:
        return f"\x1B[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
    else:
        return f"\x1B[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"

RESET = "\033[0m"
BOLD = '\033[01m'
NO_BOLD = "\033[22m"
WHITE_FRONT = rgb2console_color([255, 255, 255])
GREY_FRONT = rgb2console_color([170, 170, 170])
DARK_GREY_FRONT = rgb2console_color([140, 140, 140])
FUNC_COLOR = rgb2console_color([220, 97, 159])
OP_COLOR = rgb2console_color([204, 86, 166])
BRACKET_COLOR = rgb2console_color([172, 240, 255])


def auto_newline(text, threshold, newline="\n", splitter=" "):
    import numpy as np
    
    if type(text) == str and len(text) > threshold:
        normal_split = text.split(splitter)
        
        # rimozione codici colore ANSI
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_text = ansi_escape.sub('', text)
        
        clean_split = clean_text.split(splitter)
        
        br_idxs = [0]  # viene nserito per permettere una continuità nel for loop ma poi viene tolto
        
        for i, word in enumerate(clean_split):
            current_split = clean_split[br_idxs[-1]:i + 1]
            if sum([len(i) + 1 for i in current_split]) - 1 > threshold and i != 0:
                br_idxs.append(i)
        
        del br_idxs[0]
        
        np_split = np.array(normal_split)
        np_split = np.insert(np_split, br_idxs, f"{newline}")
        
        return " ".join(np_split.tolist()).replace(f" {newline} ", f"{newline}")
    
    else:
        return text


def get_type_hint(obj: Any) -> str:
    """
    [[...], [...]] -> list[list]
    [["..."], ["..."]] -> list[list[str]]
    ["...", 0] -> list[Union[str, int]]
    {"...": ["..."]} -> dict[str, list[str]]   
    """

    if isinstance(obj, list):
        if not obj:
            return "list"
        inner_types = set(get_type_hint(item) for item in obj)
        if len(inner_types) == 1:
            return f"list[{inner_types.pop()}]"
        else:
            return f"list[Union[{', '.join(sorted(inner_types))}]]"
    elif isinstance(obj, dict):
        if not obj:
            return "dict"
        key_types = set(get_type_hint(key) for key in obj.keys())
        value_types = set(get_type_hint(value) for value in obj.values())
        key_type = f"Union[{', '.join(sorted(key_types))}]" if len(key_types) > 1 else key_types.pop()
        value_type = f"Union[{', '.join(sorted(value_types))}]" if len(value_types) > 1 else value_types.pop()
        return f"dict[{key_type}, {value_type}]"
    elif isinstance(obj, tuple):
        if not obj:
            return "tuple"
        inner_types = [get_type_hint(item) for item in obj]
        if len(set(inner_types)) == 1:
            return f"tuple[{inner_types[0]}, ...]"
        else:
            return f"tuple[Union[{', '.join(inner_types)}]]"
    elif isinstance(obj, set):
        if not obj:
            return "set"
        inner_types = set(get_type_hint(item) for item in obj)
        if len(inner_types) == 1:
            return f"set[{inner_types.pop()}]"
        else:
            return f"set[Union[{', '.join(sorted(inner_types))}]]"
    elif isinstance(obj, bool):
        return "bool"
    elif isinstance(obj, int):
        return "int"
    elif isinstance(obj, float):
        return "float"
    elif isinstance(obj, str):
        return "str"
    elif obj is None:
        return "None"
    else:
        return type(obj).__name__


def format_arg(arg, padding, indent):
    """ codice aggiunto per colorare il nome della variabile"""
    arg = f"{BOLD}{WHITE_FRONT}{arg} {OP_COLOR}>{RESET}"
    return add_left_padding(arg, padding, indent)


class NoSourceAvailableError(OSError):
    """
    Raised when pr fails to find or access source code that's
    required to parse and analyze. This can happen, for example, when

      - pr() is invoked inside a REPL or interactive shell, e.g. from the
        command line (CLI) or with python -i.

      - The source code is mangled and/or packaged, e.g. with a project
        freezer like PyInstaller.

    """
    infoMessage = (
        'Failed to access the underlying source code for analysis. Was pr() '
        'invoked in a REPL (e.g. from the command line), a frozen application '
        '(e.g. packaged with PyInstaller), or did the underlying source code '
        'change during execution?')


def callOrValue(obj):
    return obj() if callable(obj) else obj

class Source(executing.Source):
    def get_text_with_indentation(self, node):
        result = self.asttokens().get_text(node)
        if '\n' in result:
            result = ' ' * node.first_token.start[1] + result
            result = dedent(result)
        result = result.strip()
        return result

def prefixLinesAfterFirst(prefix, s):
    """ aggiunge " " all'inizio di tutte le righe stringhe dopo la prima"""
    lines = s.splitlines(True)
    
    for i in range(1, len(lines)):
        lines[i] = prefix + lines[i]
    
    return ''.join(lines)

def indented_lines(prefix, string):
    lines = string.splitlines()
    return [prefix[:-4] + lines[0]] + [
        # return [prefix] + lines[0]] + [
        ' ' + line for line in lines[1:]
    ]

def isLiteral(s):
    try:
        ast.literal_eval(s) 
    except Exception:
        return False
    
    return True

def singledispatch(func):
    if "singledispatch" not in dir(functools):
        def unsupport_py2(*args, **kwargs):
            raise NotImplementedError(
                "functools.singledispatch is missing in " + sys.version
            )
        
        func.register = func.unregister = unsupport_py2
        return func
    
    func = functools.singledispatch(func)
    
    # add unregister based on https://stackoverflow.com/a/25951784
    closure = dict(zip(func.register.__code__.co_freevars,
                       func.register.__closure__))
    registry = closure['registry'].cell_contents
    dispatch_cache = closure['dispatch_cache'].cell_contents
    
    def unregister(cls):
        del registry[cls]
        dispatch_cache.clear()
    
    func.unregister = unregister
    return func


@singledispatch
def argumentToString(obj):
    s = pprint.pformat(obj)
    s = s.replace('\\n', '\n')  # Preserve string newlines in output.
    return s

def get_context_str(callFrame, callNode, start_time, time_taken, padding, indent):
    lineNumber = callNode.lineno
    frameInfo = inspect.getframeinfo(callFrame)
    # parentFunction = frameInfo.function
    
    filename = basename(frameInfo.filename)
    split = filename.split(".")
    extension = split[-1]
    filename = ".".join(split[:-1])

    time_since_start = time.perf_counter() - start_time
    time_since_previous = 0 if not time_taken else time_since_start - time_taken[-1]
    lt_1 = time_since_previous < 1
    if time_since_previous:
        time_since_previous_str = f"{int(time_since_previous * 1000)}ms" if lt_1 else f"{time_since_previous:.2f}s"
    
    context = f"{WHITE_FRONT}{filename}{GREY_FRONT}.{extension}{OP_COLOR}:{WHITE_FRONT}{lineNumber}"
    timer = f"{GREY_FRONT}{time_since_start:.2f}s"
    
    time_since_previous = f"{DARK_GREY_FRONT}({time_since_previous_str})" if time_since_previous else ""
    
    prefix = f"{FUNC_COLOR}pr{BRACKET_COLOR}(){OP_COLOR}:"

    context = f"{BOLD}{prefix}{RESET} {context}  {timer} {time_since_previous}{RESET}"
    context = add_left_padding(context, padding=padding, indent=indent)

    return context, time_since_start

def num_of_zeros(n):
    if n < 1 and n != 0:
        s = '{:.16f}'.format(n).split('.')[1]
        return min(len(s) - len(s.lstrip('0')) + 1, 8)
    else:
        return 0

def args_format_msg_type(txt, wrap_width, msg_type):
    # 0: front  1: bg
    type2color = dict(warning=[[255, 230, 70], [105, 90, 35]],
                      success=[[100, 255, 100], [35, 128, 35]],
                      info=[[143, 235, 236], [43, 146, 148]],
                      error=[[255, 100, 100], [128, 35, 35]])
    
    console_c_front = rgb2console_color(type2color[msg_type][0])
    console_c_bg = rgb2console_color(type2color[msg_type][1], background=True)
    start = f"{console_c_bg}{WHITE_FRONT} {msg_type.upper()}: {RESET}{console_c_front}\n"
    
    txt = txt.replace('"', f'"{console_c_front}')
    
    if "\n" not in txt:
        txt = auto_newline(txt, threshold=wrap_width+len(console_c_front), newline=f"\n{console_c_front}")
        txt = start + txt + RESET

    else:
        lines = txt.split("\n")
        txt = "\n".join([f"{console_c_front}{i}" for i in lines])
        txt = start + txt + RESET
                    
    return txt

def get_dim1_edgeitems(obj, wrap_width, np_max_rows, precision):
    edgeitems = 10
    threshold = edgeitems * 2 + 1
    
    real_lenght_obj = np.copy(obj)
    
    if np.issubdtype(obj.dtype, np.number):
        real_lenght_obj[0] = np.max(obj)
        real_lenght_obj[1] = np.min(obj)
    str_np = np.array2string(real_lenght_obj, wrap_width, threshold=threshold, 
                            edgeitems=edgeitems, precision=precision).split("\n")
    str_np_lens = [len(i) for i in str_np]
    
    n_rows = len(str_np)
    line_width = max(str_np_lens) * n_rows  # - 3 rimuove l'effetto dei "..." di numpy
    item_width = (line_width / (edgeitems * 2)) * n_rows  # spazio compreso
    
    line_width_diff = wrap_width - line_width
    n_rows_diff = np_max_rows - n_rows
    
    if line_width_diff > 0:
        # edge items per linea:
        edgeitems = wrap_width / item_width / 2  ## 0.25 diventerà 0.5 con il x2 e viene messo perchè l'elemento dopo i ... viene rimosso
        edgeitems = int(edgeitems * 2) / 2  # approssima in questo modo: 0.3 -> 0  0.7 -> 0.5,  1.1 -> 1, lo facciamo perchè se i decimali di edgeitems sono maggiori di 0.5 vuol dire che sta sforando e questo crea uno sbilanciamento
    
    edgeitems *= ((n_rows_diff / n_rows) + 1)
    edgeitems -= 0.5  # se l'array ha i ... verrà rimosso il valore dopo i ...
    edgeitems = int(edgeitems)
    
    str_obj = np.array2string(obj, wrap_width, threshold=edgeitems, edgeitems=edgeitems, precision=precision)
    
    nrows = len(str_obj.split("\n"))
    if nrows == 11:
        array_str_len = len(str_obj) - 4
        edgeitem_len = round(array_str_len / 2 / edgeitems) - 1  # totale - "... " / 2 / edgeitems - 1 (spazio tra i numeri in numpy array)
        
        if edgeitem_len >= 3:
            dots = "..." + " " * (edgeitem_len - 3)
        else:
            dots = "..." + " " * (3 - edgeitem_len)
        
        str_obj = str_obj.replace(f" ... ", f" {dots} ")
        
        str_obj_s = str_obj.split("\n")
        first_line_len = len(str_obj_s[0])
        middle_line_len = len(str_obj_s[5])
        
        if first_line_len < middle_line_len:
            str_obj_s = str_obj.split(dots)
            str_obj = str_obj_s[0] + dots + str_obj_s[1][int(item_width):]
    
    return str_obj


def element_wrap(wrap_width, obj, max_list_newlines=None, truncation=False, key=None):
    # se ha una key viene trattato come un dizionario
    obj_type = type(obj)
    
    text = str(obj)
    
    string_split = text.split(",")
    str_split_lengths = [len(i) + 2 for i in string_split]
    newline_idxs = []
    new_str = ""
    newline_i = 0
    n_line_padding = 2
    last_split = 0
    
    if "\n" not in text:
        for i in range(len(string_split)):
            if sum(str_split_lengths[i] for i in range(last_split, i + 1)) + n_line_padding > wrap_width:
                last_split = i
                newline_idxs.append(last_split)
                n_line_padding = 1
    
    more_newlines = True if len(newline_idxs) > 0 else False  # nel prossimo loop segnala se ci saranno altre newlines o meno
    
    truncation = True if truncation and len(newline_idxs) >= max_list_newlines*2 + 1 else False
    hide = False
    
    for i, split_, in enumerate(string_split):
        
        if truncation and i == newline_idxs[max_list_newlines-1]:
            hide = True
            new_str += "\n ..."
            
        elif truncation and i == newline_idxs[-max_list_newlines]:
            hide = False
        
        if more_newlines and i == newline_idxs[newline_i]:  # se siamo ad un index newline
            
            if not hide:        
                if not key:  # non dizionario
                    new_str += "\n" + " " + split_ + ", "
                else:  # dizionario
                    new_str += "\n" + " " * (len(str(key)) + 6) + split_ + " "
            
            newline_i += 1
            if newline_i >= len(newline_idxs):
                more_newlines = False

        elif not hide:  # caso in cui non è l'i di una newline e non è nascosto
            new_str += split_ + ", "
            
    new_str = new_str[:-2]
    
    if obj_type == str:
        new_str = new_str.replace("\n", "'\n'")
        return f"'{new_str}'"
    else:
        return new_str

# MARK: format obj
def print_formatted_arg(arg, padding, indent):
    print(format_arg(arg, padding, indent))  # OK
    return ""

def get_single_line_output(arg, val, padding, indent):
    out = f"{format_arg(arg, padding, indent)} {colorize(str(val))}"
    print(out)  # OK
    return ""

def print_formatted_info(arg, padding, indent, *args, **kwargs):
    
    info = ""
    for a in args:
        info += f"{colorize(str(a))}, "
    for key, value in kwargs.items():
        info += f"{BRACKET_COLOR}{key}{OP_COLOR}:{RESET} {colorize(str(value))}, "
        
    info = info[:-1]  # toglie la ,
        
    print(f"{format_arg(arg, padding, indent)} {info}")  # OK


def format_dict(obj, arg, wrap_width, min_padding, dict_max_keys, parse_as_json, truncation, padding, indent):
    kwargs = dict(type=get_type_hint(obj), len=len(obj))
    print_formatted_info(arg, padding, indent, **kwargs)
    
    if parse_as_json:
        return json.dumps(obj, indent=4)
        
    new_str = "{"
    n_keys = len(obj)
        
    truncate_keys = n_keys > dict_max_keys and truncation 
        
    if truncate_keys:
        keys = list(obj.keys())[:dict_max_keys - 1]
        obj = {k: obj[k] for k in keys}
    
    for key, value in obj.items():
        dict_wrap_width = wrap_width - (len(str(key)) + 5) - min_padding
        new_value = element_wrap(dict_wrap_width, value, key=key)
        
        q = '"' if type(key) == str else ''
        new_str += f'{q}{key}{q}: {new_value},\n '
    
    if truncate_keys:
        new_str += f"... (other {n_keys - dict_max_keys + 1} keys), \n"
        
    return new_str[:-3] + "}"


def format_array(obj, arg, wrap_width, np_max_rows, min_padding, truncation, padding, indent):
    kwargs = dict()
    precision = None
                    
    np.set_printoptions(suppress=True)
    dims = len(obj.shape)
    
    if dims > 0 and obj.size > 0:
        if np.issubdtype(obj.dtype, np.number):  # controlla se è numerico
            if np.issubdtype(obj.dtype, np.integer):
                precision = 0  # era 1
            else:
                precision = np.absolute(np.nanmedian(obj.flatten(), axis=-1))
                
                max_num_zeros = num_of_zeros(precision)
                precision = max_num_zeros + 2  # se abbiamo 0.01 (max_num_zeros: 1) precision sarà 3, quindi vedremo i 3 decimali dopo il punto
            
            if "int" not in str(obj.dtype):
                kwargs = dict(
                    decimals=precision, mean=f"{np.mean(obj):.{precision}f}",
                    median=f"{np.median(obj):.{precision}f}", std=f"{np.std(obj):.{precision}f}"
                    )
    
    args = [str(type(obj)).replace("'", ''), obj.dtype, obj.shape]
    print_formatted_info(arg, padding, indent, *args, **kwargs)
    
    if truncation == True:
        if obj.size == 0:
            str_obj = str(obj)
        elif dims == 1:
            if obj.shape[0] != 1:
                str_obj = get_dim1_edgeitems(obj, wrap_width, np_max_rows, precision)  # va bene com'è
            else:
                str_obj = str(obj)
        elif dims == 2:
            edgeitems = 4 if wrap_width > 99 else 3
            # precision
            str_obj = np.array2string(obj, 1000, threshold=edgeitems, edgeitems=edgeitems, precision=precision)
        else:  # dims >= 3
            edgeitems = 2
            str_obj = np.array2string(obj, 1000, threshold=edgeitems, edgeitems=edgeitems, precision=precision)
    
    else:
        np.set_printoptions(suppress=True, threshold=sys.maxsize)
        str_obj = np.array2string(obj, threshold=sys.maxsize, max_line_width=wrap_width - min_padding, precision=precision)
    
    return str_obj

def format_dataframe(obj, arg, def_pd_max_rows, pd_max_col_width, pd_max_rows, pd_line_width, truncation, padding, indent):
    args = [str(type(obj)).replace("'", ''), 
            obj.shape]
    print_formatted_info(arg, padding, indent, *args)
    
    pd_max_rows = pd_max_rows or def_pd_max_rows
    
    to_string_kwargs = dict()
    if pd_max_rows and truncation:
        to_string_kwargs["max_rows"] = pd_max_rows
        
    if pd_max_col_width:
        to_string_kwargs["max_colwidth"] = pd_max_col_width
        
    # Series non supporta line_width
    try:
        str_obj = obj.to_string(**to_string_kwargs, line_width=pd_line_width)
    except:
        str_obj = obj.to_string(**to_string_kwargs)
        

    return str_obj

def format_list(obj, arg, wrap_width, list_max_rows, min_padding, table_columns, max_list_newlines, parse_as_json, 
                truncation, padding, indent):

    info_kwargs = dict(type=get_type_hint(obj), len=len(obj))
    print_formatted_info(arg, padding, indent, **info_kwargs)
        
    if parse_as_json:
        return json.dumps(obj, indent=4)
        
    new_str = ""
    obj_type = type(obj).__name__
    brackets = {"list": ["[", "]"], "set": ["{", "}",], "tuple": ["(", ")"]}

    if type(obj) == set:  # soluzione temporanea per printare sets 
        obj = list(obj)

    if not obj:
        return str(obj)

    elif table_columns:
        try:
            # list of dicts
            if all(type(i) == dict for i in obj):
                pd.options.display.max_rows = sys.maxsize
                new_str = "\n" + str(pd.DataFrame(obj))
                
            # list
            else:
                pd.options.display.max_rows = sys.maxsize
                new_str = "\n" + str(pd.DataFrame(obj, columns=table_columns))
            
        except Exception as exc:
            return {"exc": exc}  
        
    elif len(str(obj[0])) > int(wrap_width / 2):
        bo = brackets[obj_type]
        
        txt_func = lambda x: auto_newline(str(x), wrap_width) if type(x) != str else \
            f'"{auto_newline(str(x), wrap_width).replace('"', '\\"')}"'
        
        if len(obj) > list_max_rows and truncation:
            obj_list = [txt_func(i) for i in obj[:4]] + ["..."] + [txt_func(i) for i in obj[-4:]]
        else:
            obj_list = [txt_func(i) for i in obj]
        
        new_str += bo[0] + ",\n ".join(obj_list) + bo[1]

    else:
        new_str += element_wrap(wrap_width, obj, max_list_newlines, truncation) 
            
    return new_str


def format_string(obj, arg, wrap_width, padding, indent):
    obj = obj.replace("\n", "\\n")
    string = auto_newline(obj, wrap_width, newline="'\n'")
    string = f"'{string}'"
    
    if f"'{obj}'" != string:
        print_formatted_arg(arg, padding, indent)
        return string
    else:
        return get_single_line_output(arg, string, padding, indent) 


def add_left_padding(string, padding=0, indent=0):
    padding = " " * padding if padding else ""
    whitespaces = padding + " " * 3 * indent
    
    if "\n" in string:
        split = string.split("\n")
        split[0] = padding + split[0]
        return ("\n" + whitespaces).join(split)
    else:
        return whitespaces + string
