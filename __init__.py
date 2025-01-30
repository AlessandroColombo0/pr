from __future__ import print_function

# pip install numpy pandas colorama pygments executing regex asttokens

import time

START_TIME = time.perf_counter()

from .functions import *
    
# SETUP

import numpy as np
import pandas as pd
import traceback
import sys
from datetime import datetime

""" MIN PADDING:
pr(): WRK_model.py:42 ' 6.769 '
     ' x ': ...
|__________| = 12 chars
"""
MIN_PADDING = 2
LEFT_PADDING = 2 

pd.options.display.max_columns = None
pd.options.display.width = None
DEF_PD_MAX_ROWS = 7
pd.options.display.max_rows = DEF_PD_MAX_ROWS

NP_MAX_ROWS = 7
LIST_MAX_ROWS = 9
MAX_LIST_NEWLINES = 3 # 3 linee sopra e 3 sotto
DICT_MAX_KEYS = 20

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

time_taken = []


_absent = object()

DEFAULT_CONTEXT_DELIMITER = '- '

# MARK: class Pr 
class Pr:
    _pairDelimiter = ', '  # Used by the tests in tests/.
    contextDelimiter = DEFAULT_CONTEXT_DELIMITER
    
    def __init__(self, wrap_width=85):
        self.enabled = True
        self.wrap_width = wrap_width
        self.compact_output = False 
    
    def __call__(self, *args, skip=False, msg_type=None, truncation=True, pd_max_col_width=None, round_num=0, 
                 table_columns=None, indent=0, parse_as_json=False, compact=False, pd_max_rows=None, pd_line_width=None):
        """ 
            TABLE COLUMNS
                a = [("a", 1), ("b", 2), ("c", 3)]
                pr(a, table_columns=["letter", "number"])
                
                ' a ':
                    letter  number
                0      a       1
                1      b       2
                2      c       3
                
            pd_line_width: lunghezza datframe intero, usare -1 per non far mai andare a capo (lunghezza infinita)
        """
        if skip:
            return
        
        if not self.enabled:
            return
        
        pd_line_width = pd_line_width or self.wrap_width
        pd_line_width = 9*10**10 if pd_line_width <= -1 else pd_line_width
                
        callFrame = inspect.currentframe().f_back
        callNode = Source.executing(callFrame).node
        if callNode is None:
            raise NoSourceAvailableError()
        
        compact = self.compact_output or compact
        
        if not compact:
            context, time = get_context_str(callFrame, callNode, START_TIME, time_taken, padding=0, indent=indent)
            time_taken.append(time)
            print(context)  # OK

        left_padding = LEFT_PADDING if not compact else 0
        indent_kwargs = dict(padding=left_padding, indent=indent)
        
        # MARK: msg_type
        if msg_type:
            txt = "\n".join([str(i) for i in args]) if len(args) > 1 else args[0]
            
            lines = args_format_msg_type(txt, self.wrap_width, msg_type)
            lines = add_left_padding(lines, **indent_kwargs)
            # lines = "\n".join([line_format(l, indent) for l in lines.split("\n")])
            print(lines)  # OK
            return
            
        elif not args:
            return None
            
        # MARK: regular pr()
        source = Source.for_frame(callFrame)
        sanitizedArgStrs = [source.get_text_with_indentation(arg)
                            for arg in callNode.args]  # ottiene il nome della variabile
        
        pairs = list(zip(sanitizedArgStrs, args))
        
        for arg, obj in pairs:
            arg = "_" if isLiteral(arg) else arg

            try:
                # DICT
                if type(obj) == dict:
                    new_str = format_dict(obj, arg, self.wrap_width, MIN_PADDING, DICT_MAX_KEYS, 
                                            parse_as_json, truncation, **indent_kwargs)
                    
                # ARRAY
                elif isinstance(obj, np.ndarray):
                    new_str = format_array(obj, arg, self.wrap_width, NP_MAX_ROWS, MIN_PADDING, 
                                            truncation, **indent_kwargs)
                
                # TENSOR
                elif "tensor" in str(type(obj)) and truncation == False:
                    np.set_printoptions(suppress=True, threshold=sys.maxsize)
                    new_str = str(obj)
                
                # DATAFRAME
                elif isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
                    new_str = format_dataframe(obj, arg, DEF_PD_MAX_ROWS, pd_max_col_width, pd_max_rows, 
                                               pd_line_width, truncation, **indent_kwargs)
                
                # LIST
                elif type(obj) in [list, set, tuple]:
                    new_str = format_list(obj, arg, self.wrap_width, LIST_MAX_ROWS, MIN_PADDING, table_columns, MAX_LIST_NEWLINES, parse_as_json, truncation, **indent_kwargs)
                    
                    if type(new_str) == dict() and "exc" in new_str:
                        pr(f"Error while printing object with parameter table_columns={table_columns},\n" \
                            f"make sure the object is formatted correctly (table structure).\nException: {exc}", 
                            msg_type="error")   
                    
                # STR                
                elif type(obj) == str:
                    new_str = format_string(obj, arg, self.wrap_width, **indent_kwargs)
                
                # BUILT IN OBJ
                elif type(obj) in [float, int, bool, type(None)]:
                    if type(obj) == float and round_num:
                        obj = round(obj, round_num)

                    new_str = get_single_line_output(arg, obj, **indent_kwargs)
                    
                # OTHER
                else:
                    print_formatted_info(arg, type=type(obj), **indent_kwargs)
                    new_str = element_wrap(self.wrap_width, obj)

                if new_str:
                    new_str = add_left_padding(new_str, **indent_kwargs)
                
                if new_str:
                    colorizedStderrPrint(new_str)
                
            except Exception as exc:
                pr(f"pr() error with arg {BOLD}{arg}{RESET}", traceback.format_exc(), msg_type="error")
                
       
    def _formatTime(self):
        now = datetime.now()
        formatted = now.strftime('%H:%M:%S.%f')[:-3]
        return ' at %s' % formatted
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False
    
    def configure(self, wrap_width=_absent):
        noParameterProvided = all(
            v is _absent for k, v in locals().items() if k != 'self'
            )
        
        if noParameterProvided: 
            raise TypeError('configure() missing at least one argument')
        
        if wrap_width is not _absent:
            self.wrap_width = wrap_width
            np.set_printoptions(linewidth=wrap_width, threshold=6)
            pd.options.display.width = wrap_width
        
    def compact_toggle(self):
        self.compact_output = False if self.compact_output else True 

pr = Pr()






