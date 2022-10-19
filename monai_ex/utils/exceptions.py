from typing import Optional, Union, Sequence, Type
import re
import sys
import logging
import traceback
import functools
from termcolor import colored


__all__ = ["GenericException", "TransformException", "WorkflowException", "DatasetException", "catch_exception", "trycatch"]


class GenericException(Exception):
    """ Custom exception class for generic Strix exceptions."""
    pass


class TransformException(GenericException):
    """ Custom exception class for Strix exceptions of transforms."""
    pass


class DatasetException(GenericException):
    """ Custom exception class for Strix exceptions of transforms."""
    pass


class WorkflowException(GenericException):
    """ Custom exception class for Strix exceptions in workflow."""
    pass


def catch_exception(
    handled_exception_type: Type[Exception],
    logger_name: Optional[str] = None,
    show_args: bool = True,
    show_details: bool = False,
    return_none: bool = False,
    path_keywords: Union[Sequence[str], str] = [],
):
    """Try catch exception function, help to better show error msg.

    Args:
        handled_exception_type (Exception): own exception type that have handled.
        logger_name (Option[str], optional): logger name. Defaults to None.
        show_args (bool, optional): whether to show the input args of error func. Defaults to True.
        show_details (bool, optional): wheter to show the original exception msg. Defaults to False.
        return_none (bool, optional): if true return None, otherwise exit. Defaults to False.
        path_keywords (Union[str, Sequence[str]]): keywords of own code path, help to locate key func.
    """
    file_regex = r"(\/.*?\.[\w:]+)"

    def _get_related_error_msg(tb):
        error_msg = None
        keywords = path_keywords if isinstance(path_keywords, (tuple, list)) else [path_keywords, ]
        for line in reversed(traceback.format_tb(tb)):
            file_path = re.search(file_regex, line).group(1).lower()
            if any([keyword.lower() in file_path for keyword in keywords]):
                error_msg = line
                break
        return error_msg

    def wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except handled_exception_type as e:
                log_info = \
                  f"\n{'! '*30}\n{colored('Error occurred! Please check your code! Msg:', color='red')} {e}\n\n"
                while hasattr(e, "__cause__") and e.__cause__:
                    log_info += f"- {''.join(traceback.format_tb(e.__traceback__, limit=-2)[0])}"
                    log_info += f"→ {e.__cause__}\n\n"
                    e = e.__cause__
                log_info += f"{'! '*30}\n"

                if logger_name is None:
                    print(log_info)
                else:
                    logger = logging.getLogger(logger_name)
                    logger.exception(log_info)

                if return_none:
                    return
                else:
                    sys.exit(-1)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print(colored("Exception trace:", color="red"))
                print(colored("Exception entry is:", color='yellow'))
                print(" ".join(traceback.format_tb(exc_tb, limit=2)[-1:]))
                if show_args:
                    print(colored("    ↑ Input arguments:", color='cyan'), args, kwargs)

                if path_keywords:
                    error_msg = _get_related_error_msg(exc_tb)
                else:
                    error_msg = " ".join(traceback.format_tb(exc_tb, limit=-1))

                print(colored("Exception occured at:", color='yellow'))
                print(error_msg)
                print(traceback.format_exc().splitlines()[-1], '\n')

                if show_details:
                    print(traceback.format_exc())

                if return_none:
                    return
                else:
                    sys.exit(-1)

        return inner_wrapper

    return wrapper


trycatch = functools.partial(catch_exception, handled_exception_type=GenericException, path_keywords=['strix', 'monai_ex'])
