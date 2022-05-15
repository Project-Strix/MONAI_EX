
__all__ = ["GenericException", "TransformException", "WorkflowException"]

class  GenericException(Exception):
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