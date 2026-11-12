"""Domain errors for reliability life analysis.

Failures that must be reported to the caller (analytic or diagnostic) use
``Failure``. They are deliberately separate from ``VaultError``: a failed fit is
a result to be recorded, not an invalid request.
"""


class Failure(Exception):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code = code
        self.details = details

    def as_dict(self):
        return {"code": self.code, "message": str(self), "details": self.details}
