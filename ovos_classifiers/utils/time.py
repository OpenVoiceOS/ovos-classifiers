from datetime import datetime
from dateutil.tz import gettz, tzlocal
from ovos_utils.time import DAYS_IN_1_YEAR, DAYS_IN_1_MONTH, is_leap_year, get_next_leap_year, now_utc, now_local, to_utc, to_local, to_system


__default_tz = None


def set_default_tz(tz):
    global __default_tz
    if isinstance(tz, str):
        tz = gettz(tz)
    __default_tz = tz


def default_timezone():
    """ Get the default timezone

    either a value set by downstream user with
    lingua_franca.internal.set_default_tz
    or default system value

    Returns:
        (datetime.tzinfo): Definition of the default timezone
    """
    return __default_tz or tzlocal()

