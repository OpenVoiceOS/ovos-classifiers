from datetime import datetime, timedelta
from dateutil.tz import gettz, tzlocal
from dateutil.relativedelta import relativedelta
from enum import Enum   
from typing import Tuple, Optional

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



class DurationResolution(Enum):
    TIMEDELTA = 0
    RELATIVEDELTA = 1
    RELATIVEDELTA_STRICT = 1
    RELATIVEDELTA_FALLBACK = 2
    RELATIVEDELTA_APPROXIMATE = 3
    TOTAL_SECONDS = 4
    TOTAL_MICROSECONDS = 5
    TOTAL_MILLISECONDS = 6
    TOTAL_MINUTES = 7
    TOTAL_HOURS = 8
    TOTAL_DAYS = 9
    TOTAL_WEEKS = 10
    TOTAL_MONTHS = 11
    TOTAL_YEARS = 12
    TOTAL_DECADES = 13
    TOTAL_CENTURIES = 14
    TOTAL_MILLENNIUMS = 15


class DateTimeResolution(Enum):
    # absolute units
    MICROSECOND = 0
    MILLISECOND = 1
    SECOND = 2
    MINUTE = 3
    HOUR = 4

    DAY = 5
    WEEKEND = 6
    WEEK = 7
    MONTH = 8
    YEAR = 9
    DECADE = 10
    CENTURY = 11
    MILLENNIUM = 12

    SEASON = 13
    SPRING = 14
    FALL = 15
    WINTER = 16
    SUMMER = 17

    # {unit} of {resolution}
    MICROSECOND_OF_MILLISECOND = 18
    MICROSECOND_OF_SECOND = 19
    MICROSECOND_OF_MINUTE = 20
    MICROSECOND_OF_HOUR = 21
    MICROSECOND_OF_DAY = 22
    MICROSECOND_OF_WEEKEND = 23
    MICROSECOND_OF_WEEK = 24
    MICROSECOND_OF_MONTH = 25
    MICROSECOND_OF_YEAR = 26
    MICROSECOND_OF_DECADE = 27
    MICROSECOND_OF_CENTURY = 28
    MICROSECOND_OF_MILLENNIUM = 29

    MICROSECOND_OF_SEASON = 30
    MICROSECOND_OF_SPRING = 31
    MICROSECOND_OF_FALL = 32
    MICROSECOND_OF_WINTER = 33
    MICROSECOND_OF_SUMMER = 34

    MILLISECOND_OF_SECOND = 35
    MILLISECOND_OF_MINUTE = 36
    MILLISECOND_OF_HOUR = 37
    MILLISECOND_OF_DAY = 38
    MILLISECOND_OF_WEEKEND = 39
    MILLISECOND_OF_WEEK = 40
    MILLISECOND_OF_MONTH = 41
    MILLISECOND_OF_YEAR = 42
    MILLISECOND_OF_DECADE = 43
    MILLISECOND_OF_CENTURY = 44
    MILLISECOND_OF_MILLENNIUM = 45

    MILLISECOND_OF_SEASON = 46
    MILLISECOND_OF_SPRING = 47
    MILLISECOND_OF_FALL = 48
    MILLISECOND_OF_WINTER = 49
    MILLISECOND_OF_SUMMER = 50

    SECOND_OF_MINUTE = 51
    SECOND_OF_HOUR = 52
    SECOND_OF_DAY = 53
    SECOND_OF_WEEKEND = 54
    SECOND_OF_WEEK = 55
    SECOND_OF_MONTH = 56
    SECOND_OF_YEAR = 57
    SECOND_OF_DECADE = 58
    SECOND_OF_CENTURY = 59
    SECOND_OF_MILLENNIUM = 60

    SECOND_OF_SEASON = 61
    SECOND_OF_SPRING = 62
    SECOND_OF_FALL = 63
    SECOND_OF_WINTER = 64
    SECOND_OF_SUMMER = 65

    MINUTE_OF_HOUR = 66
    MINUTE_OF_DAY = 67
    MINUTE_OF_WEEKEND = 68
    MINUTE_OF_WEEK = 69
    MINUTE_OF_MONTH = 70
    MINUTE_OF_YEAR = 71
    MINUTE_OF_DECADE = 72
    MINUTE_OF_CENTURY = 73
    MINUTE_OF_MILLENNIUM = 74

    MINUTE_OF_SEASON = 75
    MINUTE_OF_SPRING = 76
    MINUTE_OF_FALL = 77
    MINUTE_OF_WINTER = 78
    MINUTE_OF_SUMMER = 79

    HOUR_OF_DAY = 80
    HOUR_OF_WEEKEND = 81
    HOUR_OF_WEEK = 82
    HOUR_OF_MONTH = 83
    HOUR_OF_YEAR = 84
    HOUR_OF_DECADE = 85
    HOUR_OF_CENTURY = 86
    HOUR_OF_MILLENNIUM = 87

    HOUR_OF_SEASON = 88
    HOUR_OF_SPRING = 89
    HOUR_OF_FALL = 90
    HOUR_OF_WINTER = 91
    HOUR_OF_SUMMER = 92

    DAY_OF_WEEKEND = 93
    DAY_OF_WEEK = 94
    DAY_OF_MONTH = 95
    DAY_OF_YEAR = 96
    DAY_OF_DECADE = 97
    DAY_OF_CENTURY = 98
    DAY_OF_MILLENNIUM = 99

    DAY_OF_SEASON = 100
    DAY_OF_SPRING = 101
    DAY_OF_FALL = 102
    DAY_OF_WINTER = 103
    DAY_OF_SUMMER = 104

    WEEKEND_OF_MONTH = 105
    WEEKEND_OF_YEAR = 106
    WEEKEND_OF_DECADE = 107
    WEEKEND_OF_CENTURY = 108
    WEEKEND_OF_MILLENNIUM = 109

    WEEKEND_OF_SEASON = 110
    WEEKEND_OF_SPRING = 111
    WEEKEND_OF_FALL = 112
    WEEKEND_OF_WINTER = 113
    WEEKEND_OF_SUMMER = 114

    WEEK_OF_MONTH = 115
    WEEK_OF_YEAR = 116
    WEEK_OF_CENTURY = 117
    WEEK_OF_DECADE = 118
    WEEK_OF_MILLENNIUM = 119

    WEEK_OF_SEASON = 120
    WEEK_OF_SPRING = 121
    WEEK_OF_FALL = 122
    WEEK_OF_WINTER = 123
    WEEK_OF_SUMMER = 124

    MONTH_OF_YEAR = 125
    MONTH_OF_DECADE = 126
    MONTH_OF_CENTURY = 127
    MONTH_OF_MILLENNIUM = 128

    MONTH_OF_SEASON = 129
    MONTH_OF_SPRING = 130
    MONTH_OF_FALL = 131
    MONTH_OF_WINTER = 132
    MONTH_OF_SUMMER = 133

    YEAR_OF_DECADE = 134
    YEAR_OF_CENTURY = 135
    YEAR_OF_MILLENNIUM = 136

    DECADE_OF_CENTURY = 137
    DECADE_OF_MILLENNIUM = 138

    CENTURY_OF_MILLENNIUM = 139

    SEASON_OF_YEAR = 140
    SEASON_OF_DECADE = 141
    SEASON_OF_CENTURY = 142
    SEASON_OF_MILLENNIUM = 143

    SPRING_OF_YEAR = 144
    SPRING_OF_DECADE = 145
    SPRING_OF_CENTURY = 146
    SPRING_OF_MILLENNIUM = 147

    FALL_OF_YEAR = 148
    FALL_OF_DECADE = 149
    FALL_OF_CENTURY = 150
    FALL_OF_MILLENNIUM = 151

    WINTER_OF_YEAR = 152
    WINTER_OF_DECADE = 153
    WINTER_OF_CENTURY = 154
    WINTER_OF_MILLENNIUM = 155

    SUMMER_OF_YEAR = 156
    SUMMER_OF_DECADE = 157
    SUMMER_OF_CENTURY = 158
    SUMMER_OF_MILLENNIUM = 159

    # Special reference dates
    # number of days since 1 January 4713 BC, 12:00:00 (UTC).
    JULIAN = 668
    JULIAN_MICROSECOND = 160
    JULIAN_MILLISECOND = 161
    JULIAN_SECOND = 162
    JULIAN_MINUTE = 163
    JULIAN_HOUR = 164
    JULIAN_DAY = 165
    JULIAN_WEEK = 166
    JULIAN_WEEKEND = 167
    JULIAN_MONTH = 168
    JULIAN_YEAR = 169
    JULIAN_DECADE = 170
    JULIAN_CENTURY = 171
    JULIAN_MILLENNIUM = 172

    JULIAN_SEASON = 173
    JULIAN_SPRING = 174
    JULIAN_FALL = 175
    JULIAN_WINTER = 176
    JULIAN_SUMMER = 177

    # Julian day corrected for differences  in the Earth's position with
    # respect to the Sun.
    HELIOCENTRIC_JULIAN_MICROSECOND = 178
    HELIOCENTRIC_JULIAN_MILLISECOND = 179
    HELIOCENTRIC_JULIAN_SECOND = 180
    HELIOCENTRIC_JULIAN_MINUTE = 181
    HELIOCENTRIC_JULIAN_HOUR = 182
    HELIOCENTRIC_JULIAN_DAY = 183
    HELIOCENTRIC_JULIAN_WEEK = 184
    HELIOCENTRIC_JULIAN_WEEKEND = 185
    HELIOCENTRIC_JULIAN_MONTH = 186
    HELIOCENTRIC_JULIAN_YEAR = 187
    HELIOCENTRIC_JULIAN_DECADE = 188
    HELIOCENTRIC_JULIAN_CENTURY = 189
    HELIOCENTRIC_JULIAN_MILLENNIUM = 190

    HELIOCENTRIC_JULIAN_SEASON = 191
    HELIOCENTRIC_JULIAN_SPRING = 192
    HELIOCENTRIC_JULIAN_FALL = 193
    HELIOCENTRIC_JULIAN_WINTER = 194
    HELIOCENTRIC_JULIAN_SUMMER = 195

    # Julian day corrected for differences in the Earth's position with
    # respect to the barycentre of the Solar System.
    BARYCENTRIC__JULIAN_MICROSECOND = 196
    BARYCENTRIC__JULIAN_MILLISECOND = 197
    BARYCENTRIC__JULIAN_SECOND = 198
    BARYCENTRIC__JULIAN_MINUTE = 199
    BARYCENTRIC__JULIAN_HOUR = 200
    BARYCENTRIC_JULIAN_DAY = 201
    BARYCENTRIC_JULIAN_WEEK = 202
    BARYCENTRIC_JULIAN_WEEKEND = 203
    BARYCENTRIC_JULIAN_MONTH = 204
    BARYCENTRIC_JULIAN_YEAR = 205
    BARYCENTRIC_JULIAN_DECADE = 206
    BARYCENTRIC_JULIAN_CENTURY = 207
    BARYCENTRIC_JULIAN_MILLENNIUM = 208

    BARYCENTRIC_JULIAN_SEASON = 209
    BARYCENTRIC_JULIAN_SPRING = 210
    BARYCENTRIC_JULIAN_FALL = 211
    BARYCENTRIC_JULIAN_WINTER = 212
    BARYCENTRIC_JULIAN_SUMMER = 213

    # Unix time, number of seconds elapsed since 1 January 1970, 00:00:00 (
    # UTC).
    UNIX = 667
    UNIX_MICROSECOND = 214
    UNIX_MILLISECOND = 215
    UNIX_SECOND = 216
    UNIX_MINUTE = 217
    UNIX_HOUR = 218
    UNIX_DAY = 219
    UNIX_WEEK = 220
    UNIX_WEEKEND = 221
    UNIX_MONTH = 222
    UNIX_YEAR = 223
    UNIX_DECADE = 224
    UNIX_CENTURY = 225
    UNIX_MILLENNIUM = 226

    UNIX_SEASON = 227
    UNIX_SPRING = 228
    UNIX_FALL = 229
    UNIX_WINTER = 230
    UNIX_SUMMER = 231

    # Lilian date, number of days elapsed since the beginning of
    # the Gregorian Calendar on 15 October 1582.
    LILIAN = 669
    LILIAN_MICROSECOND = 232
    LILIAN_MILLISECOND = 233
    LILIAN_SECOND = 234
    LILIAN_MINUTE = 235
    LILIAN_HOUR = 236
    LILIAN_DAY = 237
    LILIAN_WEEK = 238
    LILIAN_WEEKEND = 239
    LILIAN_MONTH = 240
    LILIAN_YEAR = 241
    LILIAN_DECADE = 242
    LILIAN_CENTURY = 243
    LILIAN_MILLENNIUM = 244

    LILIAN_SEASON = 245
    LILIAN_SPRING = 246
    LILIAN_FALL = 247
    LILIAN_WINTER = 248
    LILIAN_SUMMER = 249

    # Holocene/Human Era s a year numbering system that adds exactly
    # 10,000 years to the currently dominant (AD/BC or CE/BCE) numbering scheme,
    # placing its first year near the beginning of the Holocene geological
    # epoch and the Neolithic Revolution
    HOLOCENE = 700
    HOLOCENE_MICROSECOND = 250
    HOLOCENE_MILLISECOND = 251
    HOLOCENE_SECOND = 252
    HOLOCENE_MINUTE = 253
    HOLOCENE_HOUR = 254
    HOLOCENE_DAY = 255
    HOLOCENE_WEEK = 256
    HOLOCENE_WEEKEND = 257
    HOLOCENE_MONTH = 258
    HOLOCENE_YEAR = 259
    HOLOCENE_DECADE = 260
    HOLOCENE_CENTURY = 261
    HOLOCENE_MILLENNIUM = 262

    HOLOCENE_SEASON = 263
    HOLOCENE_SPRING = 264
    HOLOCENE_FALL = 265
    HOLOCENE_WINTER = 266
    HOLOCENE_SUMMER = 267

    # Before Present (BP) years is a time scale used mainly in archaeology,
    # geology and other scientific disciplines to specify when events
    # occurred in the past. Because the "present" time changes, standard
    # practice is to use 1 January 1950 as the commencement date
    BEFORE_PRESENT = 666
    BEFORE_PRESENT_MICROSECOND = 268
    BEFORE_PRESENT_MILLISECOND = 269
    BEFORE_PRESENT_SECOND = 270
    BEFORE_PRESENT_MINUTE = 271
    BEFORE_PRESENT_HOUR = 272
    BEFORE_PRESENT_DAY = 273
    BEFORE_PRESENT_WEEK = 274
    BEFORE_PRESENT_WEEKEND = 275
    BEFORE_PRESENT_MONTH = 276
    BEFORE_PRESENT_YEAR = 277
    BEFORE_PRESENT_DECADE = 278
    BEFORE_PRESENT_CENTURY = 279
    BEFORE_PRESENT_MILLENNIUM = 280

    BEFORE_PRESENT_SEASON = 281
    BEFORE_PRESENT_SPRING = 282
    BEFORE_PRESENT_FALL = 283
    BEFORE_PRESENT_WINTER = 284
    BEFORE_PRESENT_SUMMER = 285

    # After the Development of Agriculture (ADA) is a system for
    # counting years forward from 8000 BCE, making 2020 the year 10020 ADA
    ADA = 701
    ADA_MICROSECOND = 286
    ADA_MILLISECOND = 287
    ADA_SECOND = 288
    ADA_MINUTE = 289
    ADA_HOUR = 290
    ADA_DAY = 291
    ADA_WEEK = 292
    ADA_WEEKEND = 293
    ADA_MONTH = 294
    ADA_YEAR = 295
    ADA_DECADE = 296
    ADA_CENTURY = 297
    ADA_MILLENNIUM = 298

    ADA_SEASON = 299
    ADA_SPRING = 300
    ADA_FALL = 301
    ADA_WINTER = 302
    ADA_SUMMER = 303

    # Alexandrian Era - 25 March 5493 BC
    ALEXANDRIAN_MICROSECOND = 304
    ALEXANDRIAN_MILLISECOND = 305
    ALEXANDRIAN_SECOND = 306
    ALEXANDRIAN_MINUTE = 307
    ALEXANDRIAN_HOUR = 308
    ALEXANDRIAN_DAY = 309
    ALEXANDRIAN_WEEK = 310
    ALEXANDRIAN_WEEKEND = 311
    ALEXANDRIAN_MONTH = 312
    ALEXANDRIAN_YEAR = 313
    ALEXANDRIAN_DECADE = 314
    ALEXANDRIAN_CENTURY = 315
    ALEXANDRIAN_MILLENNIUM = 316

    ALEXANDRIAN_SEASON = 317
    ALEXANDRIAN_SPRING = 318
    ALEXANDRIAN_FALL = 319
    ALEXANDRIAN_WINTER = 320
    ALEXANDRIAN_SUMMER = 321

    # "Creation Era of Constantinople" or "Era of the World"
    # September 1, 5509 BC
    CEC = 702
    CEC_MICROSECOND = 322
    CEC_MILLISECOND = 323
    CEC_SECOND = 324
    CEC_MINUTE = 325
    CEC_HOUR = 326
    CEC_DAY = 327
    CEC_WEEK = 328
    CEC_WEEKEND = 329
    CEC_MONTH = 330
    CEC_YEAR = 331
    CEC_DECADE = 332
    CEC_CENTURY = 333
    CEC_MILLENNIUM = 334

    CEC_SEASON = 335
    CEC_SPRING = 336
    CEC_FALL = 337
    CEC_WINTER = 338
    CEC_SUMMER = 339

    ### Everything bellow only for convenience

    # Rata Die, number of days elapsed since 1 January 1 in the proleptic
    # Gregorian calendar.
    # TODO this has to be changed \
    #  this is pulling utterances like "11 may..." \
    #  elif not date_found and resolution == DateTimeResolution.RATADIE and \
    #                     is_numeric_de(token.word): \
    #  resolution == DateTimeResolution.RATADIE => DateTimeResolution.DAY
    RATADIE = DAY
    RATADIE_MICROSECOND = MICROSECOND
    RATADIE_MILLISECOND = MILLISECOND
    RATADIE_SECOND = SECOND
    RATADIE_MINUTE = MINUTE
    RATADIE_HOUR = HOUR
    RATADIE_DAY = DAY
    RATADIE_WEEK = WEEK
    RATADIE_WEEKEND = WEEKEND
    RATADIE_MONTH = MONTH
    RATADIE_YEAR = YEAR
    RATADIE_DECADE = DECADE
    RATADIE_CENTURY = CENTURY
    RATADIE_MILLENNIUM = MILLENNIUM

    RATADIE_SEASON = SEASON
    RATADIE_SPRING = SPRING
    RATADIE_FALL = FALL
    RATADIE_WINTER = WINTER
    RATADIE_SUMMER = SUMMER

    # CommonEra, since 1 January 1 in the proleptic Gregorian calendar.
    CE = DAY
    CE_MICROSECOND = MICROSECOND
    CE_MILLISECOND = MILLISECOND
    CE_SECOND = SECOND
    CE_MINUTE = MINUTE
    CE_HOUR = HOUR
    CE_DAY = DAY
    CE_WEEK = WEEK
    CE_WEEKEND = WEEKEND
    CE_MONTH = MONTH
    CE_YEAR = YEAR
    CE_DECADE = DECADE
    CE_CENTURY = CENTURY
    CE_MILLENNIUM = MILLENNIUM

    CE_SEASON = SEASON
    CE_SPRING = SPRING
    CE_FALL = FALL
    CE_WINTER = WINTER
    CE_SUMMER = SUMMER


class Season(Enum):
    SPRING = 0
    SUMMER = 1
    FALL = 2
    WINTER = 3


class Hemisphere(Enum):
    NORTH = 0
    SOUTH = 1


def get_active_hemisphere():
    """
    Get the hemisphere of the current location.

    Returns:
        Enum: Hemisphere
    """
    from ovos_config import Configuration
    __latitude = Configuration().get("location", {}).get("coordinate", {})\
                .get("latitude", 38.971669)
    if __latitude < 0:
        return Hemisphere.SOUTH
    return Hemisphere.NORTH


def get_week_range(ref_date: datetime) -> Tuple[datetime, datetime]:
    """
    Get the start and end dates of the week containing the given reference date.

    Args:
        ref_date (datetime): The reference date to use for calculating the week
                             range.

    Returns:
        Tuple[datetime, datetime]: A tuple of datetime objects representing the
                                   start and end dates of the week containing the
                                   given reference date.
    """
    start = ref_date - timedelta(days=ref_date.weekday())
    end = start + timedelta(days=6)
    return start, end


def get_weekend_range(ref_date: datetime) -> Tuple[datetime, datetime]:
    """
    Args:
        ref_date (datetime): The reference date to use for calculating the weekend
                             range.

    Returns:
        Tuple[datetime, datetime]: Returns a tuple of datetime objects representing
                                   the start and end dates of the coming weekend.
    """
    if ref_date.weekday() < 5:
        start, _ = get_week_range(ref_date)
        start = start + timedelta(days=5)
    elif ref_date.weekday() == 5:
        start = ref_date
    elif ref_date.weekday() == 6:
        start = ref_date - timedelta(days=1)
    return start, start + timedelta(days=1)


def get_month_range(ref_date: datetime) -> Tuple[datetime, datetime]:
    """
    Get the start and end dates for the month containing the given reference date.

    Args:
        ref_date (datetime): The reference date to use for calculating the month
                             range.

    Returns:
        Tuple[datetime, datetime]: Returns a tuple of datetime objects representing
                                   the start and end dates.
    """
    start = ref_date.replace(day=1)
    if ref_date.month == 12:
        end = ref_date.replace(day=31)
    else:
        end = ref_date.replace(day=1, month=ref_date.month + 1) - timedelta(days=1)
    return start, end


def get_year_range(ref_date: datetime) -> Tuple[datetime, datetime]:
    """
    Get the start and end dates for the year containing the given reference date.

    Args:
        ref_date (datetime): The reference date to use for calculating the year
                                range.

    Returns:
        Tuple[datetime, datetime]: Returns a tuple of datetime objects representing
                                      the start and end dates.
    """

    start = ref_date.replace(day=1, month=1)
    end = ref_date.replace(day=31, month=12)
    return start, end


def get_decade_range(ref_date: datetime) -> Tuple[datetime, datetime]:
    """
    Get the start and end dates for the decade containing the given reference date.
    
    Args:
        ref_date (datetime): The reference date to use for calculating the decade
                             range.
                                
    Returns:
        Tuple[datetime, datetime]: Returns a tuple of datetime objects representing
                                   the start and end dates.
                                            
    """
    start = datetime(day=1, month=1, year=(ref_date.year // 10)*10)
    end = datetime(day=31, month=12, year=start.year + 9)
    return start, end


def get_century_range(ref_date: datetime) -> Tuple[datetime, datetime]:
    """
    Get the start and end dates for the century containing the given reference date.

    Args:
        ref_date (datetime): The reference date to use for calculating the century
                                range.
    
    Returns:
        Tuple[datetime, datetime]: Returns a tuple of datetime objects representing
                                        the start and end dates.
    """
    start = datetime(day=1, month=1, year=(ref_date.year // 100) * 100)
    end = datetime(day=31, month=12, year=start.year + 99)
    return start, end


def get_millennium_range(ref_date: datetime) -> Tuple[datetime, datetime]:
    """
    Get the start and end dates for the millennium containing the given reference date.

    Args:
        ref_date (datetime): The reference date to use for calculating the millennium
                                range.
    
    Returns:
        Tuple[datetime, datetime]: Returns a tuple of datetime objects representing
                                        the start and end dates.
    """
    start = datetime(day=1, month=1, year=(ref_date.year // 1000) * 1000)
    end = datetime(day=31, month=12, year=start.year + 999)
    return start, end


def get_date_ordinal(ordinal,
                     offset: Optional[int] = None,
                     ref_date: Optional[datetime] = None,
                     resolution: Enum = DateTimeResolution.DAY_OF_MONTH)\
                     -> datetime:
    """
    Returns a datetime object representing the date of the given ordinal 
    based on the resolution and offset.

    Example:
        >>> get_date_ordinal(1, ref_date=datetime(2020, 1, 1), resolution=DateTimeResolution.DAY_OF_MONTH)
        datetime.datetime(2020, 1, 1, 0, 0)
        >>> get_date_ordinal(2, ref_date=datetime(2020, 1, 1), resolution=DateTimeResolution.DAY_OF_WEEK)
        datetime.datetime(2019, 12, 31, 0, 0)

    Args:
        ordinal (int): The ordinal day of the month or week to get the date for.
        offset (int, optional): offset relative to the reference date.
        ref_date (datetime, optional): The reference date to use as a starting
                                       point. Defaults to None, which uses the
                                       current date.
        resolution (DateTimeResolution, optional): The resolution to use when
                                       calculating the date. Defaults to
                                       DateTimeResolution.DAY_OF_MONTH.

    Returns:
        datetime: A datetime object representing the date of the given ordinal
                  day of the month or week.
    """

    ordinal = int(ordinal)
    ref_date = ref_date or now_local()

    _decade = (ref_date.year // 10) * 10 or 1
    _century = (ref_date.year // 100) * 100 or 1
    _mil = (ref_date.year // 1000) * 1000 or 1

    # before present
    bp = datetime(year=1950, day=1, month=1)

    if resolution == DateTimeResolution.DAY:
        if ordinal < 0:
            raise OverflowError("The last day of existence can not be "
                                "represented")
        ordinal -= 1
        return datetime(year=1, day=1, month=1) + timedelta(days=ordinal)
    elif resolution == DateTimeResolution.DAY_OF_WEEKEND:
        raise NotImplementedError
    # second day of last week, .. week 53
    elif resolution == DateTimeResolution.DAY_OF_WEEK:
        _start, _ = get_week_range(ref_date)
        if offset:
            _start += timedelta(days=offset*7)
        return _start + timedelta(days=ordinal-1)        
    # second friday of july, second day of july, last day of july
    elif resolution == DateTimeResolution.DAY_OF_MONTH:
        day = ordinal
        if ordinal == -1:
            # last day
            if ref_date.month + 1 == 13:
                return ref_date.replace(day=31, month=12)
            return ref_date.replace(month=ref_date.month + 1, day=1) - \
                timedelta(days=1)
        if offset:
            first_day = ref_date.weekday()
            if offset >= first_day:
                day = 1 + offset - first_day + ((ordinal-1) * 7)
            else:
                day = 1 + offset - first_day + (ordinal * 7)
        return ref_date.replace(day=day)

    elif resolution == DateTimeResolution.DAY_OF_YEAR:
        if ordinal == -1:
            # last day
            return datetime(year=ref_date.year, day=31, month=12)
        ordinal -= 1
        return datetime(year=ref_date.year, day=1, month=1) + \
            timedelta(days=ordinal)
    elif resolution == DateTimeResolution.DAY_OF_DECADE:
        if ordinal == -1:
            # last day
            if _decade + 10 == 10000:
                return datetime(year=9999, day=31, month=12)
            return datetime(year=_decade + 10, day=1, month=1) - timedelta(1)
        ordinal -= 1
        return datetime(year=_decade, day=1, month=1) + timedelta(days=ordinal)

    elif resolution == DateTimeResolution.DAY_OF_CENTURY:
        if ordinal == -1:
            # last day
            if _century + 100 == 10000:
                return datetime(year=9999, day=31, month=12)
            return datetime(year=_century + 100, day=1, month=1) - timedelta(1)

        return datetime(year=_century, day=1, month=1) + timedelta(days=ordinal - 1)

    elif resolution == DateTimeResolution.DAY_OF_MILLENNIUM:
        if ordinal == -1:
            # last day
            if _mil + 1000 == 10000:
                return datetime(year=9999, day=31, month=12)
            return datetime(year=_mil + 1000, day=1, month=1) - timedelta(1)
        return datetime(year=_mil, day=1, month=1) + timedelta(days=ordinal - 1)

    elif resolution == DateTimeResolution.WEEK:
        if ordinal < 0:
            raise OverflowError("The last week of existence can not be "
                                "represented")
        _day = datetime(1, 1, 1) + relativedelta(weeks=ordinal) - timedelta(days=1)
        _start, _end = get_week_range(_day)
        return _start

    elif resolution == DateTimeResolution.WEEK_OF_MONTH:
        if ordinal == -1:
            _day = ref_date.replace(day=1) + relativedelta(months=1) - \
                timedelta(days=1)
        else:
            if not 0 < ordinal <= 4:
                raise ValueError("months only have 4 weeks")

            _day = ref_date.replace(day=1) + relativedelta(weeks=ordinal) - \
                timedelta(days=1)

        _start, _end = get_week_range(_day)
        return _start

    elif resolution == DateTimeResolution.WEEK_OF_YEAR:
        if ordinal == -1:
            _day = ref_date.replace(day=31, month=12)
        else:
            _day = ref_date.replace(day=1, month=1) + relativedelta(
                weeks=ordinal) - timedelta(days=1)

        _start, _end = get_week_range(_day)
        return _start

    elif resolution == DateTimeResolution.WEEK_OF_DECADE:
        if ordinal == -1:
            _day = datetime(day=31, month=12, year=_decade + 9)
        else:
            _day = datetime(day=1, month=1, year=_decade) + \
                   relativedelta(weeks=ordinal) - timedelta(days=1)
        _start, _end = get_week_range(_day)
        return _start

    elif resolution == DateTimeResolution.WEEK_OF_CENTURY:
        if ordinal == -1:
            _day = datetime(day=31, month=12, year=_century + 99)
        else:
            _day = datetime(day=1, month=1, year=_century) + \
                   relativedelta(weeks=ordinal) - timedelta(days=1)

        _start, _end = get_week_range(_day)

        return _start
    elif resolution == DateTimeResolution.WEEK_OF_MILLENNIUM:
        if ordinal == -1:
            _day = datetime(day=31, month=12, year=_mil + 999)
        else:
            _day = datetime(day=1, month=1, year=_mil) + \
                   relativedelta(weeks=ordinal) - timedelta(days=1)

        _start, _end = get_week_range(_day)
        return _start

    elif resolution == DateTimeResolution.MONTH:
        if ordinal < 0:
            raise OverflowError("The last month of existence can not be "
                                "represented")
        return datetime(year=1, day=1, month=1) + relativedelta(months=ordinal - 1)
    elif resolution == DateTimeResolution.MONTH_OF_YEAR:
        if ordinal == -1:
            return ref_date.replace(month=12, day=1)
        return ref_date.replace(day=1, month=1) + \
            relativedelta(months=ordinal - 1)
    elif resolution == DateTimeResolution.MONTH_OF_CENTURY:
        if ordinal == -1:
            return datetime(year=_century + 99, day=1, month=12)
        _date = ref_date.replace(month=1, day=1, year=_century)
        _date += relativedelta(months=ordinal - 1)
        return _date
    elif resolution == DateTimeResolution.MONTH_OF_DECADE:
        if ordinal == -1:
            return datetime(year=_decade + 9, day=1, month=12)
        _date = ref_date.replace(month=1, day=1, year=_decade)
        _date += relativedelta(months=ordinal - 1)
        return _date
    elif resolution == DateTimeResolution.MONTH_OF_MILLENNIUM:
        if ordinal == -1:
            return datetime(year=_mil + 999, day=1, month=12)
        _date = ref_date.replace(month=1, day=1, year=_mil)
        _date += relativedelta(months=ordinal - 1)
        return _date

    elif resolution == DateTimeResolution.YEAR:
        if ordinal == -1:
            raise OverflowError("The last year of existence can not be "
                                "represented")
        if ordinal == 0:
            # NOTE: no year 0
            return datetime(year=1, day=1, month=1)
        return datetime(year=ordinal, day=1, month=1)
    elif resolution == DateTimeResolution.YEAR_OF_DECADE:
        if ordinal == -1:
            return datetime(year=_decade + 9, day=1, month=1)
        if ordinal == 0:
            # NOTE: no year 0
            return datetime(year=1, day=1, month=1)
        assert 0 < ordinal < 10
        return datetime(year=_decade + ordinal - 1, day=1, month=1)
    elif resolution == DateTimeResolution.YEAR_OF_CENTURY:
        if ordinal == -1:
            return datetime(year=_century + 99, day=1, month=1)
        if ordinal == 0:
            # NOTE: no year 0
            return datetime(year=1, day=1, month=1)
        return datetime(year=_century + ordinal - 1, day=1, month=1)
    elif resolution == DateTimeResolution.YEAR_OF_MILLENNIUM:
        if ordinal == -1:
            return datetime(year=_mil + 999, day=1, month=1)
        if ordinal == 0:
            # NOTE: no year 0
            return datetime(year=1, day=1, month=1)
        return datetime(year=_mil + ordinal - 1, day=1, month=1)
    elif resolution == DateTimeResolution.DECADE:
        if ordinal == -1:
            raise OverflowError("The last decade of existence can not be "
                                "represented")
        if ordinal == 1:
            return datetime(day=1, month=1, year=1)
        ordinal -= 1
        return datetime(year=ordinal * 10, day=1, month=1)
    elif resolution == DateTimeResolution.DECADE_OF_CENTURY:
        if ordinal == -1:
            return datetime(year=_century + 90, day=1, month=1)

        assert 0 < ordinal < 10

        if ordinal == 1:
            return datetime(day=1, month=1, year=_century)
        ordinal -= 1
        return datetime(year=_century + ordinal * 10, day=1, month=1)
    elif resolution == DateTimeResolution.DECADE_OF_MILLENNIUM:
        if ordinal == -1:
            return datetime(year=_mil + 990, day=1, month=1)

        assert 0 < ordinal < 1000

        if ordinal == 1:
            return datetime(day=1, month=1, year=_mil)
        ordinal -= 1
        return datetime(year=_mil + ordinal * 10,  day=1, month=1)
    elif resolution == DateTimeResolution.CENTURY:
        if ordinal == -1:
            raise OverflowError("The last century of existence can not be "
                                "represented")
        if ordinal == 1:
            return datetime(day=1, month=1, year=1)
        ordinal -= 1  # no century 0 / year 0
        return datetime(year=ordinal * 100, day=1, month=1)
    elif resolution == DateTimeResolution.CENTURY_OF_MILLENNIUM:
        if ordinal == -1:
            return datetime(year=_mil + 900, day=1, month=1)

        assert 0 < ordinal < 100

        if ordinal == 1:
            return datetime(day=1, month=1, year=_mil)
        ordinal -= 1
        return datetime(year=_mil + ordinal * 100,  day=1, month=1)
    elif resolution == DateTimeResolution.MILLENNIUM:
        if ordinal < 0:
            raise OverflowError("The last millennium of existence can not be "
                                "represented")
        if ordinal == 1:
            return datetime(day=1, month=1, year=1)
        ordinal -= 1
        return datetime(year=ordinal * 1000, day=1, month=1)
    elif resolution == DateTimeResolution.BEFORE_PRESENT_DAY:
        if ordinal < 0:
            raise OverflowError("Can not represent dates BC")
        return bp - relativedelta(days=ordinal)
    elif resolution == DateTimeResolution.BEFORE_PRESENT_WEEK:
        if ordinal < 0:
            raise OverflowError("Can not represent dates BC")
        _week = bp - relativedelta(weeks=ordinal)
        _start, _end = get_week_range(_week)
        return _end
    elif resolution == DateTimeResolution.BEFORE_PRESENT_MONTH:
        if ordinal < 0:
            raise OverflowError("Can not represent dates BC")
        return bp - relativedelta(months=ordinal)
    elif resolution == DateTimeResolution.BEFORE_PRESENT_YEAR:
        if ordinal < 0:
            raise OverflowError("Can not represent dates BC")
        return bp - relativedelta(years=ordinal)
    elif resolution == DateTimeResolution.BEFORE_PRESENT_DECADE:
        if ordinal < 0:
            raise OverflowError("Can not represent dates BC")
        return bp - relativedelta(years=10 * ordinal)
    elif resolution == DateTimeResolution.BEFORE_PRESENT_CENTURY:
        if ordinal < 0:
            raise OverflowError("Can not represent dates BC")
        return bp - relativedelta(years=100 * ordinal)
    elif resolution == DateTimeResolution.BEFORE_PRESENT_MILLENNIUM:
        if ordinal < 0:
            raise OverflowError("Can not represent dates BC")
        return bp - relativedelta(years=1000 * ordinal)

    raise ValueError("Invalid DateTimeResolution")


def date_to_season(ref_date: Optional[datetime] = None,
                   hemisphere: Enum = Hemisphere.NORTH):
    """
    Returns the season of the given date.

    Example:
        >>> date_to_season(datetime(day=1, month=1, year=2018))
        Season.WINTER
    Args:
        ref_date, optional(datetime): The date to get the season of.
                                      Defaults to now.
        hemisphere: The hemisphere to use. Defaults to the northern hemisphere.

    Returns:
        Enum: The season of the given date.    
    """
    ref_date = ref_date or now_local()

    if hemisphere == Hemisphere.NORTH:
        fall = (
            datetime(day=1, month=9, year=ref_date.year),
            datetime(day=30, month=11, year=ref_date.year)
        )
        spring = (
            datetime(day=1, month=3, year=ref_date.year),
            datetime(day=31, month=5, year=ref_date.year)
        )
        summer = (
            datetime(day=1, month=6, year=ref_date.year),
            datetime(day=31, month=8, year=ref_date.year)
        )

        if fall[0] <= ref_date < fall[1]:
            return Season.FALL
        if summer[0] <= ref_date < summer[1]:
            return Season.SUMMER
        if spring[0] <= ref_date < spring[1]:
            return Season.SPRING
        return Season.WINTER

    else:
        spring = (
            datetime(day=1, month=9, year=ref_date.year),
            datetime(day=30, month=11, year=ref_date.year)
        )
        fall = (
            datetime(day=1, month=3, year=ref_date.year),
            datetime(day=31, month=5, year=ref_date.year)
        )
        winter = (
            datetime(day=1, month=6, year=ref_date.year),
            datetime(day=31, month=8, year=ref_date.year)
        )

        if fall[0] <= ref_date < fall[1]:
            return Season.FALL
        if winter[0] <= ref_date < winter[1]:
            return Season.WINTER
        if spring[0] <= ref_date < spring[1]:
            return Season.SPRING
        return Season.SUMMER


def season_to_date(season: Enum,
                   year: Optional[int] = None,
                   hemisphere: Optional[Enum] = Hemisphere.NORTH):
    """
    Returns the date of the given season.

    Example:
        >>> season_to_date(Season.SPRING, year=2018)
        datetime(day=1, month=3, year=2018)
    
    Args:
        season, Enum: The season to get the date of.
        year, optional(int): The year to get the date of. Defaults to the current
                             year.
        hemisphere, optional(Enum): The hemisphere to use. Defaults to the
                                    northern hemisphere.
    
    Returns:
        datetime: The date of the given season.
    """
    if year is None:
        year = now_local().year
    elif not isinstance(year, int):
        year = year.year

    if hemisphere == Hemisphere.NORTH:
        if season == Season.SPRING:
            return datetime(day=20, month=3, year=year)
        elif season == Season.FALL:
            return datetime(day=22, month=9, year=year)
        elif season == Season.WINTER:
            return datetime(day=21, month=12, year=year)
        elif season == Season.SUMMER:
            return datetime(day=21, month=6, year=year)
    else:
        if season == Season.SPRING:
            return datetime(day=22, month=9, year=year)
        elif season == Season.FALL:
            return datetime(day=20, month=3, year=year)
        elif season == Season.WINTER:
            return datetime(day=21, month=6, year=year)
        elif season == Season.SUMMER:
            return datetime(day=21, month=12, year=year)
    raise ValueError("Unknown Season")


def next_season_date(season, ref_date=None, hemisphere=Hemisphere.NORTH):
    """
    Returns the date of the next season.
    
    Example:
        >>> next_season_date(Season.SPRING,
                             ref_date=datetime(day=1, month=1, year=2018))
        datetime(day=20, month=3, year=2018)
    
    Args:
        season, Enum: The season to get the date of.
        ref_date, optional(datetime): The date to get the next season of.
                                      Defaults to now.
        hemisphere, optional(Enum): The hemisphere to use. Defaults to the
                                    northern hemisphere.
    
    Returns:
        datetime: The date of the next season.
    """
    ref_date = ref_date or now_local()
    start_day = season_to_date(season, ref_date, hemisphere) \
        .timetuple().tm_yday
    # get the current day of the year
    doy = ref_date.timetuple().tm_yday

    if doy <= start_day:
        # season is this year
        return season_to_date(season, ref_date, hemisphere)
    else:
        # season is next year
        ref_date = ref_date.replace(year=ref_date.year + 1)
        return season_to_date(season, ref_date, hemisphere)


def last_season_date(season, ref_date=None, hemisphere=Hemisphere.NORTH):
    """
    Returns the date of the last season.

    Example:
        >>> last_season_date(Season.SPRING,
                             ref_date=datetime(day=1, month=1, year=2018))
        datetime(day=20, month=3, year=2017)
    
    Args:
        season, Enum: The season to get the date of.
        ref_date, optional(datetime): The date to get the last season of.
                                        Defaults to now.
        hemisphere, optional(Enum): The hemisphere to use. Defaults to the
                                    northern hemisphere.
    
    Returns:
        datetime: The date of the last season.
    """
    ref_date = ref_date or now_local()

    start_day = season_to_date(season, ref_date, hemisphere)\
        .timetuple().tm_yday
    # get the current day of the year
    doy = ref_date.timetuple().tm_yday

    if doy <= start_day:
        # season is previous year
        ref_date = ref_date.replace(year=ref_date.year - 1)
        return season_to_date(season, ref_date, hemisphere)
    else:
        # season is this year
        return season_to_date(season, ref_date, hemisphere)


def get_season_range(ref_date=None, hemisphere=Hemisphere.NORTH):
    """
    Returns the date range of the current season.

    Example:
        >>> get_season_range(ref_date=datetime(day=1, month=1, year=2018))
        (datetime(day=1, month=1, year=2018),
         datetime(day=20, month=3, year=2018))

    Args:
        ref_date, optional(datetime): The date to get the season of.
                                      Defaults to now.
        hemisphere, optional(Enum): The hemisphere to use. Defaults to the
                                    northern hemisphere.
    
    Returns:
        tuple(datetime, datetime): The start and end dates of the current
                                   season.
    """
    ref_date = ref_date or now_local()
    if hemisphere == Hemisphere.NORTH:
        fall = (
            datetime(day=1, month=9, year=ref_date.year),
            datetime(day=30, month=11, year=ref_date.year)
        )
        spring = (
            datetime(day=20, month=3, year=ref_date.year),
            datetime(day=21, month=6, year=ref_date.year)
        )
        summer = (
            datetime(day=21, month=6, year=ref_date.year),
            datetime(day=22, month=9, year=ref_date.year)
        )
        fall = (
            datetime(day=22, month=9, year=ref_date.year),
            datetime(day=21, month=12, year=ref_date.year)
        )
        winter = (
            datetime(day=21, month=12, year=ref_date.year),
            datetime(day=20, month=3, year=ref_date.year + 1)
        )

        if fall[0] <= ref_date < fall[1]:
            return fall
        if summer[0] <= ref_date < summer[1]:
            return summer
        if spring[0] <= ref_date < spring[1]:
            return spring
        if winter[0] <= ref_date < winter[1]:
            return winter

    else:
        spring = (
            datetime(day=22, month=9, year=ref_date.year),
            datetime(day=21, month=12, year=ref_date.year)
        )
        summer = (
            datetime(day=21, month=12, year=ref_date.year),
            datetime(day=20, month=3, year=ref_date.year + 1)
        )
        fall = (
            datetime(day=20, month=3, year=ref_date.year),
            datetime(day=21, month=6, year=ref_date.year)
        )
        winter = (
            datetime(day=21, month=6, year=ref_date.year),
            datetime(day=22, month=9, year=ref_date.year)
        )

        if fall[0] <= ref_date < fall[1]:
            return fall
        if winter[0] <= ref_date < winter[1]:
            return winter
        if spring[0] <= ref_date < spring[1]:
            return spring
        if summer[0] <= ref_date < summer[1]:
            return summer


def get_week_number(ref_date=None):
    """
    Returns the week number of the year.

    Example:
        >>> get_week_number(ref_date=datetime(day=1, month=1, year=2018))
        1
    
    Args:
        ref_date, optional(datetime): The date to get the week number of.
                                      Defaults to now.
    
    Returns:
        int: The week number of the year.
    """
    ref_date = ref_date or now_local()
    return ref_date.isocalendar()[1]
