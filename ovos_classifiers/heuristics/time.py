import re
from datetime import datetime, timedelta

from ovos_classifiers.heuristics.numbers import EnglishNumberParser
from ovos_classifiers.utils.time import DAYS_IN_1_MONTH, DAYS_IN_1_YEAR


class EnglishTimeTagger:
    def extract_date(self, text, anchorDate=None):
        """
              Extracts date information from a sentence.  Parses many of the
              common ways that humans express dates and times, including relative dates
              like "5 days from today", "tomorrow', and "Tuesday".

              Args:
                  text (str): the text to be interpreted
                  anchorDate (:obj:`datetime`, optional): the date to be used for
                      relative dating (for example, what does "tomorrow" mean?).
                      Defaults to the current local date/time.
              Returns:
                  extracted_date (datetime.date): 'date' is the extracted date as a datetime.date object.
                      Returns 'None' if no date related text is found.

              Examples:

                  >>> extract_datetime(
                  ... "What is the weather like the day after tomorrow?",
                  ... datetime(2017, 6, 30, 00, 00)
                  ... )
                  datetime.date(2017, 7, 2)

                  >>> extract_datetime(
                  ... "Set up an appointment 2 weeks from Sunday at 5 pm",
                  ... datetime(2016, 2, 19, 00, 00)
                  ... )
                  datetime.date(2016, 3, 6)

                  >>> extract_datetime(
                  ... "Set up an appointment",
                  ... datetime(2016, 2, 19, 00, 00)
                  ... )
                  None
              """
        raise NotImplementedError

    def extract_time(self, text, anchorDate=None):
        """
        Extracts time information from a sentence.  Parses many of the
        common ways that humans express dates and times".

        Vague terminology are given arbitrary values, like:
            - morning = 8 AM
            - afternoon = 3 PM
            - evening = 7 PM

        If a time isn't supplied or implied, the function defaults to 12 AM

        Args:
            text (str): the text to be interpreted
            anchorDate (:obj:`datetime`, optional): the date to be used for
                relative dating (for example, what does "tomorrow" mean?).
                Defaults to the current local date/time.
        Returns:
            extracted_time (datetime.time): 'time' is the extracted time
                as a datetime.time object in the anchorDate (or default if None) timezone.
                Returns 'None' if no time related text is found.

        Examples:

            >>> extract_time(
            ... "What is the weather like the day after tomorrow?",
            ... datetime(2017, 6, 30, 00, 00)
            ... )
            datetime.time(0, 0, 0)

            >>> extract_time(
            ... "Set up an appointment 2 weeks from Sunday at 5 pm",
            ... datetime(2016, 2, 19, 00, 00)
            ... )
            datetime.time(17, 0, 0)

            >>> extract_datetime(
            ... "Set up an appointment",
            ... datetime(2016, 2, 19, 00, 00)
            ... )
            None
        """
        raise NotImplementedError

    def extract_duration(self, text):
        """
     Convert an english phrase into a number of seconds

     Convert things like:
         "10 minute"
         "2 and a half hours"
         "3 days 8 hours 10 minutes and 49 seconds"
     into an int, representing the total number of seconds.

     The words used in the duration will be consumed, and
     the remainder returned.

     As an example, "set a timer for 5 minutes" would return
     (300, "set a timer for").

     Args:
         text (str): string containing a duration

     Returns:
         duration (timedelta): the duration or None if no duration is found
     """
        if not text:
            return None

        time_units = {
            'microseconds': 0,
            'milliseconds': 0,
            'seconds': 0,
            'minutes': 0,
            'hours': 0,
            'days': 0,
            'weeks': 0
        }
        # NOTE: these are spelled wrong on purpose because of the loop below that strips the s
        units = ['months', 'years', 'decades', 'centurys', 'millenniums'] + \
                list(time_units.keys())

        pattern = r"(?P<value>\d+(?:\.?\d+)?)(?:\s+|\-){unit}s?"
        text = EnglishNumberParser().convert_words_to_numbers(text)
        text = text.replace("centuries", "century").replace("millenia", "millennium")
        for word in ('day', 'month', 'year', 'decade', 'century', 'millennium'):
            text = text.replace(f'a {word}', f'1 {word}')

        for unit_en in units:
            unit_pattern = pattern.format(unit=unit_en[:-1])  # remove 's' from unit

            def repl(match):
                time_units[unit_en] += float(match.group(1))
                return ''

            def repl_non_std(match):
                val = float(match.group(1))
                if unit_en == "months":
                    val = DAYS_IN_1_MONTH * val
                if unit_en == "years":
                    val = DAYS_IN_1_YEAR * val
                if unit_en == "decades":
                    val = 10 * DAYS_IN_1_YEAR * val
                if unit_en == "centurys":
                    val = 100 * DAYS_IN_1_YEAR * val
                if unit_en == "millenniums":
                    val = 1000 * DAYS_IN_1_YEAR * val
                time_units["days"] += val
                return ''

            if unit_en not in time_units:
                text = re.sub(unit_pattern, repl_non_std, text)
            else:
                text = re.sub(unit_pattern, repl, text)

        duration = timedelta(**time_units) if any(time_units.values()) else None

        return duration
