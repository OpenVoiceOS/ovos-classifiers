import re
from datetime import datetime, timedelta
from typing import List, Dict, Union, Any, Optional

from ovos_classifiers.heuristics.numeric import EnglishNumberParser, GermanNumberParser
from ovos_classifiers.heuristics.tokenize import ReplaceableNumber, ReplaceableTimedelta, \
    ReplaceableTime, ReplaceableDate, Token, word_tokenize
from ovos_utils.time import DAYS_IN_1_MONTH, DAYS_IN_1_YEAR


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

    def extract_durations(self, tokens):
        """
        Extract all timedeltas from a list of Tokens, with the words that
        represent them.

        Args:
            [Token]: The tokens to parse.

        Returns:
            [ReplaceableTimedelta]: A list of tuples, each containing a timedelta and a
                             string.

        """
        if isinstance(tokens, str):
            tokens = [Token(word.lower(), index) for index, word in enumerate(word_tokenize(tokens))]

        time_units = {
            'microseconds': 0,
            'milliseconds': 0,
            'seconds': 0,
            'minutes': 0,
            'hours': 0,
            'days': 0,
            'weeks': 0
        }

        # handle "a day" -> "1 day"
        for idx, tok in enumerate(tokens):
            if tok.word != "a" or idx == len(tokens) - 1:
                continue
            next_tok = tokens[idx + 1]
            is_dur = next_tok.word in ['day', 'month', 'year', 'decade', 'century', 'millennium'] or \
                     next_tok.word + "s" in time_units.keys()
            if is_dur:
                tokens[idx] = Token("1", idx)


        numbers = EnglishNumberParser().extract_numbers(tokens)

        durations = []
        for idx, number in enumerate(numbers):
            if number.end_index == len(tokens) - 1:
                break

            next_token = tokens[number.end_index + 1]
            unit_en = next_token.word.rstrip("s")

            if unit_en + "s" in time_units:
                time_units[unit_en+  "s"] += number.value
            elif unit_en == "month":
                time_units["days"] += DAYS_IN_1_MONTH * number.value
            elif unit_en == "year":
                time_units["days"] += DAYS_IN_1_YEAR * number.value
            elif unit_en == "decade":
                time_units["days"] += 10 * DAYS_IN_1_YEAR * number.value
            elif unit_en == "century" or unit_en == "centuries":
                time_units["days"] += 100 * DAYS_IN_1_YEAR * number.value
            elif unit_en == "millennium" or unit_en == "millenia":
                time_units["days"] += 1000 * DAYS_IN_1_YEAR * number.value

            # if we have any duration, save the extraction, else it was just a number
            if any(time_units.values()):
                toks = tokens[number.start_index:number.end_index+2]
                delta = timedelta(**time_units)

                # if we have a previous duration without intermediate tokens
                # AND it is larger than current, merge
                prev_dur = None
                prev_word = "" if number.start_index == 0 else tokens[number.start_index - 1].word
                if len(durations):
                    prev_dur = durations[-1]

                if prev_dur and prev_dur.value > delta and \
                        any((prev_dur.end_index == number.start_index - 1,
                            prev_dur.end_index == number.start_index - 2 and prev_word == "and"
                            )):
                    delta = prev_dur.value + delta
                    toks  = tokens[prev_dur.start_index:number.end_index+3]
                    durations[-1] = ReplaceableTimedelta(delta, toks)
                else:
                    durations.append(ReplaceableTimedelta(delta, toks))

                # reset for next number
                time_units = {
                    'microseconds': 0,
                    'milliseconds': 0,
                    'seconds': 0,
                    'minutes': 0,
                    'hours': 0,
                    'days': 0,
                    'weeks': 0
                }

        durations.sort(key=lambda n: n.start_index)
        return durations


class GermanTimeTagger:
    def extract_date(self, text: str, anchorDate: Optional[datetime] = None):
        raise NotImplementedError

    def extract_time(self, text: str, anchorDate: Optional[datetime] = None):
        raise NotImplementedError
    
    def extract_durations(self, tokens: Union[List[Token], str]) -> List[ReplaceableTimedelta]:

        if isinstance(tokens, str):
            tokens = [Token(word.lower(), index) for index, word in enumerate(word_tokenize(tokens))]

        numbers = GermanNumberParser().extract_numbers(tokens)

        # Einzahl, Mehrzahl und Flexionen
        pattern = r"\b(?P<unit>{unit}[nes]?[sn]?\b)"

        durations = []
        for number in numbers:
            if number.end_index == len(tokens) - 1:
                break

            time_units: Dict[str, Any] = {
                'microseconds': 'mikrosekunden',
                'milliseconds': 'millisekunden',
                'seconds': 'sekunden',
                'minutes': 'minuten',
                'hours': 'stunden',
                'days': 'tage',
                'weeks': 'wochen'
            }

            next_token = tokens[number.end_index + 1]
            test_str = next_token.word
            toks = []

            for (unit_en, unit_de) in time_units.items():
                time_units[unit_en] = 0
                if toks:
                    continue

                if re.match(pattern.format(unit=unit_de[:-1]), test_str):
                    time_units[unit_en] = number.value
                    toks = tokens[number.start_index:number.end_index+2]
            
            if toks:  
                delta = timedelta(**time_units)
                prev_dur = durations[-1] if len(durations) else None
                prev_word = "" if number.start_index == 0 else tokens[number.start_index - 1].word

                if prev_dur and prev_dur.value > delta and \
                        any((prev_dur.end_index == number.start_index - 1,
                            prev_dur.end_index == number.start_index - 2 and prev_word == "und"
                            )):
                    delta = prev_dur.value + delta
                    toks  = tokens[prev_dur.start_index:number.end_index+3]
                    durations[-1] = ReplaceableTimedelta(delta, toks)
                else:
                    durations.append(ReplaceableTimedelta(delta, toks))
    
        durations.sort(key=lambda n: n.start_index)
        return durations



if __name__ == "__main__":
    t = EnglishTimeTagger()
    print(t.extract_durations("remind me in a minute"))
    print(t.extract_durations("remind me in one hundred minutes"))
    print(t.extract_durations("remind me in 10 minutes 5 seconds"))
    print(t.extract_durations("remind me in 10 minutes and 5 seconds"))
    print(t.extract_durations("remind me in 10 seconds and 5 hours and 10 seconds"))
    # [ReplaceableTimedelta(0:01:00, ['1', 'minute'])]
    # [ReplaceableTimedelta(1:40:00, ['one', 'hundred', 'minutes'])]
    # [ReplaceableTimedelta(0:10:05, ['10', 'minutes', '5', 'seconds'])]
    # [ReplaceableTimedelta(0:10:05, ['10', 'minutes', 'and', '5', 'seconds'])]
    # [ReplaceableTimedelta(0:00:10, ['10', 'seconds']), ReplaceableTimedelta(5:00:10, ['5', 'hours', 'and', '10', 'seconds'])]
