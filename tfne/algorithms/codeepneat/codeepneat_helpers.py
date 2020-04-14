from typing import Union


def round_to_nearest_multiple(value, minimum, maximum, multiple) -> Union[int, float]:
    """"""
    lower_multiple = int(value / multiple) * multiple
    if value % multiple - (multiple / 2.0) < 0:
        if minimum <= lower_multiple <= maximum:
            return lower_multiple
        if lower_multiple < minimum:
            return minimum
        if lower_multiple > maximum:
            return maximum
    else:
        higher_multiple = lower_multiple + multiple
        if minimum <= higher_multiple <= maximum:
            return higher_multiple
        if higher_multiple < minimum:
            return minimum
        if higher_multiple > maximum:
            return maximum
