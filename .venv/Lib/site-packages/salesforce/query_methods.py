from typing import List, Any


def _find_val_pos(criteria: str, i: int) -> int:
    parts = criteria.split("%s", i + 1)
    if len(parts) <= i + 1:
        return -1
    return len(criteria) - len(parts[-1]) - 2


def _flatten_values(values: List[Any]) -> List[Any]:
    flat_values = []
    for val in values:
        if type(val) == list:
            flat_values.extend(val)
        else:
            flat_values.append(val)
    return flat_values


def _replace_criteria(criteria: str, i: int, values: List[Any]) -> str:
    num_vals = len(values)
    loc = _find_val_pos(criteria, i)
    place_holder = "'%s'" if type(values[0]) == str else "%s"
    cr_l = [place_holder for __ in range(num_vals)]
    cr_s = ",".join(cr_l)
    cr_ins = f"({cr_s})"
    b = criteria[:loc]
    a = criteria[loc + 2 :]
    return b + cr_ins + a


def in_criteria(criteria: str = None, values: List[Any] = None) -> (str, List[Any]):
    if values:
        if not criteria:
            raise Exception("Criteria are require to set criteria values")
        i = 0
        for value in values:
            if type(value) == list:
                criteria = _replace_criteria(criteria, i, value)
                i += len(value)
            else:
                i += 1
        return criteria % tuple(_flatten_values(values))
    else:
        return criteria
