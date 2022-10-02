import re
from typing import List


def remove_redundant_whitespaces(s: str) -> str:
    return re.sub(" +", " ", s.strip()).strip()


def remove_strings(s: str, s_list: List[str]) -> str:
    for c in s_list:
        s = s.replace(c, "")
    return s


def replace_strings(s: str, s_list: List[str], t_list: List[str]) -> str:
    assert len(s_list) == len(t_list)
    for c, t in zip(s_list, t_list):
        s = s.replace(c, t)
    return s
