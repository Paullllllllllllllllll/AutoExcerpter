"""Roman numeral conversion utilities."""

# Roman numeral constants
ROMAN_NUMERAL_VALUES = [
    (1000, 'm'), (900, 'cm'), (500, 'd'), (400, 'cd'),
    (100, 'c'), (90, 'xc'), (50, 'l'), (40, 'xl'),
    (10, 'x'), (9, 'ix'), (5, 'v'), (4, 'iv'), (1, 'i')
]


def int_to_roman(num: int) -> str:
    """Convert integer to lowercase Roman numeral string.
    
    Args:
        num: Positive integer to convert.
        
    Returns:
        Lowercase Roman numeral string.
    """
    if num <= 0:
        return ""
    result = []
    for value, numeral in ROMAN_NUMERAL_VALUES:
        while num >= value:
            result.append(numeral)
            num -= value
    return "".join(result)
