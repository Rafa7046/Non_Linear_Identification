def combine(arr, k):
    """
    Returns all possible combinations of k elements with repetition from arr.

    Parameters
    ----------
    arr : array_like
        Array of elements.
    k : int
        Number of elements to combine.

    Returns
    -------
    list
        List of all possible combinations of k elements from arr.
    """
    if k == 0:
        return [[]]
    elif not arr:
        return []
    else:
        head = arr[0]
        tail = arr[1:]

        without_head = combine(tail, k)
        with_head = combine(arr, k - 1)

        with_head = [[head] + x for x in with_head]

        return with_head + without_head


def generate_combinations(arr, size):
    """
    Returns all possible combinations of elements from arr up to length size.

    Parameters
    ----------
    arr : array_like
        Array of elements.
    size : int
        Maximum length of combinations.

    Returns
    -------
    list
        List of all possible combinations.
    """
    answer = []
    for k in range(1, size + 1):
        for comb in combine(arr, k):
            answer.append(comb)
    return answer
