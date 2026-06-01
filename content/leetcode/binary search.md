Binary search is a "divide and conquer" algorithm designed to find a target value or a specific boundary within a **sorted array** or a **monotonic search space**. In interviews, they tend to ask not just to find a specific number but to identify the "inflection point" where a condition flips from True to False (or vice versa).

Binary search is also one of the rare few algorithms with sub linear time complexity, so when interviewers that a naive solution of O(n) can be optimised further, think binary search!


## monotonic property

Binary search works whenever a search space is **monotonic**. You are looking for the boundary in a sequence where the state changes.
- **The Search Space:** A logical sequence like `[F, F, F, T, T, T]`.
- **The Decision:** If `mid` is `F`, the transition is to the **right**. If `mid` is `T`, `mid` might be the answer, but the first `T` could be further **left**.
- **Calculation:** Always use `mid = low + (high - low) // 2` to prevent integer overflow.

## common patterns in questions

### binary search on a range of answers
there can be a range of answers, and we want to find the "minimum/maximum" possible answer, i.e. the inflection point where True becomes False.

e.g. the range of answers could be like [F, F, F, T, T, T], we want to find the min possible index where it starts to be True.

Define a `check(val)` function that returns `True/False`. If `check(mid)` is true, you try to find a better (smaller/larger) answer.


**Key Problems:** 
[875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/) (Find min speed $K$)
[1011. Capacity To Ship Packages](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/) (Find min capacity $C$)


```python
def solve_bs_on_answer(input_data):
    def check(guess):
        # O(N) Greedy logic to see if 'guess' is valid
        # Return True if valid (T zone), False if invalid (F zone)
        pass

    low, high = min_possible_val, max_possible_val
    ans = high # Default to safest/largest value
    
    while low <= high:
        mid = low + (high - low) // 2
        if check(mid):
            ans = mid      # Mid works, let's try a smaller "True"
            high = mid - 1
        else:
            low = mid + 1  # Mid fails, must increase value
    return ans
```


### maximise the minimum / minimise the maximum
Distributing items or picking values where you want to optimise the "worst-case" scenario.

```python
def maximize_minimum(positions, k):
    positions.sort() # Prerequisite
    
    def can_place(min_dist):
        count, last_pos = 1, positions[0]
        for i in range(1, len(positions)):
            if positions[i] - last_pos >= min_dist:
                count += 1
                last_pos = positions[i]
        return count >= k

    low, high = 1, positions[-1] - positions[0]
    ans = 1
    while low <= high:
        mid = low + (high - low) // 2
        if can_place(mid):
            ans = mid     # Can we place them even further apart?
            low = mid + 1
        else:
            high = mid - 1
    return ans
```



Key Problems:
[Magnetic Force between Two Balls](https://leetcode.com/problems/magnetic-force-between-two-balls/description/)
[Minimise the Maximum Difference of Pairs](https://leetcode.com/problems/minimize-the-maximum-difference-of-pairs/)





## using STL

### python

Core idea: bisect finds **insertion points**, not elements

```python
arr = [1, 3, 3, 5, 7]
#      0  1  2  3  4
```

```python
bisect_left(arr, 3)   # → 1  (insert before existing 3s)
bisect_right(arr, 3)  # → 3  (insert after existing 3s)
```

Think of it as: **where would X slot in to keep the array sorted?**

**Examples**

```python
arr = [1, 3, 3, 5, 7]
```

First element ≥ X → `bisect_left(arr, X)`

```python
bisect_left(arr, 3)  # → 1,  arr[1] = 3  ✓ (first 3)
bisect_left(arr, 4)  # → 3,  arr[3] = 5  ✓ (5 is first ≥ 4)
```

First element > X → `bisect_right(arr, X)`

```python
bisect_right(arr, 3)  # → 3,  arr[3] = 5  ✓ (first thing > 3)
bisect_right(arr, 4)  # → 3,  arr[3] = 5  ✓ (same, since no 4 exists)
```

 Last element ≤ X → `bisect_right(arr, X) - 1`

```python
bisect_right(arr, 3) - 1  # → 2,  arr[2] = 3  ✓ (last 3)
bisect_right(arr, 4) - 1  # → 2,  arr[2] = 3  ✓ (3 is last thing ≤ 4)
```

Last element < X → `bisect_left(arr, X) - 1`

```python
bisect_left(arr, 3) - 1  # → 0,  arr[0] = 1  ✓ (last thing strictly < 3)
bisect_left(arr, 4) - 1  # → 2,  arr[2] = 3  ✓ (3 is last thing < 4)
```

| **Use Case**          | **Function**               | **Logic**                 |
| --------------------- | -------------------------- | ------------------------- |
| First element **≥ X** | `bisect_left(arr, X)`      | Insert point before dupes |
| First element **> X** | `bisect_right(arr, X)`     | Insert point after dupes  |
| Last element **≤ X**  | `bisect_right(arr, X) - 1` | One before right boundary |
| Last element **< X**  | `bisect_left(arr, X) - 1`  | One before left boundary  |

key parameter, allows for transformation of array.
```python
arr = [6, 7, 9, 1, 2, 5]

bisect_left(arr, True, key=lambda n: n <= 5)
```

this becomes, `[False, False, False, True, True, True]`

### `key` parameter — bisect on transformed values

When the array isn't sorted by raw value but **sorted by some property**, use `key`.

```python
# Rotated sorted array — find the minimum
nums = [6, 7, 9, 1, 2, 5]  # nums[-1] = 5
```

```python
# key transforms each element → False/True
# [False, False, False, True, True, True]  ← this is sorted!

bisect_left(nums, True, key=lambda n: n <= nums[-1])
# → 3,  nums[3] = 1  ✓  (first element that drops into lower half)
```

`bisect_left(arr, True, key=...)` = **first index where key flips to True** — same idea as before, just on the transformed array.

---

### `lo` and `hi` — restrict the search range

```python
arr = [1, 3, 3, 5, 7]

bisect_left(arr, 3, lo=2)      # → 2  (search starts at index 2, skips index 1)
bisect_left(arr, 3, lo=0, hi=1) # → 0  (only searches arr[0:1] = [1], inserts at 0)
```

Useful when you **know the answer lives in a subrange** — avoids searching the whole array.
### c++
C++ returns **iterators**. You must use `std::distance` to get the integer index.

**`std::lower_bound`: First element $\ge X$, Last element $< X$\**
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> weights = {10, 20, 30, 30, 40};
    // Find first weight >= 30
    auto it = std::lower_bound(weights.begin(), weights.end(), 30);
    
    int index = std::distance(weights.begin(), it);
    std::cout << "Index: " << index << std::endl; // Output: 2
}
```

**`std::upper_bound`: First element $> X$, Last element $\leq X$**

