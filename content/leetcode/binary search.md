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
We can use the `bisect` module. Works on any sorted list.

| **Use Case**              | **Function**                 | **Logic / Comparison**                          |
| ------------------------- | ---------------------------- | ----------------------------------------------- |
| **First element $\ge X$** | `bisect_left(arr, X)`        | Finds the first index $i$ where $arr[i] \ge X$. |
| **First element $> X$**   | `bisect_right(arr, X)`       | Finds the first index $i$ where $arr[i] > X$.   |
| **Last element $\le X$**  | `bisect_right(arr, X) - 1`   | Finds the boundary where values are $\le X$.    |

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

