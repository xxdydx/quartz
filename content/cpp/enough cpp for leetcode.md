for leetcode, use `#include <bits/stdc++.h>`
### types: 
`int` (32 bit), `long` (64 bit), `double` (fp), `bool`, `char`
`size_t` is a unsigned integer type that is returned when calling `.size()`

### standard template library:
#### `vector<T>`:
- a dynamic array
- `push_back` amortised O(1), random access O(1), erase in the middle O(n)

```cpp
#include <vector>
using namespace std;

vector<int> a;                  // empty
a.push_back(5);                 // [5]
vector<int> b(3, 7);            // [7,7,7]
int x = a.back();               // last element, x = 5
a.pop_back();                   // remove last, a = []
size_t n = a.size();            // number of elements, n = 0
for (int i = 0; i < (int)a.size(); ++i) { /* a[i] */ } // runs 0 times
for (int v : b) { /* v */ }     // range-for
```

#### `string`:
- a `vector<char>` with helpers
```cpp
#include <string>
string s = "abba";
s.size(); s.empty();
s.push_back('c'); s.pop_back();
s.substr(pos, len);
size_t pos = s.find("ab");      // or string::npos if not found
```

#### `array<T, N>`:
- fixed size array, stack allocated so faster for small fixed tables, like letter counts

```cpp
#include <array>
array<int, 26> cnt{};   // zero initialised
cnt[0]++;               // for 'a'
```

#### `deque<T>`:
- for queue/stack based operations
```cpp
#include <bits/stdc++.h> 
using namespace std;

int main() {
    deque<int> dq;                 // dq: []

    dq.push_front(1);              // dq: [1]
    dq.push_back(2);               // dq: [1, 2]

    cout << dq.front() << "\n";    // expected: 1
    cout << dq.back()  << "\n";    // expected: 2

    dq.pop_front();                // dq: [2]
    dq.pop_back();                 // dq: []

    cout << dq.empty()  << "\n";   // expected: 1  (true means empty)

    return 0;
}
```

#### `stack<T>` and `queue<T>`:
- they wrap a container and restrict ops
```cpp
#include <stack>
stack<int> st;
st.push(3); 
st.top(); // 3
st.pop();

#include <queue>
queue<int> q;
q.push(5); 
q.front(); // 5
q.pop();
```

#### `priority_queue<T>`:
- heap, for top-k, dijkstra, scheduling questions
- default is max-heap; use greater<> for a min-heap

```cpp
#include <queue>
priority_queue<int> maxq; // 10, 8, 7...
priority_queue<int, vector<int>, greater<int>> minq; // 1, 2, 3...

minq.push(5); minq.top(); minq.pop();

// Custom key
priority_queue<
  pair<int,int>,                  // 1) the value type stored in the heap
  vector<pair<int,int>>,          // 2) the underlying container that holds elements, don't need to change this (just provide vector<T> where T is the 1st argument)
  greater<pair<int,int>>          // 3) the comparator that orders the heap
> pq;
```

#### Unordered containers: `unordered_map` or `unordered_set`:
- O(1) insert and find
```cpp
#include <unordered_map>
unordered_map<int,int> freq;
freq[5]++;                     // inserts 5 with 0 then increments
if (freq.count(7)) { /* present */ }

#include <unordered_set>
unordered_set<string> seen;
seen.insert("abc"); seen.count("abc"); // 1 or 0
```

#### Ordered containers: `map`, `set`, `multiset`:
- balanced trees, $O(\log{n})$ operations, deterministic ordering
```cpp
#include <set>
set<int> s; s.insert(5);
auto it = s.lower_bound(4);  // first element >= 4
if (it != s.end() && *it == 5) { /* found 5 */ }

#include <map>
map<int,int> mp;
mp[10] = 3;
for (auto [k, v] : mp) { /* ascending keys */ }

#include <multiset>
multiset<int> ms;
ms.insert(2); ms.insert(2);           // duplicates allowed
auto it2 = ms.find(2); if (it2 != ms.end()) ms.erase(it2); // erase one copy

```

#### Iterators:
- every container provides `begin()` and `end()`
- `[begin, end)` — start inclusive, end exclusive
```cpp
#include <vector>
#include <iostream>
using namespace std;

int main() {
    vector<int> v = {3, 1, 4};

    // 1) An iterator points at elements within [begin, end)
    auto it = v.begin();                 // points at the first element, value 3
    cout << "Initial *it: " << *it << "\n";
    ++it;                                // move to the next element
    cout << "After ++it: " << *it << "\n"; // now points at 1

    // 2) Manual iteration with iterators over the half-open range [begin, end)
    cout << "Manual loop: ";
    for (auto it = v.begin(); it != v.end(); ++it) {
        cout << *it << ' ';              // prints element values
    }
    cout << "\n";

    // 3) Range-for by value, reads elements using begin()/end() under the bonnet
    cout << "Range-for by value: ";
    for (int x : v) {                     // copies each element into x
        cout << x << ' ';
    }
    cout << "\n";

    // 4) Range-for by reference, modifies elements in place
    for (int &x : v) {                    // references the element, no copy
        x += 10;                          // mutate the vector itself
    }
    cout << "After modifying by reference: ";
    for (const int &x : v) cout << x << ' ';
    cout << "\n";

    // 5) Iterating a subrange: first two elements only, still half-open
    cout << "Subrange [begin, begin+2): ";
    for (auto it = v.begin(); it != v.begin() + 2; ++it) {
        cout << *it << ' ';
    }
    cout << "\n";

    // 6) Const context uses const_iterators
    const vector<int>& vc = v;
    cout << "Const iteration: ";
    for (auto it = vc.begin(); it != vc.end(); ++it) {
        // *it = 0; // would not compile, cannot write through const_iterator
        cout << *it << ' ';
    }
    cout << "\n";
}

```


### functions and parameter passing:

