## stack vs. heap

**Stack** — Local variables are allocated here; these variables are **automatically destroyed** when the function returns.

**Heap** — Allocated via `new`, objects here persist till they're explicitly destroyed. Every `new` must have a corresponding `delete`. Failure to delete causes memory leaks.

## pointers vs. references

| **Feature**     | **Pointer (int*)**                    | **Reference (int&)**                         |
| --------------- | ------------------------------------- | -------------------------------------------- |
| **Nullability** | Can be `nullptr`.                     | **Gotcha:** Cannot be uninitialized or null. |
| **Re-seating**  | Can change what it points to.         | Cannot be re-pointed after initialization.   |
| **Syntax**      | Requires dereferencing (`*ptr`).      | "Transparent"; used like the object itself.  |
| **Arithmetic**  | Supports arithmetic (e.g., `ptr + 1`) | No arithmetic                                |


```cpp
#include <iostream>

int main() {
    int a = 10;
    int b = 999;
    // ==========================================
    // 1. NULLABILITY & INITIALIZATION
    // ==========================================
    
    int* ptr = nullptr; // OK: Pointers can be null/empty.
    // int& ref;        // ERROR: References MUST be initialized when declared.
    
    int& ref = a;       // OK: 'ref' is now permanently an alias for 'a'
    ptr = &a;           // OK: 'ptr' now points to 'a'.
    // ==========================================
    // 2. SYNTAX & ACCESS
    // ==========================================
    // Pointer: Must explicitly "dereference" with * to get the value
    *ptr = 20;          // 'a' becomes 20.
    
    // Reference: "Transparent" access. No special symbols needed.
    ref = 30;           // 'a' becomes 30.

    // ==========================================
    // 3. RE-SEATING (The most common bug source)
    // ==========================================

    // Goal: Make the pointer/reference point to 'b' instead of 'a'.

    // POINTER: Successfully re-points
    ptr = &b;           
    *ptr = 50;          // Changes 'b'. 'a' is untouched.
                        // ptr is now address of b, ref is still alias for a.

    // REFERENCE: Fails to re-point! 
    ref = b;            // CRITICAL GOTCHA: This does NOT make ref refer to b.
                        // Instead, it copies b's value (50) into 'a'.
}
```


### examples of "dangling pointers"
this happens as local variables within a function get destroyed when the function returns (as they're stored in the stack).
```cpp
int& xyz()  // return by reference
{
    int x = 10;
    return x;  
}

int main()
{
    int& y = xyz();
    std::cout << y; 
}
```

this causes **undefined behaviour** (still compiles though).
- let's say `x` was stored in `0x1234` during the function run of `xyz()`
- after `xyz` returns, the function's stack frame is popped; the variable `x` goes out of scope and is destroyed.
- the return value of `xyz` is `0x1234`, but it's not guaranteed that the value 10 is in `0x1234` now. it could be a garbage value.


## arrays, structs & memory layout

**Array Decay** — When passing a raw array (`arr`) to a function, it decays to a pointer to `arr[0]`.
- Inside the function, `sizeof(array)` will return the size of the _pointer_ (usually 8 bytes), not the size of the array data. You must pass the size separately or use references `int (&arr)[N]`.

**Struct Padding & Alignment**
- The size of a struct is often **larger** than the sum of its parts.
- The compiler inserts "padding bytes" to ensure members align with memory boundaries (e.g., 4-byte integers sit on 4-byte addresses)
- Do not use `memcmp` to compare structs. It compares the padding bytes, which contain garbage data, leading to incorrect results.

**Unions**

All members share the same memory offset.
However, only one member is "active" at a time. Reading from an inactive member (e.g., writing to `float` then reading `int`) is **undefined behaviour**.
```cpp
box.y = 3.1; // y is active

// this is undefined behaviour!
std::cout << "box.x = " << box.x << "\n";
```