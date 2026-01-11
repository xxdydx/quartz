
## overview
### structs vs. classes
- `struct` members are `public` by default, whereas `class` members are `private` by default.
- `struct` is typically used for **composition** (bundling data types), whereas `class` is used for **encapsulation** (hiding state behind behavior).

### inline functions
- Member functions defined _inside_ the class definition are implicitly `inline`.
- To satisfy the **One Definition Rule (ODR)**, functions defined in headers should be marked `inline`, though member functions inside the body get this for free.

### const functions
- A function marked `const` (e.g., `int get_energy() const`) guarantees it will not modify the object.
- Inside a `const` function, the `this` pointer becomes a `const Animal*` rather than `Animal*`.
- You cannot call a **non-const** member function on a `const` reference or object.

```cpp
int get_energy() const {
	return energy;
}
// Take a const reference to `anim`
const Animal& anim_ref = anim;

// Call a const member function
int energy = anim_ref.get_energy();
```

If const is subsequently removed from get_energy() this will result in a compile error.



## encapsulation

### access modifiers
- **Private:** Accessible only by the class itself.
- **Protected:** Accessible by the class and its derived classes (subclasses).
- **Public:** Accessible by anyone.
- **Note:** In C++, modifiers are applied to blocks of fields (e.g., `private:` applies to everything until the next label), unlike Java/C# where it is per-field.

### inheritance mechanics & memory layout
- When class `Cat` inherits from `Animal`, the fields of `Animal` are placed first in memory, followed immediately by `Cat`'s fields.
    - The compiler may insert padding bytes between fields or classes to satisfy alignment requirements (e.g., 3 bytes of padding after a `bool` to align the next `int`).
- **Inheritance Type:**
    - `public` Inheritance: Public members of Base stay Public in Derived.
    - `private` Inheritance: All members of Base become **private** in Derived. This effectively hides the inheritance relationship from the outside world.
    - If omitted, inheritance defaults to `private` for classes and `public` for structs.


### slicing
- If you assign a derived object (Cat) to a base variable (Animal) **by value**, the derived parts are truncated.
```cpp
Cat cat(10.0);
Animal anim = cat; // "Slicing" occurs here.
```

Only the `Animal` portion is copied. The `Cat` fields (like `cuddliness`) are lost .

