// Sample C++ code with intentional bugs for testing

#include <iostream>
#include <vector>
#include <string>

// Memory leak - BUG!
int* createArray(int size) {
    int* arr = new int[size];
    return arr;
}

// Buffer overflow - BUG!
void copyString(char* dest, const char* src) {
    while (*src) {
        *dest++ = *src++;
    }
    *dest = '\0';
}

// Use after free - BUG!
int* dangerousPointer() {
    int* ptr = new int(42);
    delete ptr;
    return ptr;
}

// Out of bounds access - BUG!
int getElement(std::vector<int>& vec, int index) {
    return vec[index];
}

// Null pointer dereference - BUG!
void processString(std::string* str) {
    std::cout << str->length() << std::endl;
}

// Division by zero - BUG!
double divide(double a, double b) {
    return a / b;
}

