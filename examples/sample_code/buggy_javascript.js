// Sample JavaScript code with intentional bugs for testing

// Undefined reference - BUG!
function getUserName(user) {
    return user.profile.name;
}

// Type coercion issues - BUG!
function addValues(a, b) {
    return a + b; // Could concatenate strings instead of adding
}

// Async without error handling - BUG!
async function fetchData(url) {
    const response = await fetch(url);
    return response.json();
}

// Memory leak - BUG!
function createLargeArray() {
    const cache = [];
    setInterval(() => {
        cache.push(new Array(1000000));
    }, 100);
}

// Prototype pollution - BUG!
function merge(target, source) {
    for (let key in source) {
        target[key] = source[key];
    }
    return target;
}

// Race condition - BUG!
let counter = 0;
function incrementCounter() {
    setTimeout(() => {
        counter = counter + 1;
    }, Math.random() * 100);
}

