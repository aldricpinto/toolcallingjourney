const users = [
    { name: "Alice", profile: { age: 30 } },
    { name: "Bob", profile: { age: 25 } },
    { name: "Charlie" },                      // BUG: 'profile' is missing on this user
    { name: "Diana", profile: { age: 28 } },
];

function calculateAverageAge(users) {
    // BUG: no guard for missing `profile` — will throw TypeError on Charlie
    const total = users.reduce((sum, user) => sum + user.profile.age, 0);
    const average = total / users.length;
    console.log(`Average age: ${average.toFixed(1)}`);
}

calculateAverageAge(users);
