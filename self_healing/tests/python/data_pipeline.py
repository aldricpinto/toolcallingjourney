import json
import os

def calculate_average_age(filepath):
    with open(filepath) as f:
        users = json.load(f)

    # BUG: age is nested under "profile", but we're accessing it at the top level
    total = sum(user["age"] for user in users)
    average = total / len(users)
    print(f"Average age: {average:.1f}")

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "users.json")
    calculate_average_age(data_path)
