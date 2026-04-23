"""Sample Python code with intentional bugs for testing."""


def divide_numbers(a, b):
    """Division without zero check - BUG!"""
    return a / b


def access_list(items, index):
    """List access without bounds check - BUG!"""
    return items[index]


def unsafe_file_read(filename):
    """File operation without exception handling - BUG!"""
    f = open(filename, 'r')
    content = f.read()
    return content


class UserManager:
    def __init__(self):
        self.users = {}
    
    def add_user(self, user_id, data):
        """No validation on user_id - BUG!"""
        self.users[user_id] = data
    
    def get_user(self, user_id):
        """KeyError possible - BUG!"""
        return self.users[user_id]


def calculate_average(numbers):
    """Division by zero if empty list - BUG!"""
    return sum(numbers) / len(numbers)

