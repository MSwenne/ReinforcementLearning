def get_input(message, valid):
    print(message)
    result = input()
    while result not in valid:
        print("Invalid value!")
        result = input()
    return result
